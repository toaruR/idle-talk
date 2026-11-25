"""居住地の自然な質問、短い返事と通常会話からの保存。"""
import asyncio
import json
import threading

import pytest
from mocks import FakeClock, mock_asr_send
from test_memory import add_speaker, make, until

from idle_talk import memory
from idle_talk import topics
from idle_talk.clients import AsrClient
from idle_talk.residence import ASKED_KIND, ASK_GAP_MS, QUESTIONS, REPLY_WINDOW_MS, ResidenceQuestions, residence
from idle_talk.store import Store
from idle_talk.voice_id import VoiceBook


def director(tmp_path):
    d, b, store, chats, extracts = make(tmp_path)
    sid = add_speaker(store, "太郎")
    d.voices = VoiceBook(store)
    d._speaker = sid
    d.residence_questions.rand = lambda: 0
    return d, b, store, sid


def operation(text="横浜"):
    return json.dumps(dict(kind="speaker", id=None, content="利用者は横浜に住んでいる。", evidence=text))


def test_question_probability_and_persistent_daily_cooldown(tmp_path):
    d, _, store, sid = director(tmp_path)
    try:
        q = ResidenceQuestions(rand=lambda: 0.1)
        assert q.choose(store, "a", sid, 1000) == ""
        q.rand = lambda: 0.099
        assert q.choose(store, "a", sid, 1000) in QUESTIONS
        q.heard(store, sid, 1000, 0)
        again = Store(store.path, db_backup=False)
        try:
            restarted = ResidenceQuestions(rand=lambda: 0)
            assert restarted.choose(again, "a", sid, 1000 + ASK_GAP_MS - 1) == ""
            assert restarted.choose(again, "a", sid, 1000 + ASK_GAP_MS) in QUESTIONS
            other = add_speaker(store, "花子")
            assert restarted.choose(again, "a", other, 1000) in QUESTIONS
        finally:
            again.close()
    finally:
        d.close()
        store.close()


def test_heard_question_then_short_city_reply_is_saved_and_persists(tmp_path):
    async def run():
        d, b, store, sid = director(tmp_path)
        seen = []
        try:
            b.speak()
            await until(lambda: d._awaiting_residence())
            if d._mem_task:
                await d._mem_task
            assert [m["text"] for m in b.of("text")][-1] in QUESTIONS
            assert store.query("SELECT COUNT(*) FROM events WHERE kind=?", (ASKED_KIND,))[0][0] == 1

            def extract(messages):
                seen.append(messages)
                yield operation()
            d._llm = extract
            d.ingress.asr = AsrClient(mock_asr_send(FakeClock(), body={
                "utterance_id": 2, "text": "横浜", "no_speech_prob": 0.1, "compression_ratio": 1.0}))
            b.speak()
            await until(lambda: residence(store, d.personas.current_id, sid) == "横浜")
            assert "居住地を尋ねた直後の本人の返事" in seen[0][1]["content"]
            row = store.query("SELECT speaker_id, persona, evidence_utterance_id FROM memories WHERE content=?",
                              ("利用者は横浜に住んでいる。",))[0]
            assert tuple(row) == (sid, d.personas.current_id, 2)
            assert d._residence_answers == {} and not d._awaiting_residence()
            assert d._plan_residence_question() == ""
            again = Store(store.path, db_backup=False)
            try:
                assert residence(again, d.personas.current_id, sid) == "横浜"
            finally:
                again.close()
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_spontaneous_mention_of_residence_is_saved_without_question(tmp_path):
    async def run():
        d, _, store, sid = director(tmp_path)
        try:
            d._llm = lambda messages: iter([operation("横浜に住んでいる")])
            assert d._on_transcript(9, "そういえば、横浜に住んでいるんだよ")
            assert d._plan_residence_question() == ""  # 教えてくれた返答で聞き返さない。
            d._start_memory()
            await d._mem_task
            assert residence(store, d.personas.current_id, sid) == "横浜"
            assert not store.query("SELECT id FROM events WHERE kind=?", (ASKED_KIND,))
            assert store.query("SELECT evidence_utterance_id FROM memories")[0][0] == 9
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_spontaneous_question_counts_only_after_playback_and_keeps_news_unused(tmp_path):
    async def run():
        d, b, store, sid = director(tmp_path)
        try:
            cid = topics.add_card(store, "新しい公園が開業")
            d.spontaneous._minutes = lambda: 12 * 60
            d.spontaneous.enabled = True
            d.clock.advance(1200)
            assert d.spontaneous_tick() == "spontaneous"
            assert not d._awaiting_residence()
            await until(lambda: not d.spontaneous.pending)
            assert d._awaiting_residence()
            assert [m["text"] for m in b.of("text")][-1] in QUESTIONS
            assert store.query("SELECT COUNT(*) FROM events WHERE kind=?", (ASKED_KIND,))[0][0] == 1
            assert store.query("SELECT used_at FROM topic_cards WHERE id=?", (cid,))[0][0] is None
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_declining_to_answer_does_not_save_residence_or_trigger_another_question(tmp_path):
    async def run():
        d, _, store, sid = director(tmp_path)
        try:
            d.residence_questions.heard(store, sid, d.now_ms(), d.epoch.current)
            d._residence_reply = (sid, d.personas.current_id, d.now_ms() + REPLY_WINDOW_MS)
            d._llm = lambda messages: iter(["なし"])
            assert d._on_transcript(3, "教えたくない")
            d._start_memory()
            await d._mem_task
            assert residence(store, d.personas.current_id, sid) is None
            assert d._plan_residence_question() == ""
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_cancelled_extraction_keeps_short_answer_context_for_retry(tmp_path):
    async def run():
        d, _, store, sid = director(tmp_path)
        started, release = threading.Event(), threading.Event()

        def extract(messages):
            started.set()
            assert release.wait(5)
            yield operation()

        try:
            d._residence_reply = (sid, d.personas.current_id, d.now_ms() + REPLY_WINDOW_MS)
            assert d._on_transcript(2, "横浜")
            d._llm = extract
            d._start_memory()
            await until(started.is_set)
            d._cancel_memory()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await d._mem_task
            assert residence(store, d.personas.current_id, sid) is None
            assert d._residence_answers == {2: sid}
            d._start_memory()
            await d._mem_task
            assert residence(store, d.personas.current_id, sid) == "横浜"
            assert d._residence_answers == {}
        finally:
            release.set()
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["unknown", "pc", "drama", "screen", "radio", "name", "known_place", "question"])
def test_question_skipped_when_unsuitable(tmp_path, reason):
    d, _, store, sid = director(tmp_path)
    try:
        if reason == "unknown":
            d._speaker = None
        elif reason in ("pc", "drama", "screen"):
            setattr(d, reason + "_on", True)
        elif reason == "radio":
            d.spontaneous.radio = True
        elif reason == "name":
            d._ask_plan = sid
        elif reason == "known_place":
            memory.add_memory(store, "利用者は横浜に住んでいる。", 1, 1, speaker_id=sid)
        else:
            store.add_turn(1, "ai", "今日は何をしたの？", 1)
        assert d._plan_residence_question() == ""
    finally:
        d.close()
        store.close()


def test_cancelled_or_unplayed_question_does_not_count_as_asked(tmp_path):
    async def run():
        d, _, store, sid = director(tmp_path)
        try:
            assert d._plan_residence_question() in QUESTIONS
            d._cancel_path("interrupt")
            assert d._residence_plan is None and not d._awaiting_residence()
            assert not store.query("SELECT id FROM events WHERE kind=?", (ASKED_KIND,))
            assert d._plan_residence_question() in QUESTIONS
            turn_id = store.add_turn(1, "ai", "返答だけ再生された。", 1)
            d._residence_question_heard(turn_id)
            assert not d._awaiting_residence()
            assert not store.query("SELECT id FROM events WHERE kind=?", (ASKED_KIND,))
        finally:
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("reason", ["expired", "different_speaker", "different_persona", "never_asked"])
def test_short_place_without_matching_recent_question_is_not_a_residence_reply(tmp_path, reason):
    d, _, store, sid = director(tmp_path)
    try:
        d._residence_reply = (sid, d.personas.current_id, 1000 + REPLY_WINDOW_MS)
        if reason == "expired":
            d.now_ms = lambda: 1001 + REPLY_WINDOW_MS
        elif reason == "different_speaker":
            d._speaker = None
        elif reason == "different_persona":
            d.personas.current_id = "other"
        else:
            d._residence_reply = None
        assert not d._on_transcript(3, "横浜")
        assert d._residence_answers == {}
    finally:
        d.close()
        store.close()


@pytest.mark.parametrize("content,kind,evidence,expected", [
    ("利用者は横浜に住んでいる。", "speaker", "横浜", True),
    ("利用者は大阪に住んでいる。", "speaker", "横浜", False),
    ("利用者はコーヒー好き。", "speaker", "コーヒー", False),
    ("利用者は横浜に住んでいる。", "shared", "横浜", False),
])
def test_only_grounded_residence_is_allowed_as_short_reply(tmp_path, content, kind, evidence, expected):
    d, _, store, sid = director(tmp_path)
    try:
        op = json.dumps(dict(kind=kind, id=None, content=content, evidence=evidence))
        memory.remember_extracted(store, op, [(4, evidence)], [], 1, "a", sid)
        assert store.count("memories") == 0
        memory.remember_extracted(store, op, [(4, evidence)], [], 1, "a", sid, {4})
        assert store.count("memories") == int(expected)
    finally:
        d.close()
        store.close()
