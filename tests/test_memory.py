"""長期記憶: 抽出結果の整形・保存・プロンプト注入と Director 結線 (設計書 3.4)。"""
import asyncio
import json
import threading

import numpy as np
import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk import memory
from idle_talk.clients import AsrClient
from idle_talk.director import IdleTalkDirector
from idle_talk.store import Store
from idle_talk.vad import EnergyVad
from idle_talk.voice_id import NAME_PROMPT, VoiceBook

FACT = "利用者は毎朝コーヒーを飲む。"


def add_speaker(store, name):
    return store.execute("INSERT INTO speakers(name, embedding, n, updated_at) VALUES (?, ?, 2, 0)",
                         (name, np.array([1, 0], dtype=np.float32).tobytes()))


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", db_backup=False)
    yield s
    s.close()


def test_parse_extracted_keeps_only_subject_lines():
    text = "- 利用者は猫を飼っている。\n2. 利用者は3月に引っ越す。\nなし\n今日は晴れです。\n\n- 利用者は猫を飼っている。"
    assert memory.parse_extracted(text) == ["利用者は猫を飼っている。", "利用者は3月に引っ越す。"]


def test_parse_extracted_limits():
    assert memory.parse_extracted("利用者は" + "あ" * memory.MAX_CHARS) == []
    lines = "\n".join(f"利用者は{i}が好き。" for i in range(5))
    assert len(memory.parse_extracted(lines)) == memory.MAX_PER_UTTERANCE


def test_remember_dedupes_and_updates(store):
    memory.remember(store, [FACT], 1, 100)
    assert memory.remember(store, ["利用者は毎朝 コーヒーを飲む"], 2, 200) == []
    rows = memory.list_memories(store)
    assert len(rows) == 1 and rows[0]["updated_at"] == 200 and rows[0]["evidence_utterance_id"] == 2


def test_remember_caps_oldest_deletable(store):
    memory.add_memory(store, "利用者は固定。", 1, 0, deletable=0)
    for i in range(memory.MAX_MEMORIES):
        memory.remember(store, [f"利用者は{i}番目。"], 1, 10 + i)
    assert store.count("memories") == memory.MAX_MEMORIES
    contents = [r["content"] for r in memory.list_memories(store)]
    assert "利用者は固定。" in contents and "利用者は0番目。" not in contents


def test_memories_are_per_persona(store):
    memory.add_memory(store, "利用者は共通。", 1, 1)  # v2 以前の記憶 (persona NULL) は全員に見える
    memory.remember(store, ["利用者は猫好き。"], 1, 10, "a")
    memory.remember(store, ["利用者は犬好き。"], 2, 20, "b")
    assert "猫" in memory.memory_prompt(store, "a") and "犬" not in memory.memory_prompt(store, "a")
    assert "犬" in memory.memory_prompt(store, "b") and "共通" in memory.memory_prompt(store, "b")
    assert memory.remember(store, ["利用者は猫好き"], 3, 30, "b") != []  # 別のペルソナには別に覚える
    for i in range(memory.MAX_MEMORIES):  # 上限はペルソナごと
        memory.remember(store, [f"利用者は{i}番目。"], 1, 100 + i, "a")
    assert "犬" in memory.memory_prompt(store, "b")
    assert len(memory.list_memories(store, "b")) == 3


def test_memory_prompt(store):
    assert memory.memory_prompt(store) == ""
    memory.remember(store, [FACT], 1, 100)
    assert f"- {FACT}" in memory.memory_prompt(store)


def test_memories_isolate_speakers_and_legacy_and_persist(store):
    a, b = add_speaker(store, "太郎"), add_speaker(store, "花子")
    region, like = "利用者は横浜に住んでいる。", "利用者は猫が好き。"
    memory.remember(store, ["利用者は昔の話者未割当。"], 1, 1, "a")
    memory.remember(store, [region, like], 2, 2, "a", a)
    memory.remember(store, [like], 3, 3, "a", b)  # 同じ好みも別の人の根拠として保存
    memory.remember(store, ["利用者は犬が好き。"], 4, 4, "b", a)
    assert region in memory.memory_prompt(store, "a", a)
    assert region not in memory.memory_prompt(store, "a", b)
    assert "未割当" not in memory.memory_prompt(store, "a", a)
    assert "犬" not in memory.memory_prompt(store, "a", a)
    assert "横浜" not in memory.memory_prompt(store, "a")
    assert len(store.query("SELECT id FROM memories WHERE content=?", (like,))) == 2
    again = Store(store.path, db_backup=False)
    try:
        assert region in memory.memory_prompt(again, "a", a)
        assert region not in memory.memory_prompt(again, "a", b)
    finally:
        again.close()


def test_speaker_limit_does_not_evict_another_person(store):
    a, b = add_speaker(store, "太郎"), add_speaker(store, "花子")
    memory.remember(store, [FACT], 1, 0, "a", b)
    for i in range(memory.MAX_MEMORIES + 1):
        memory.remember(store, [f"利用者は{i}が好き。"], 2, i + 1, "a", a)
    assert len(store.query("SELECT id FROM memories WHERE speaker_id=?", (a,))) == memory.MAX_MEMORIES
    assert FACT in memory.memory_prompt(store, "a", b)


def make(tmp_path, gate=None):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    chats: list[list[dict]] = []
    extracts: list[list[dict]] = []

    def llm(messages):
        if messages[0]["content"] == memory.EXTRACT_PROMPT:
            extracts.append(messages)
            if gate is not None:
                gate.wait(10)
            yield f"- {FACT}\n"
            return
        chats.append(messages)
        yield "いいですね。"

    asr = AsrClient(_asr(clock))
    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(), asr,
                         clock=clock, now_ms=lambda: 1000)
    b = MockBrowser()
    b.attach(d.core)
    return d, b, store, chats, extracts


def _asr(clock):
    def send(req, timeout):
        return 200, {"utterance_id": req["utterance_id"], "text": "毎朝コーヒーを飲んでいます",
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}
    return send


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def test_extracted_after_response_and_injected(tmp_path):
    async def run():
        d, b, store, chats, extracts = make(tmp_path)
        try:
            b.speak()
            await until(lambda: store.count("memories") == 1)
            assert FACT not in chats[0][0]["content"]  # 抽出は応答の再生完了後
            assert "毎朝コーヒーを飲んでいます" in extracts[0][1]["content"]
            assert d._mem_pending == []
            b.speak()
            await until(lambda: len(chats) == 2)
            assert FACT in chats[1][0]["content"]
        finally:
            store.close()
    asyncio.run(run())


def test_user_speech_cancels_extraction_and_retries(tmp_path):
    async def run():
        gate = threading.Event()
        d, b, store, chats, extracts = make(tmp_path, gate)
        try:
            b.speak()
            await until(lambda: len(extracts) == 1)
            b.speak()  # 抽出中に利用者が話す: 取消して結果は保存しない
            assert d._mem_task.cancelled() or d._mem_stop.is_set()
            gate.set()
            await until(lambda: len(extracts) == 2)  # 次の応答完了後に未処理分をまとめて再試行
            await until(lambda: store.count("memories") == 1)
            assert len(extracts[1][1]["content"].splitlines()) == 2
        finally:
            gate.set()
            store.close()
    asyncio.run(run())


def test_short_utterance_not_extracted(tmp_path):
    async def run():
        d, b, store, chats, extracts = make(tmp_path)
        d.ingress.asr = AsrClient(mock_asr_send(FakeClock()))  # 「こんにちは」(5 文字)
        try:
            b.speak()
            await until(lambda: len(chats) == 1 and d.fsm.ctx.state.value == "idle")
            await asyncio.sleep(0.05)
            assert extracts == [] and store.count("memories") == 0
        finally:
            store.close()
    asyncio.run(run())


def test_director_batches_by_received_speaker_not_current_speaker(tmp_path):
    async def run():
        d, _, store, _, extracts = make(tmp_path)
        try:
            a, b = add_speaker(store, "太郎"), add_speaker(store, "花子")
            d._speaker = b
            d._mem_pending = [(1, "横浜に住んでいます", a), (2, "猫が大好きです", b)]
            d._start_memory()
            await d._mem_task
            assert len(extracts) == 2
            assert "横浜" in extracts[0][1]["content"] and "猫" not in extracts[0][1]["content"]
            assert "猫" in extracts[1][1]["content"] and "横浜" not in extracts[1][1]["content"]
            assert [(r["speaker_id"], r["evidence_utterance_id"]) for r in memory.list_memories(store)] == [(a, 1), (b, 2)]
            assert d._mem_pending == []
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_unknown_voice_does_not_learn_or_use_legacy_memory(tmp_path):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        try:
            d.voices = VoiceBook(store)
            memory.remember(store, [FACT], 1, 1, d.personas.current_id)
            assert d._on_transcript(2, "横浜に住んでいます")
            assert d._mem_pending == []
            assert FACT not in d._chat_messages()[0]["content"]
            a, b = add_speaker(store, "太郎"), add_speaker(store, "花子")
            memory.remember(store, ["利用者は横浜に住んでいる。"], 3, 2, d.personas.current_id, a)
            d._speaker = a
            assert "横浜" in d._chat_messages()[0]["content"]
            d._speaker = b
            assert "横浜" not in d._chat_messages()[0]["content"]
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_background_follows_name_merge_for_pending_memory(tmp_path):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        try:
            a, b = add_speaker(store, "太郎"), add_speaker(store, None)
            d.voices = VoiceBook(store)
            d._speaker = d._ask_sid = b
            d._name_pending = ["太郎です。横浜に住んでいます"]
            d._mem_pending = [(1, d._name_pending[0], b)]
            d._llm = lambda messages: iter(["太郎" if messages[0]["content"] == NAME_PROMPT else FACT])
            d._start_memory()
            await d._mem_task
            assert d._speaker == a and d._mem_pending == []
            assert [r["speaker_id"] for r in memory.list_memories(store)] == [a]
        finally:
            d.close()
            store.close()
    asyncio.run(run())


def test_director_short_correction_is_extracted_and_used_in_next_response(tmp_path):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        seen = []
        mid = memory.add_memory(store, "利用者は猫が好き。", 1, 1, persona=d.personas.current_id)

        def llm(messages):
            seen.append(messages)
            yield json.dumps(dict(kind="speaker", id=mid, content="利用者は猫が嫌い。", evidence="猫は嫌い"))

        d._llm = llm
        try:
            assert d._on_transcript(2, "猫は嫌い")
            assert d._mem_pending == [(2, "猫は嫌い", None)]
            d._start_memory()
            await d._mem_task
            assert "利用者は猫が好き。" in seen[0][1]["content"]
            prompt = d._chat_messages()[0]["content"]
            assert "猫が嫌い" in prompt and "猫が好き" not in prompt
            assert d._mem_pending == []
        finally:
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("kind", ["speaker", "shared"])
def test_correction_cancelled_during_generation_is_only_applied_on_retry(tmp_path, kind):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        gate = threading.Event()
        started = threading.Event()
        old = "利用者は横浜に住む。" if kind == "speaker" else "作品の舞台は横浜。"
        new = "利用者は大阪に住む。" if kind == "speaker" else "作品の舞台は大阪。"
        if kind == "speaker":
            mid = memory.add_memory(store, old, 1, 1, persona=d.personas.current_id)
        else:
            mid, = memory.remember_knowledge(store, [old], 1, 1)

        def llm(messages):
            started.set()
            gate.wait(10)
            yield json.dumps(dict(kind=kind, id=mid, content=new, evidence="横浜ではなく大阪です"))

        d._llm = llm
        try:
            d._mem_pending = [(2, "横浜ではなく大阪です", None)]
            d._start_memory()
            await until(started.is_set)
            d._cancel_memory()
            gate.set()
            with pytest.raises(asyncio.CancelledError):
                await d._mem_task
            assert old in memory.memory_prompt(store, d.personas.current_id)
            assert new not in memory.memory_prompt(store, d.personas.current_id)
            assert len(d._mem_pending) == 1
            d._start_memory()
            await d._mem_task
            assert old not in memory.memory_prompt(store, d.personas.current_id)
            assert new in memory.memory_prompt(store, d.personas.current_id)
            assert d._mem_pending == []
        finally:
            gate.set()
            d.close()
            store.close()
    asyncio.run(run())


def test_unknown_voice_uses_shared_knowledge_without_personal_memories(tmp_path):
    d, _, store, _, _ = make(tmp_path)
    try:
        d.voices = VoiceBook(store)
        memory.remember(store, [FACT], 1, 1, d.personas.current_id)
        memory.remember_knowledge(store, ["この作品の舞台は大阪。"], 1, 1)
        prompt = d._chat_messages()[0]["content"]
        assert "この作品の舞台は大阪。" in prompt and FACT not in prompt
    finally:
        d.close()
        store.close()


def test_speaker_deleted_during_extraction_drops_pending_without_learning(tmp_path):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        sid = add_speaker(store, "太郎")

        def llm(messages):
            store.execute("DELETE FROM speakers WHERE id=?", (sid,))
            yield json.dumps(dict(kind="shared", id=None, content="作品の舞台は大阪。", evidence="作品の舞台は大阪です"))

        d._llm = llm
        try:
            d._mem_pending = [(2, "作品の舞台は大阪です", sid)]
            d._start_memory()
            await d._mem_task
            assert d._mem_pending == []
            assert store.count("knowledge") == store.count("memories") == 0
        finally:
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("merge", [False, True])
@pytest.mark.parametrize("has_memory", [False, True])
def test_spoken_name_correction_updates_database_and_voice_cache(tmp_path, merge, has_memory):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        sid = add_speaker(store, "太郎")
        kept = add_speaker(store, "花子") if merge else sid
        d.voices = VoiceBook(store)
        d._speaker = sid
        if has_memory:
            memory.remember(store, [FACT], 1, 1, d.personas.current_id, sid)
        seen = []

        def llm(messages):
            seen.append(messages)
            yield json.dumps(dict(kind="speaker_name", id=sid, content="花子", evidence="花子です"))

        d._llm = llm
        try:
            # 新規の記憶抽出より短い発話でも、名前の訂正は受け付ける。
            assert d._on_transcript(2, "花子です")
            assert d._mem_pending == [(2, "花子です", sid)]
            d._start_memory()
            await d._mem_task
            assert f'"id": {sid}, "name": "太郎"' in seen[0][1]["content"]
            assert d._speaker == kept and d.voices.name(kept) == "花子"
            assert "「花子」さん" in d._chat_messages()[0]["content"]
            assert [r["speaker_id"] for r in memory.list_memories(store)] == ([kept] if has_memory else [])
            assert d._mem_pending == []
            again = Store(store.path, db_backup=False)
            try:
                assert VoiceBook(again).name(kept) == "花子"
                assert [(r["id"], r["name"]) for r in again.query("SELECT id, name FROM speakers")] == [(kept, "花子")]
            finally:
                again.close()
        finally:
            d.close()
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("mutation", ["cancel", "rename", "delete", "switch"])
def test_name_correction_does_not_overwrite_changed_or_cancelled_speaker(tmp_path, mutation):
    async def run():
        d, _, store, _, _ = make(tmp_path)
        sid, other = add_speaker(store, "太郎"), add_speaker(store, "次郎")
        d.voices = VoiceBook(store)
        d._speaker = sid

        def llm(messages):
            if mutation == "cancel":
                d._mem_stop.set()
            elif mutation == "rename":
                store.execute("UPDATE speakers SET name='三郎' WHERE id=?", (sid,))
            elif mutation == "delete":
                store.execute("DELETE FROM speakers WHERE id=?", (sid,))
            elif mutation == "switch":
                d._speaker = other
            yield json.dumps(dict(kind="speaker_name", id=sid, content="花子", evidence="花子です"))

        d._llm = llm
        try:
            d._mem_pending = [(2, "太郎ではなく花子です", sid)]
            d._start_memory()
            await d._mem_task
            assert d.voices.name(other) == "次郎"
            if mutation == "delete":
                assert not store.query("SELECT id FROM speakers WHERE id=?", (sid,))
            else:
                expected = {"cancel": "太郎", "rename": "三郎", "switch": "花子"}[mutation]
                assert store.query("SELECT name FROM speakers WHERE id=?", (sid,))[0][0] == expected
            assert d._speaker == (other if mutation == "switch" else sid)
        finally:
            d.close()
            store.close()
    asyncio.run(run())
