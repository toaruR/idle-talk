"""ドラマモード (設計書 6.5): 共有したタブの音を聞き、場面の切れ目にたまに一言。"""
import asyncio
import json

import pytest
from mocks import FakeClock
from mocks_browser import LOUD, QUIET, MockBrowser

from idle_talk import director as director_mod
from idle_talk.clients import AsrClient
from idle_talk.config import Drama, load_drama
from idle_talk.director import DRAMA_ECHO_KIND, IdleTalkDirector
from idle_talk.drama import DramaListener
from idle_talk.protocol import DRAMA_FLAG, is_drama, pack_up_frame, validate_message
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

SETTINGS = Drama(interval_s=60, gap_ms=2000, window_s=120, min_chars=10, ctx_chars=50, max_seg_s=3)


def listener(clock, **kw):
    return DramaListener(SETTINGS._replace(**kw), EnergyVad(), 200, clock)


def feed(dl, frames):
    return [seg for f in frames for seg in [dl.feed(f)] if seg is not None]


# ---- 設定・プロトコル ----
def test_drama_flag_and_ui_schema():
    assert is_drama(DRAMA_FLAG | 5) and not is_drama(5) and not is_drama(DRAMA_FLAG - 1)
    assert validate_message({"t": "ui", "drama": True})
    assert not validate_message({"t": "ui", "drama": 1})


def test_load_drama(tmp_path):
    p = tmp_path / "s.toml"
    assert load_drama(p) == Drama()
    p.write_text("[drama]\ninterval_s = 10\ngap_ms = 30000\n", encoding="utf-8")
    assert load_drama(p)[:2] == (10, 30000)
    for bad in ("interval_s = 9", "gap_ms = 2.5", "max_seg_s = 61", "foo = 1"):
        p.write_text(f"[drama]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_drama(p)


# ---- DramaListener ----
def test_segments_and_max_seg_split():
    clock = FakeClock()
    dl = listener(clock)
    segs = feed(dl, [LOUD] * 15 + [QUIET] * 12)
    assert len(segs) == 1 and dl._buf is None
    clock.advance(1)
    assert dl.quiet_ms() == 1000
    segs = feed(dl, [LOUD] * 200)  # 4s 鳴りっぱなし: max_seg_s (3s) で切る
    assert len(segs) == 1 and len(segs[0]) == 3 * 1000 * 32
    assert dl.quiet_ms() == 0.0  # 区間の途中
    assert feed(dl, [LOUD] * 3 + [QUIET] * 1) == []


def test_short_segment_dropped():
    dl = DramaListener(SETTINGS, EnergyVad(), 400, FakeClock())
    assert feed(dl, [LOUD] * 5 + [QUIET] * 12) == []  # 100ms + hangover 180ms < min_seg_ms
    assert dl._buf is None and dl.quiet_ms() == 0.0  # 区間としては終わっている (台詞の切れ目は測る)


def test_window_and_ctx_chars():
    clock = FakeClock()
    dl = listener(clock)
    dl.add_line("最初の台詞です")
    clock.advance(121)
    dl.add_line("あとの台詞")
    assert dl.context_text() == "あとの台詞"  # window_s を過ぎた台詞は捨てる
    for i in range(10):
        dl.add_line(f"台詞{i}番目です")
    text = dl.context_text()
    assert len(text) <= 50 and text.endswith("台詞9番目です")  # ctx_chars に収まるよう古い方から捨てる
    dl.add_line("…")  # 語の無い認識結果は足さない
    assert dl.lines[-1][1] == "台詞9番目です"


def test_should_react_boundaries():
    clock = FakeClock()
    dl = listener(clock)
    dl.start()
    assert not dl.should_react()  # 何も聞いていない
    feed(dl, [LOUD] * 15 + [QUIET] * 12)
    dl.add_line("一二三四五六七八九十")  # min_chars ちょうど
    clock.advance(59.9)
    assert not dl.should_react()  # interval_s 未満
    clock.advance(0.1)
    assert dl.should_react()  # interval_s ちょうど・gap_ms 超
    dl.mark_reacted()
    clock.advance(60)
    assert not dl.should_react()  # 新しい台詞が無い
    dl.add_line("一二三四五六七八九")
    assert not dl.should_react()  # min_chars 未満
    dl.add_line("十")
    feed(dl, [LOUD] * 15 + [QUIET] * 12)
    clock.advance(1.999)
    assert not dl.should_react()  # gap_ms 未満
    clock.advance(0.001)
    assert dl.should_react()


def test_contains_tolerates_asr_differences():
    dl = listener(FakeClock())
    dl.add_line("お前がやったんだろう、正直に言え。")
    assert dl.contains("お前がやったんだろ")
    assert not dl.contains("今日の晩ご飯なに？")
    assert not dl.contains("お前")  # 短すぎる


def test_stop_clears_lines():
    dl = listener(FakeClock())
    dl.add_line("台詞です")
    dl.stop()
    assert dl.context_text() == "" and dl.quiet_ms() == float("inf")


# ---- Director ----
def make(tmp_path, replies, texts):
    """replies: LLM の応答 (順に)。texts: ASR の認識結果 (ドラマ・マイク共通で順に)。"""
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    seen: list[list[dict]] = []

    def llm(messages):
        seen.append(messages)
        yield replies.pop(0) if replies else "うん。"

    def send(req, timeout):
        return 200, {"utterance_id": req["utterance_id"], "text": texts.pop(0),
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(send, clock=clock), clock=clock, now_ms=lambda: 1000)
    d.drama = DramaListener(SETTINGS, EnergyVad(), 200, clock)
    d.spontaneous._minutes = lambda: 12 * 60
    b = MockBrowser()
    b.attach(d.core)
    d.clock_, d.seen_ = clock, seen
    return d, b, store


def drama_speak(b, loud=15, quiet=12):
    for f in [LOUD] * loud + [QUIET] * quiet:
        b.core.on_binary(b, pack_up_frame(DRAMA_FLAG | b.seq, f))
        b.seq += 1


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def drama_on(d, b):
    d.core.on_text(b, json.dumps({"t": "ui", "drama": True}))
    assert d.drama_on


def test_drama_audio_never_drives_fsm_and_ignored_when_off(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, [], ["犯人はお前だ、そうだろう。"])
        drama_speak(b)
        assert d.fsm.ctx.state.value == "idle" and not d.drama.lines  # オフ: 捨てる
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        assert d.fsm.ctx.state.value == "idle" and not b.of("duck") and d._task is None
        assert d.drama.lines[-1][1] == "犯人はお前だ、そうだろう。"
        d.core.on_text(b, json.dumps({"t": "ui", "drama": False}))
        assert not d.drama_on and not d.drama.lines
    asyncio.run(run())


def test_reaction_plays_via_spontaneous_path(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["え、まさかの展開！"], ["犯人はお前だ、そうだろう。"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(59)
        assert d.spontaneous_tick() is None
        d.clock_.advance(1)
        assert d.spontaneous_tick() == "drama"
        await until(lambda: not d.spontaneous.pending)
        assert len(b.chunks) == 1 and ("spontaneous", "play_utterance") in d.path_calls
        assert [m["role"] for m in d.seen_[0]] == ["system", "user"]  # 履歴は渡さない
        assert "犯人はお前だ" in d.seen_[0][1]["content"]  # 今の場面の台詞を見せている
        ai = store.query("SELECT text FROM turns WHERE role='ai'")
        assert [r[0] for r in ai] == ["え、まさかの展開！"]
        assert d.spontaneous_tick() is None  # 新しい台詞が無いので続けて言わない
    asyncio.run(run())


def test_nothing_to_say_stays_silent(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["なし"], ["犯人はお前だ、そうだろう。"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        await until(lambda: not d.spontaneous.pending)
        assert b.chunks == [] and d.drama._new_chars == 0
        assert store.query("SELECT COUNT(*) FROM events WHERE kind='drama_skip'")[0][0] == 1
    asyncio.run(run())


def test_dialogue_resumed_during_generation_waits_for_next_gap(tmp_path):
    """生成中に台詞が再開したら捨てずに次の切れ目で言う (gap_ms が短いとほぼ毎回再開する。実機 2026-10-14)。"""
    async def run():
        d, b, _ = make(tmp_path, ["びっくりした！\n二文目は読まない。"], ["犯人はお前だ、そうだろう。", "続きの台詞"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        drama_speak(b, quiet=0)  # 生成中に台詞が再開した (区間の途中)
        await asyncio.sleep(0.3)
        assert b.chunks == [] and d.spontaneous.pending  # かぶせず待っている
        drama_speak(b, loud=0)  # 台詞が終わり
        d.clock_.advance(2)  # 切れ目 (gap_ms) が来た
        await until(lambda: not d.spontaneous.pending)
        assert len(b.chunks) == 1 and b.of("text")[0]["text"] == "びっくりした！"
    asyncio.run(run())


def test_no_gap_within_hold_stays_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(director_mod, "DRAMA_HOLD_S", 0.3)
    async def run():
        d, b, store = make(tmp_path, ["びっくりした！"], ["犯人はお前だ、そうだろう。"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        drama_speak(b, quiet=0)  # 台詞が続いたまま切れ目が来ない
        await until(lambda: not d.spontaneous.pending)
        assert b.chunks == []
        assert store.query("SELECT COUNT(*) FROM events WHERE kind='drama_skip'")[0][0] == 1
    asyncio.run(run())


def test_mic_picking_up_drama_is_discarded(tmp_path):
    async def run():
        d, b, store = make(tmp_path, [], ["お前がやったんだろう、正直に言え。", "お前がやったんだろ"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        b.speak()  # スピーカーの台詞をマイクが拾った
        await until(lambda: d.ingress.asr_pending == 0)
        await asyncio.sleep(0.05)
        assert d._task is None and d.fsm.ctx.state.value == "idle"
        assert store.query(f"SELECT COUNT(*) FROM events WHERE kind='{DRAMA_ECHO_KIND}'")[0][0] == 1
    asyncio.run(run())


def test_user_reply_sees_dialogue(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, ["たぶん兄の方だと思うよ。"], ["犯人はお前だ、そうだろう。", "今のって誰が犯人なの"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        b.speak()
        await until(lambda: d.seen_)
        assert "犯人はお前だ" in d.seen_[0][0]["content"]
    asyncio.run(run())


def test_no_idle_timeout_or_topic_talk_while_watching(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, [], [])
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        drama_on(d, b)
        d.clock_.advance(3600)
        assert d.spontaneous_tick() is None and d.fsm.ctx.state.value == "idle"
        d.core.on_text(b, json.dumps({"t": "ui", "drama": False}))
        assert d.spontaneous_tick() == "spontaneous"  # やめたら元どおり (操作で無操作の計測は戻る)
        d.clock_.advance(1800)
        assert d.spontaneous_tick() == "rest"
    asyncio.run(run())


def test_scene_text_only_since_last_reaction():
    clock = FakeClock()
    dl = listener(clock)
    dl.start()
    dl.add_line("前の場面の台詞")
    clock.advance(1)
    dl.mark_reacted()
    dl.add_line("今の場面の台詞")
    assert dl.scene_text() == "今の場面の台詞"
    assert dl.context_text() == "前の場面の台詞\n今の場面の台詞"  # 利用者への返答には前の台詞も渡す


def test_second_reaction_sees_only_new_scene_without_history(tmp_path):
    """前の一言や古い台詞を見せると 4B は同じ語を繰り返すので、2 回目は新しい台詞だけを渡す。"""
    async def run():
        d, b, _ = make(tmp_path, ["噛み詰めって痛そう。", "えっ、逃げた！"],
                       ["噛み詰めにしてやる、覚悟しろ。", "待て、どこへ行くんだ。"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        await until(lambda: not d.spontaneous.pending)
        drama_speak(b)
        await until(lambda: len(d.drama.lines) == 2)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        await until(lambda: not d.spontaneous.pending)
        second = "\n".join(m["content"] for m in d.seen_[1])
        assert "待て、どこへ行くんだ" in second
        assert "噛み詰め" not in second
    asyncio.run(run())


def test_unheard_reaction_stays_silent(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, ["この台詞、聞こえなくて何が言いたいのか分かりません。"], ["犯人はお前だ、そうだろう。"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        await until(lambda: not d.spontaneous.pending)
        assert b.chunks == []
    asyncio.run(run())


def test_digest_and_finale_material():
    clock = FakeClock()
    dl = listener(clock, digest_chars=10, finale_gap_s=30, finale_min_chars=10, window_s=30)
    dl.start()
    assert not dl.should_finale()  # 何も聞いていない
    dl.add_line("一二三四五")
    assert not dl.should_digest()
    dl.add_line("六七八九十")
    assert dl.should_digest()  # digest_chars ちょうど
    dl.add_note("兄が弟を疑った。", 2)
    assert dl.undigested == [] and dl.notes == ["兄が弟を疑った。"]
    dl.add_line("最後の台詞")
    clock.advance(31)  # window_s を過ぎて台詞は捨てても、メモと未要約の台詞は残る
    assert dl.context_text() == ""
    assert dl.finale_material() == ("兄が弟を疑った。", "最後の台詞")
    clock.advance(-1.001)
    assert not dl.should_finale()  # finale_gap_s 未満
    clock.advance(0.001)
    assert dl.should_finale()  # ちょうど
    dl.mark_finale()
    assert not dl.should_finale() and dl.notes == [] and dl.undigested == []


def test_finale_needs_enough_dialogue():
    clock = FakeClock()
    dl = listener(clock, finale_gap_s=30, finale_min_chars=11)
    dl.add_line("一二三四五六七八九十")
    clock.advance(30)
    assert not dl.should_finale()  # finale_min_chars 未満 (少し聞いて止めただけ)
    dl.add_line("十一")
    clock.advance(30)
    assert dl.should_finale()


def test_notes_capped():
    dl = listener(FakeClock())
    for i in range(30):
        dl.add_note(f"{i:02d}" + "あ" * 98, 0)
    assert sum(map(len, dl.notes)) <= 1200 and dl.notes[-1].startswith("29")


def test_finale_after_long_silence(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["兄が弟を疑った。", "最後は泣けたね。あなたはどうだった？"],
                           ["犯人はお前だ、そうだろう。", "違う、俺じゃない。"])
        d.drama = DramaListener(SETTINGS._replace(digest_chars=10, finale_gap_s=120, finale_min_chars=10),
                                EnergyVad(), 200, d.clock_)
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        assert d.spontaneous_tick() is None  # 台詞が溜まったので裏でメモに要約する
        await until(lambda: d.drama.notes)
        assert d.drama.notes == ["兄が弟を疑った。"] and d.drama.undigested == []
        assert [m["role"] for m in d.seen_[0]] == ["user"]  # 要約に人物設定は使わない
        drama_speak(b)
        await until(lambda: len(d.drama.lines) == 2)
        d.drama.mark_reacted()  # 一言は言い終えたことにする
        d.clock_.advance(119)
        assert d.spontaneous_tick() is None
        d.clock_.advance(1)
        assert d.spontaneous_tick() == "drama_finale"
        await until(lambda: not d.spontaneous.pending)
        prompt = d.seen_[1][-1]["content"]
        assert "兄が弟を疑った" in prompt and "違う、俺じゃない" in prompt and "犯人はお前だ" not in prompt
        ai = store.query("SELECT text FROM turns WHERE role='ai'")
        assert [r[0] for r in ai] == ["最後は泣けたね。あなたはどうだった？"]
        d.clock_.advance(600)
        assert d.spontaneous_tick() is None  # 新しい台詞を聞くまで繰り返さない
    asyncio.run(run())


def test_drama_off_cancels_pending_reaction(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, ["びっくりした！"], ["犯人はお前だ、そうだろう。"])
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "drama"
        d.core.on_text(b, json.dumps({"t": "ui", "drama": False}))
        await asyncio.sleep(0.2)
        assert not d.spontaneous.pending and b.chunks == []
    asyncio.run(run())
