"""割り込み系の結合試験 (§11.2 結合層): E1 / E2 / E3 / E15 / G3。模擬ブラウザ + 模擬 LLM/TTS/ASR。"""
import asyncio
import threading
import time

from mocks import FakeClock
from mocks_browser import MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.config import Config
from idle_talk.director import IdleTalkDirector
from idle_talk.fsm import Ev, Event
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

CFG = Config()
TOKENS = ("今日はいい天気ですね。", "散歩に行きましょう。")


def make(tmp_path, tokens=TOKENS, texts=None, gate=None, auto_playback=True, asr=True, asr_delay=0.0):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    texts = texts if texts is not None else []
    calls = {"n": 0}

    def llm(messages):
        calls["n"] += 1
        if gate is not None and calls["n"] == 1:
            gate.wait(10)
        yield from tokens

    def send(req, timeout):
        clock.advance(asr_delay)
        text = texts.pop(0) if texts else "こんにちは"
        return 200, {"utterance_id": req["utterance_id"], "text": text,
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(send, CFG, clock=clock) if asr else None, cfg=CFG, clock=clock,
                         now_ms=lambda: 1000)
    b = MockBrowser(auto_playback=auto_playback)
    b.attach(d.core)
    d.calls, d.clock_, d.texts_q = calls, clock, texts
    return d, b, store


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.005)


def state(d):
    return d.fsm.ctx.state.value


def ai_rows(store, completed):
    return store.query("SELECT * FROM turns WHERE role='ai' AND completed=?", (completed,))


# ---------------- E1 ----------------

def test_e1_silence_below_t_silence_then_continuation_gives_no_response(tmp_path):
    async def run():
        d, b, store = make(tmp_path, asr=False)
        try:
            b.speak()  # 「今日は〜けど、」
            assert state(d) == "eot_check"
            d.fsm.enqueue(Event(Ev.SILENCE, silence_ms=CFG.T_silence - 1, eot=True))  # 399ms
            d._pump()
            assert state(d) == "eot_check"
            b.speak()  # 続き: 終話確認が取り消され発話に戻る
            assert state(d) == "eot_check"
            d.fsm.enqueue(Event(Ev.SILENCE, silence_ms=CFG.T_silence - 1, eot=True))
            d._pump()
            await asyncio.sleep(0.1)
            assert d.calls["n"] == 0 and b.chunks == [] and b.of("text") == []
            assert state(d) == "eot_check"
        finally:
            store.close()
    asyncio.run(run())


def test_e1_silence_exactly_t_silence_confirms_eot(tmp_path):
    async def run():
        d, b, store = make(tmp_path, asr=False)
        try:
            b.speak()
            d.fsm.enqueue(Event(Ev.SILENCE, silence_ms=CFG.T_silence, eot=True))  # 400ms ちょうど
            d._pump()
            assert state(d) in ("preparing", "ai_playing", "idle")
            assert "preparing" in [m["state"] for m in b.of("state")]
            await until(lambda: len(b.chunks) == 2)
            assert d.calls["n"] == 1
        finally:
            store.close()
    asyncio.run(run())


# ---------------- E2 ----------------

def test_e2_backchannel_during_playback_does_not_cancel(tmp_path):
    async def run():
        n = CFG.TEST_N
        for i in range(n):
            sub = tmp_path / f"{i}"
            sub.mkdir()
            d, b, store = make(sub, auto_playback=False)
            try:
                b.speak()
                await until(lambda d=d: state(d) == "ai_playing" and d._gen_done)
                ep = d.epoch.current
                utt = b.chunks[0][1]
                b.playback(ep, utt, 0, "started", pos=100)
                before = d.history._pos[(ep, utt)][0]
                d.texts_q.append("うん")
                b.speak()  # 再生中の相槌: duck -> T_confirm 内に判定 -> resume
                assert b.of("cancel") == [] and len(b.of("duck")) == 1
                assert len(b.of("resume")) == 1 and state(d) == "ai_playing"
                assert d.epoch.current == ep
                b.playback(ep, utt, 0, "finished", pos=900)  # T_confirm 後に再生位置が進行
                assert d.history._pos[(ep, utt)][0] > before
                b.playback(ep, utt, 1, "finished", pos=1800)
                await until(lambda d=d, store=store: state(d) == "idle" and ai_rows(store, 1))
                assert len(ai_rows(store, 1)) == 1
            finally:
                store.close()
    asyncio.run(run())


def test_e2_backchannel_after_t_confirm_is_cancelled(tmp_path):
    async def run():
        d, b, store = make(tmp_path, auto_playback=False, asr_delay=CFG.T_confirm / 1000)
        try:
            b.speak()
            await until(lambda: state(d) == "ai_playing" and d._gen_done)
            d.texts_q.append("うん")
            b.speak()  # ASR が T_confirm 以降に返る -> 判断不能として cancel
            assert len(b.of("cancel")) == 1 and b.of("resume") == []
            await until(lambda: d._task is not None and d._task.done())
        finally:
            store.close()
    asyncio.run(run())


# ---------------- E3 ----------------

def test_e3_stop_word_ducks_within_g2_and_cancels(tmp_path):
    async def run():
        n = CFG.TEST_N
        within = 0
        for i in range(n):
            sub = tmp_path / f"{i}"
            sub.mkdir()
            d, b, store = make(sub, auto_playback=False)
            try:
                b.speak()
                await until(lambda d=d: state(d) == "ai_playing" and d._gen_done)
                ep = d.epoch.current
                stamps = {}
                orig = b.send_text

                def rec(s, orig=orig, stamps=stamps):
                    if '"duck"' in s and "duck" not in stamps:
                        stamps["duck"] = time.perf_counter()
                    orig(s)
                b.send_text = rec
                d.texts_q.append("待って")
                t0 = time.perf_counter()
                b.speak()
                assert "duck" in stamps
                within += (stamps["duck"] - t0) * 1000 <= 250  # G2: p95 250ms
                assert len(b.of("cancel")) == 1 and d.epoch.current == ep + 1
                assert d.clock_.t * 1000 <= CFG.T_confirm + CFG.G2_SUM  # 確定停止の上限 (模擬時計)
                await until(lambda d=d: d._task is not None and d._task.done())  # 停止語への応答生成が終わるまで待つ
            finally:
                store.close()
        assert within == n
    asyncio.run(run())


# ---------------- E15 ----------------

def test_e15_late_asr_llm_results_of_old_epoch_are_discarded(tmp_path):
    async def run():
        n = CFG.TEST_N
        old_played = 0
        discarded = 0
        for i in range(n):
            sub = tmp_path / f"{i}"
            sub.mkdir()
            gate = threading.Event()
            d, b, store = make(sub, gate=gate, asr_delay=1.5)  # ASR 1500ms 遅延 (模擬時計)
            try:
                b.speak()  # 発話 A
                await until(lambda d=d: state(d) == "preparing")
                await asyncio.sleep(0.03)  # LLM スレッドが 1500ms 遅延(ゲート)に入るまで
                d.clock_.advance(0.5)
                a_epoch = d.epoch.current
                b.speak()  # A の 500ms 後に発話 B (epoch+1)
                assert d.epoch.current == a_epoch + 1
                gate.set()  # A の LLM 結果が遅れて到着
                await until(lambda d=d, store=store: state(d) == "idle" and ai_rows(store, 1))
                old_played += sum(1 for c in b.chunks if c[0] == a_epoch)
                discarded += d.pipeline.discarded
            finally:
                gate.set()
                store.close()
        assert old_played == 0
        assert discarded == n
    asyncio.run(run())


# ---------------- G3 ----------------

def test_g3_old_epoch_chunks_after_cancel_are_never_played(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        try:
            played_old = 0
            for i in range(100):
                ev = threading.Event()

                def clauses(ev=ev):
                    yield "あ。"
                    ev.wait(5)
                    yield "い。"
                    yield "う。"

                old = d.epoch.current
                n0 = len(b.chunks)
                task = asyncio.create_task(d.pipeline.play_utterance(clauses()))
                await until(lambda b=b, n0=n0: len(b.chunks) == n0 + 1)
                await d.pipeline.cancel()
                ev.set()
                await task
                d._on_chunk(old, 999, 0, "注入", b"RIFF....WAVE")  # 旧 epoch チャンクの直接注入
                played_old += sum(1 for c in b.chunks[n0 + 1:] if c[0] != d.epoch.current)
                assert len(b.chunks) == n0 + 1
            assert played_old == 0 and d.epoch.current == 100
        finally:
            store.close()
    asyncio.run(run())
