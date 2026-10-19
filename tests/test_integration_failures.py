"""障害系の結合試験 (§11.2 結合層 / §12.3): E4 / E5 / E6 / E7 / E8 / cancel と完了の競合。"""
import asyncio
import threading
import time

from mocks import FakeClock
from mocks_browser import QUIET, MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.config import Config
from idle_talk.director import IdleTalkDirector
from idle_talk.errors import ERR_LLM_TIMEOUT, ERR_MIC_LOST
from idle_talk.fsm import Ev, Event
from idle_talk.llm_supervisor import FALLBACK_PHRASE, LlmSupervisor
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

CFG = Config()
TOKENS = ("今日はいい天気ですね。", "散歩に行きましょう。")


def make(tmp_path, llm=None, tts=None, cancel_tts=None, texts=None, auto_playback=True, asr_body=None):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    texts = texts if texts is not None else []
    calls = {"llm": 0}

    def default_llm(messages):
        calls["llm"] += 1
        yield from TOKENS

    def send(req, timeout):
        body = asr_body or {"utterance_id": req["utterance_id"],
                            "text": texts.pop(0) if texts else "こんにちは",
                            "no_speech_prob": 0.1, "compression_ratio": 1.0}
        return 200, body

    d = IdleTalkDirector(store, llm or default_llm, tts or (lambda t: b"RIFF....WAVE"),
                         cancel_tts or (lambda: None), EnergyVad(), AsrClient(send, CFG, clock=clock),
                         cfg=CFG, clock=clock, now_ms=lambda: 1000)
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


def gen_done(d):
    return d._task is not None and d._task.done()


# ---------------- E4 ----------------

def test_e4_silence_input_60s_gives_no_output(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        try:
            base = len(b.texts)
            for _ in range(CFG.E_PARAMS.silence_input_s * 50):  # 60s @ 20ms (模擬時計は進めない)
                b.send_frame(QUIET)
            await asyncio.sleep(0.05)
            assert len(b.texts) == base and b.chunks == [] and d.calls["llm"] == 0
        finally:
            store.close()
    asyncio.run(run())


def test_e4_hallucination_gives_no_response(tmp_path):
    async def run():
        body = {"utterance_id": 1, "text": "ご視聴ありがとうございました", "no_speech_prob": 0.9,
                "compression_ratio": 1.0}
        d, b, store = make(tmp_path, asr_body=body)
        try:
            b.speak()
            await asyncio.sleep(0.05)
            assert b.chunks == [] and b.of("text") == [] and d.calls["llm"] == 0
        finally:
            store.close()
    asyncio.run(run())


# ---------------- E5 ----------------

def test_e5_ws_drop_detected_within_two_pings_and_old_audio_is_silent(tmp_path):
    async def run():
        gate = threading.Event()
        n = {"c": 0}

        def llm(messages):
            n["c"] += 1
            if n["c"] == 1:
                gate.wait(10)
            yield from TOKENS

        d, b, store = make(tmp_path, llm=llm)
        try:
            b.speak()
            await until(lambda: state(d) == "preparing")
            await asyncio.sleep(0.03)
            old = d.epoch.current
            ping = CFG.PING_INTERVAL
            d.clock_.advance(ping)
            assert d.core.tick() is False  # 1 回目の無応答
            d.clock_.advance(ping)
            assert d.core.tick() is True  # PING_INTERVAL x 2 以内に切断検知
            assert d.core.conn is None and b.closed
            dropped_epoch = d.epoch.current
            assert dropped_epoch > old  # 切断時点で生成中の音声を失効させる
            b2 = MockBrowser()
            b2.attach(d.core)  # 再接続で epoch+1
            assert d.epoch.current == dropped_epoch + 1 and b2.of("cancel")
            gate.set()  # 旧世代の LLM 結果が再接続後に到着
            await asyncio.sleep(0.1)
            assert b2.chunks == []
            b2.playback(old, 1, 0, "finished")  # 旧 epoch の playback は無視される
            assert d.history._state == {}
        finally:
            gate.set()
            store.close()
    asyncio.run(run())


# ---------------- E6 ----------------

def test_e6_device_loss_goes_to_rest_with_5005(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        try:
            limit = CFG.E_PARAMS.device_lost_s + 2
            while not b.of("error") and d.clock_.t <= limit:
                d.clock_.advance(0.02)
                b.send_frame(QUIET)
            assert [e["code"] for e in b.of("error")] == [ERR_MIC_LOST]
            assert d.clock_.t <= limit and state(d) == "rest"
            assert b.of("state")[-1]["state"] == "rest"
        finally:
            store.close()
    asyncio.run(run())


# ---------------- E7 ----------------

def test_e7_llm_kill_gives_fallback_then_restart_and_next_turn_replies(tmp_path):
    async def run():
        starts, sleeps = [], []
        clock = FakeClock()
        sup = LlmSupervisor(lambda: starts.append(1) or "proc", CFG, clock=clock, sleep=sleeps.append)
        killed = {"v": True}

        def llm(messages):
            if killed["v"]:
                killed["v"] = False
                sup.on_crash()  # llama-server 異常終了を検知 -> 初回バックオフ後に再起動
                raise ConnectionRefusedError("llama-server killed")
            yield from TOKENS

        d, b, store = make(tmp_path, llm=llm)
        try:
            t0 = time.monotonic()
            b.speak()
            await until(lambda: len(b.chunks) == 1)
            assert time.monotonic() - t0 <= CFG.HTTP_TIMEOUT.llm_first
            assert [e["code"] for e in b.of("error")] == [ERR_LLM_TIMEOUT]
            assert b.of("text")[0]["text"] == FALLBACK_PHRASE
            assert starts == [1] and sleeps == [CFG.LLM_RETRY.backoff_s[0]]  # 初回バックオフ 1s 内に再起動
            await until(lambda: state(d) == "idle")
            b.speak()
            await until(lambda: state(d) == "idle" and len(b.chunks) == 3)
            assert [t["text"] for t in b.of("text")][1:] == list(TOKENS)
        finally:
            store.close()
    asyncio.run(run())


# ---------------- E8 ----------------

def test_e8_tts_compute_stops_within_settle_after_cancel(tmp_path):
    async def run():
        cancelled = threading.Event()
        started = threading.Event()
        stamp = {}
        n = {"c": 0}

        def tts(text):
            n["c"] += 1
            if n["c"] > 1:
                return b"RIFF....WAVE"
            started.set()
            end = time.monotonic() + 10
            while not cancelled.is_set() and time.monotonic() < end:  # 模擬計算
                time.sleep(0.005)
            stamp["stopped"] = time.monotonic()
            return b"RIFF....WAVE"

        d, b, store = make(tmp_path, tts=tts, cancel_tts=cancelled.set)
        try:
            b.speak()
            await until(started.is_set)
            t0 = time.monotonic()
            b.speak()  # 合成中に再発話 -> cancel
            await until(lambda: "stopped" in stamp)
            assert stamp["stopped"] - t0 <= CFG.E_PARAMS.tts_cpu_settle_s
            await until(lambda: state(d) == "idle" and ai_rows(store, 1))
        finally:
            cancelled.set()
            store.close()
    asyncio.run(run())


# ---------------- 競合 ----------------

def test_race_cancel_and_completion_old_epoch_completion_is_ignored(tmp_path):
    async def run():
        d, b, store = make(tmp_path, llm=lambda m: iter(["今日はいい天気ですね。"]), auto_playback=False)
        try:
            b.speak()
            await until(lambda: state(d) == "ai_playing" and gen_done(d))
            old, utt = d.epoch.current, b.chunks[0][1]
            b.playback(old, utt, 0, "started", pos=100)
            d.fsm.enqueue(Event(Ev.VAD_START, strong=True))  # cancel 確定 (epoch+1)
            d._pump()
            assert d.epoch.current == old + 1
            n_turns = store.query("SELECT COUNT(*) FROM turns")[0][0]
            b.playback(old, utt, 0, "finished", pos=900)  # 同時に届いた旧世代の完了通知
            d.fsm.enqueue(Event(Ev.PLAYBACK_DONE, epoch=old))
            d._pump()
            d._check_done()
            assert store.query("SELECT COUNT(*) FROM turns")[0][0] - n_turns == 0
            assert ai_rows(store, 1) == [] and state(d) == "user_speaking"
        finally:
            store.close()
    asyncio.run(run())
