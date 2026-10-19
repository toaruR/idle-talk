"""ブラウザ補正なしモード (旧 PC と話すモード): スピーカーで鳴る別の音声 AI と、マイクとスピーカーで話す。AI の声はモニターが鳴らす。"""
import asyncio
import json
import logging
import time

import pytest
from mocks import FakeClock
from mocks_browser import LOUD, QUIET, MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.config import Pc, load_pc
from idle_talk.director import IdleTalkDirector
from idle_talk.fsm import State
from idle_talk.history import talk_log
from idle_talk.monitor import MicCapture, Monitor
from idle_talk.protocol import validate_message
from idle_talk.store import Store
from idle_talk.vad import EnergyVad


# ---- 設定・プロトコル ----
def test_load_pc(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text('[pc]\nmonitor = "Realtek"\nmic = "USB"\n', encoding="utf-8")
    assert load_pc(f) == Pc(monitor="Realtek", mic="USB")
    assert load_pc(tmp_path / "none.toml") == Pc()
    for bad in ("monitor = 1", 'monitor = "' + "x" * 101 + '"', "mic = 1", "x = 1"):
        f.write_text(f"[pc]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_pc(f)


def test_ui_pc_schema():
    assert validate_message({"t": "ui", "pc": True})
    assert not validate_message({"t": "ui", "pc": 1})


# ---- Monitor ----
class FakeStream:
    def __init__(self, delay: float = 0.0):
        self.data: list[bytes] = []
        self.delay = delay
        self.closed = False

    def write(self, b: bytes) -> None:
        self.data.append(b)
        if self.delay:
            time.sleep(self.delay)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def wait_for(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < end, "timeout"
        time.sleep(0.005)


def test_monitor_plays_in_order_and_reports():
    reports, stream = [], FakeStream()
    m = Monitor(reports.append, lambda ep: ep == 1, lambda: stream, rate=1000)
    m.start()
    m.play(1, 7, 0, b"\x01\x00" * 30)
    m.play(1, 7, 1, b"\x02\x00" * 10)
    wait_for(lambda: len(reports) == 4)
    assert [(r["chunk_id"], r["state"]) for r in reports] == [(0, "started"), (0, "finished"), (1, "started"), (1, "finished")]
    assert reports[1]["position_ms"] == 30 and all(validate_message(r) for r in reports)
    m.stop()
    assert stream.closed and not m.running


def test_monitor_cancel_drops_old_epoch_and_duck_scales():
    cur = {"ep": 1}
    reports, stream = [], FakeStream(delay=0.01)
    m = Monitor(reports.append, lambda ep: ep == cur["ep"], lambda: stream, rate=1000)
    m.start()
    m.duck(0.5)
    m.play(1, 1, 0, b"\xe8\x03" * 200)  # 1000
    m.play(1, 1, 1, b"\xe8\x03" * 200)
    wait_for(lambda: reports)
    assert int.from_bytes(stream.data[0][:2], "little", signed=True) == 500
    cur["ep"] = 2
    m.cancel()
    time.sleep(0.05)
    assert [r["state"] for r in reports] == ["started"]  # 旧世代は止めて報告しない
    assert m.gain == 1.0
    m.stop()


def test_monitor_open_failure_raises():
    def bad():
        raise OSError("no device")
    m = Monitor(lambda r: None, lambda ep: True, bad)
    with pytest.raises(OSError):
        m.start()
    assert not m.running


class FakeInput:
    """read(n) で用意したフレームを返し、尽きたら止まるまで無音を返す。"""
    def __init__(self, frames: list[bytes] | None = None):
        self.frames = list(frames or [])
        self.closed = False

    def read(self, n):
        time.sleep(0.001)
        return (self.frames.pop(0) if self.frames else b"\x00\x00" * n), False

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def test_mic_capture_feeds_frames():
    got: list[bytes] = []
    m = MicCapture(got.append, lambda: FakeInput([b"\x01\x00" * 320]))
    m.start()
    wait_for(lambda: len(got) >= 2)
    m.stop()
    assert got[0] == b"\x01\x00" * 320 and not m.running


# ---- Director ----
def make(tmp_path, replies, texts, open_ok=True):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    seen: list[list[dict]] = []

    def llm(messages):
        seen.append(messages)
        yield replies.pop(0) if replies else "うん。"

    def send(req, timeout):
        return 200, {"utterance_id": req["utterance_id"], "text": texts.pop(0),
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE" + b"\x10\x00" * 10, lambda: None, EnergyVad(),
                         AsrClient(send, clock=clock), clock=clock, now_ms=lambda: 1000)
    stream = FakeStream()

    def open_stream():
        if not open_ok:
            raise OSError("no device")
        return stream
    d.open_monitor = open_stream
    mic = FakeInput()
    d.open_mic = lambda: mic
    d.mic_ = mic
    b = MockBrowser()
    b.attach(d.core)
    d.clock_, d.seen_, d.stream_ = clock, seen, stream
    return d, b, store


def pc_on(d, b, on=True):
    d.core.on_text(b, json.dumps({"t": "ui", "pc": on}))


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def ai_turns(store):
    return store.query("SELECT COUNT(*) FROM turns WHERE role='ai'")[0][0]


class Lines(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_pc_mode_plays_on_monitor_and_logs_pc(tmp_path):
    h = Lines()
    talk_log.addHandler(h)
    level, talk_log.level = talk_log.level, logging.INFO

    async def run():
        d, b, store = make(tmp_path, ["和食が好きだよ。"], ["好きな食べ物は何ですか"])
        pc_on(d, b)
        assert d.pc_on and d.monitor.running
        d.mic_.frames = [LOUD] * 15 + [QUIET] * 12  # マイクは Python が録る
        await until(lambda: ai_turns(store) == 1)
        assert b.chunks == [] and d.stream_.data  # 声はモニターが鳴らし、ブラウザには届かない
        assert "".join(m["text"] for m in b.of("text")) == "和食が好きだよ。"  # 文字は画面に出す
        assert [tuple(r) for r in store.query("SELECT role, text FROM turns WHERE role!='ai'")] == [("pc", "好きな食べ物は何ですか")]
        assert "PC: 好きな食べ物は何ですか" in h.lines
        assert d._mem_pending == [] and d._mem_task is None  # PC の発言は記憶にしない
        assert d.seen_[0][-1] == {"role": "user", "content": "好きな食べ物は何ですか"}
        pc_on(d, b, False)
        assert not d.pc_on and d.monitor is None and d.stream_.closed
    try:
        asyncio.run(run())
    finally:
        talk_log.removeHandler(h)
        talk_log.setLevel(level)


def test_pc_mode_off_plays_in_browser(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["うん。"], ["こんにちは"])
        b.speak()
        await until(lambda: ai_turns(store) == 1)
        assert b.chunks and d.stream_.data == []
        assert [tuple(r) for r in store.query("SELECT role FROM turns WHERE role!='ai'")] == [("user",)]
    asyncio.run(run())


def test_switching_stops_playing_voice(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["和食が好きだよ。"], ["好きな食べ物は何ですか"])
        b.auto_playback = False  # ブラウザで鳴っている途中
        b.speak()
        await until(lambda: d.fsm.ctx.state is State.AI_PLAYING)
        ep = d.epoch.current
        pc_on(d, b)
        assert d.pc_on and d.fsm.ctx.state is State.IDLE and d.epoch.current == ep + 1
        assert b.of("cancel")[-1]["new_epoch"] == ep + 1
        assert d.spontaneous_tick() is None  # ブラウザ補正なしの間は自発発話も無操作休止もしない
    asyncio.run(run())


def test_pc_mode_does_not_duck_monitor(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, [], [])
        pc_on(d, b)
        d._act_duck(None)  # マイクが拾ったモニターの声で発話開始になった
        assert d.monitor.gain == 1.0
        assert b.of("duck")  # 画面への通知は今までどおり
    asyncio.run(run())


def test_pc_mode_discards_speech_during_ai_voice_except_stop_words(tmp_path):
    def pc_echo(store):
        return store.query("SELECT COUNT(*) FROM events WHERE kind='pc_echo_suppressed'")[0][0]

    async def run():
        d, b, store = make(tmp_path, [], [])
        d._ai_text = "そう言ってくれるの嬉しいわ！"
        d._speech_in_ai = True
        assert d._on_transcript(1, "うれしい") is True  # モード外: 文字が一致しないので自声とみなさない
        pc_on(d, b)
        d._speech_in_ai = True
        assert d._on_transcript(2, "うれしい") is False  # AI の声の聞き違い
        assert pc_echo(store) == 1
        d._speech_in_ai = True
        d._on_transcript(3, "ストップ")  # 停止語は止める
        assert pc_echo(store) == 1
        d._on_transcript(4, "好きな食べ物は何ですか")  # AI の声の最中でなければ使う
        assert pc_echo(store) == 1
    asyncio.run(run())


def test_pc_mode_listens_with_python_mic(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["和食が好きだよ。"], ["好きな食べ物は何ですか"])
        pc_on(d, b)
        assert d.mic is not None and d.mic.running
        b.speak()  # ブラウザのマイクは捨てる
        await asyncio.sleep(0.1)
        assert d.fsm.ctx.state is State.IDLE
        d.mic_.frames = [LOUD] * 15 + [QUIET] * 12  # Python のマイクで聞く
        await until(lambda: ai_turns(store) == 1)
        pc_on(d, b, False)
        assert d.mic is None and d.mic_.closed
    asyncio.run(run())


def test_pc_mode_quiet_python_mic_is_not_device_lost(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, [], [])
        d.ingress.clock = d.clock_
        pc_on(d, b)
        for _ in range(250):  # 静かな部屋の生の音 (RMS 1 未満) が 5 秒
            d._feed_mic(b"\x01\x00" * 320)
            d.clock_.advance(0.02)
        d._pump()
        assert not d.ingress.device_lost and d.fsm.ctx.state is State.IDLE
        assert not b.of("error")
        pc_on(d, b, False)
        assert d.ingress.silence_check
    asyncio.run(run())


def test_mic_capture_read_failure_reports_lost():
    class Broken(FakeInput):
        def read(self, n):
            raise OSError("unplugged")
    lost: list[bool] = []
    m = MicCapture(lambda pcm: None, Broken, lambda: lost.append(True))
    m.start()
    wait_for(lambda: lost)
    m.stop()


def test_pc_mode_mic_lost_rests(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, [], [])
        pc_on(d, b)
        d._mic_lost()
        await until(lambda: d.fsm.ctx.state is State.REST)
        assert b.of("error")[-1]["code"] == 5005
    asyncio.run(run())


def test_open_failure_refuses(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, [], [], open_ok=False)
        pc_on(d, b)
        assert not d.pc_on and d.monitor is None
        assert b.of("error")[-1]["code"] == 5006 and "no device" in b.of("error")[-1]["message"]
    asyncio.run(run())
