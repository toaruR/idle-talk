"""自発発話・無操作休止 (T020 / 設計書 4.3-5 / 6.1 / 8.1)。"""
import asyncio
import json
import threading

import pytest
from mocks import FakeClock
from mocks_browser import MockBrowser

from idle_talk.config import Config
from idle_talk.director import IdleTalkDirector
from idle_talk.spontaneous import Spontaneous, in_quiet_hours
from idle_talk.store import Store
from idle_talk.vad import EnergyVad


def sp(minute=12 * 60, **kw):
    s = Spontaneous(minutes=lambda: minute, **kw)
    s.apply_settings({"on": True})
    return s


def test_default_off_and_interval():
    s = Spontaneous()
    assert s.enabled is False
    assert Config().spontaneous_enabled is False
    s.apply_settings({"on": True})
    assert s.enabled is True and s.interval_s == 1200


def test_interval_out_of_range_rejected_5006():
    s = Spontaneous()
    with pytest.raises(ValueError):
        s.apply_settings({"interval_s": 9})
    assert s.interval_s == 1200


def test_interval_9_ui_returns_5006(tmp_path):
    d, b, _ = make(tmp_path)
    d.core.on_text(b, json.dumps({"t": "ui", "interval_s": 9}))
    assert [m["code"] for m in b.of("error")] == [5006]


def fire_count(s, minute_now):
    n = 0
    for i in range(1, 200):
        t = i * 600.0
        if s.should_fire(t, t - 1200.0):
            s.fire(t)
            s.done()
            s.on_user_speech()
            n += 1
    return n


@pytest.mark.parametrize("minute,expected_any", [(23 * 60 + 30, False), (3 * 60, False), (12 * 60, True)])
def test_quiet_hours(minute, expected_any):
    s = sp(minute)
    assert (fire_count(s, minute) > 0) is expected_any


def test_quiet_hours_boundaries():
    assert in_quiet_hours(22 * 60, "22:00", "07:00")
    assert not in_quiet_hours(7 * 60, "22:00", "07:00")
    assert not in_quiet_hours(21 * 60 + 59, "22:00", "07:00")


def test_fires_after_interval_of_silence():
    s = sp()
    s.apply_settings({"interval_s": 30})
    assert not s.should_fire(129.0, 100.0)
    assert s.should_fire(130.0, 100.0)
    assert not s.should_fire(130.0, None)  # IDLE でない


def test_silence_backoff_monotonic_no_consecutive():
    s = sp()
    s.apply_settings({"interval_s": 30})
    idle_since, fires = 0.0, []
    now = 0.0
    while len(fires) < 4:
        now += 1.0
        assert now < 100000
        if s.should_fire(now, idle_since):
            if fires:
                assert now - idle_since >= s.intervals[-1]  # 連続発火なし
            s.fire(now)
            s.done()  # 利用者は無反応。再生終了後に再び IDLE
            fires.append(now)
            idle_since = now
    assert s.intervals == [30.0, 60.0, 120.0, 240.0]
    s.on_user_speech()
    assert s.silent_count == 0 and s.wait_s == 30.0


def test_not_chat_mode_never_fires():
    s = sp(mode="focus")
    assert not any(s.should_fire(i * 1000.0, 0.0) for i in range(1, 50))
    assert s.fired == 0


def make(tmp_path, gate=None):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)

    def llm(messages):
        if gate is not None:
            gate.wait(10)
        yield from ("やあ、元気ですか。",)

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         None, clock=clock, now_ms=lambda: 1000)
    d.spontaneous._minutes = lambda: 12 * 60
    b = MockBrowser()
    b.attach(d.core)
    d.clock_ = clock
    return d, b, store


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def test_idle_timeout(tmp_path):
    async def run():
        d, _b, _ = make(tmp_path)
        d.clock_.advance(1799)
        assert d.spontaneous_tick() is None and d.fsm.ctx.state.value == "idle"
        d.clock_.advance(1)  # 1800s
        assert d.spontaneous_tick() == "rest" and d.fsm.ctx.state.value == "rest"
    asyncio.run(run())


def test_idle_timeout_29m59s(tmp_path):
    async def run():
        d, _b, _ = make(tmp_path)
        d.clock_.advance(1799.0)
        d.spontaneous_tick()
        assert d.fsm.ctx.state.value == "idle"
    asyncio.run(run())


def test_spontaneous_plays_via_common_path(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True}))
        assert d.spontaneous_tick() is None  # 初回は 1 周期待つ
        d.clock_.advance(1200)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending)
        assert len(b.chunks) >= 1
        assert ("spontaneous", "play_utterance") in d.path_calls
        assert d.spontaneous.cancelled == 0
    asyncio.run(run())


def test_cancel_prep(tmp_path):
    async def run():
        gate = threading.Event()
        d, b, _store = make(tmp_path, gate)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True}))
        d.spontaneous_tick()
        d.clock_.advance(1200)
        assert d.spontaneous_tick() == "spontaneous"
        await asyncio.sleep(0.05)
        assert d.spontaneous.pending
        b.speak()  # 準備中に利用者発話
        gate.set()
        await asyncio.sleep(0.2)
        assert d.spontaneous.cancelled == 1
        assert len(b.chunks) == 0
        assert ("spontaneous", "cancel") in d.path_calls
        assert not d.spontaneous.pending
    asyncio.run(run())


def test_off_by_default_no_fire(tmp_path):
    async def run():
        d, _b, _ = make(tmp_path)
        d.clock_.advance(1000)
        assert d.spontaneous_tick() is None
        d.clock_.advance(1000)
        assert d.spontaneous.fired == 0
    asyncio.run(run())


def test_silence_measured_from_end_of_speech_with_backoff(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        d.clock_.advance(29)
        assert d.spontaneous_tick() is None
        d.clock_.advance(1)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending and d.fsm.ctx.state.value == "idle")
        d.clock_.advance(59)  # 発話終了から計測。無反応なので 30s x 2
        assert d.spontaneous_tick() is None
        d.clock_.advance(1)
        assert d.spontaneous_tick() == "spontaneous"
        assert d.spontaneous.intervals == [30.0, 60.0]
    asyncio.run(run())


def test_spontaneous_loop_ticks():
    from idle_talk.main import spontaneous_loop
    ticks = []

    class D:
        def spontaneous_tick(self):
            ticks.append(1)

    async def sleep(_s):
        if len(ticks) == 3:
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(spontaneous_loop(D(), sleep=sleep))
    assert len(ticks) == 3


def test_silence_counted_from_connect(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.clock_.advance(100)  # 接続前の経過は数えない
        b.attach(d.core)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        assert d.spontaneous_tick() is None
        d.clock_.advance(30)
        assert d.spontaneous_tick() == "spontaneous"
    asyncio.run(run())


def test_no_fire_without_client_and_reconnect_clears_stuck(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        d.core.hello_done = False  # 未接続相当
        d.clock_.advance(100)
        assert d.spontaneous_tick() is None
        d.core.hello_done = True
        d.spontaneous.fire(d.clock_())  # 前の接続で再生されずに残った自発発話
        d.spontaneous.awaiting_reaction, d.spontaneous.silent_count = True, 2
        b.attach(d.core)  # 再接続 (hello)
        assert not d.spontaneous.pending and d.spontaneous.wait_s == 30.0
        d.clock_.advance(30)
        assert d.spontaneous_tick() == "spontaneous"
    asyncio.run(run())
