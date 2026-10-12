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


def test_default_off_and_freq():
    s = Spontaneous()
    assert s.enabled is False
    assert Config().spontaneous_enabled is False
    s.apply_settings({"on": True})
    assert s.enabled is True and s.freq_per_hour == 3


def test_freq_out_of_range_rejected_5006():
    s = Spontaneous()
    with pytest.raises(ValueError):
        s.apply_settings({"freq_per_hour": 13})
    assert s.freq_per_hour == 3


def test_freq_13_ui_returns_5006(tmp_path):
    d, b, _ = make(tmp_path)
    d.core.on_text(b, json.dumps({"t": "ui", "freq_per_hour": 13}))
    assert [m["code"] for m in b.of("error")] == [5006]


def fire_count(s, minute_now):
    n = 0
    for i in range(1, 200):
        t = i * 600.0
        if s.should_fire(t, True):
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


def test_silence_backoff_monotonic_no_consecutive():
    s = sp()
    now, fires = 0.0, []
    for _ in range(100000):
        now += 1.0
        if s.should_fire(now, True):
            if fires:
                assert now >= fires[-1] + s.intervals[-1]  # 連続発火なし
            s.fire(now)
            s.done()  # 利用者は無反応
            fires.append(now)
            if len(fires) == 4:
                break
    assert len(fires) == 4
    assert s.intervals[0] < s.intervals[1] < s.intervals[2] < s.intervals[3]
    s.on_user_speech()
    assert s.silent_count == 0


def test_not_chat_mode_never_fires():
    s = sp(mode="focus")
    assert not any(s.should_fire(i * 1000.0, True) for i in range(1, 50))
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
