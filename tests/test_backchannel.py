"""AI 相槌 (T018): 発火条件・T-BC・誤発火記録・同一経路・事前合成・G1 除外。"""
import asyncio
import inspect
import sqlite3

from mocks import FakeClock, mock_tts_post
from mocks_browser import MockBrowser

from idle_talk import backchannel
from idle_talk.backchannel import Backchannel, BackchannelGate, synthesize_clips
from idle_talk.clients import TtsClient
from idle_talk.config import Config
from idle_talk.director import IdleTalkDirector
from idle_talk.metrics import MetricsRecorder, _g1_latencies
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

CFG = Config()
N = CFG.TEST_N


def test_threshold_300_fires_299_does_not():
    assert BackchannelGate(CFG).should_fire(300, 0.0)
    assert not BackchannelGate(CFG).should_fire(299, 0.0)
    assert not BackchannelGate(CFG).should_fire(CFG.T_silence, 0.0)


def test_max_once_per_utterance():
    g = BackchannelGate(CFG)
    assert g.should_fire(300, 0.0)
    assert not g.should_fire(310, 0.1)
    g.begin_utterance()
    assert g.should_fire(300, 100.0)


def test_cooldown_boundary():
    g = BackchannelGate(CFG)
    assert g.should_fire(300, 10.0)
    g.begin_utterance()
    assert not g.should_fire(300, 10.0 + 4.999)
    g.begin_utterance()
    assert g.should_fire(300, 10.0 + 5.0)


def make_bc(clock, played, cancelled, events):
    return Backchannel([("うん", b"W1"), ("はい", b"W2")], lambda t, w: played.append(t),
                       lambda: cancelled.append(1), lambda k, p: events.append(k), CFG, clock)


def test_t_bc_silence_reaching_t_silence_leaves_zero_backchannel():
    for i in range(N):
        clock, played, cancelled, events = FakeClock(), [], [], []
        clock.advance(100.0 * (i + 1))
        bc = make_bc(clock, played, cancelled, events)
        bc.on_speech_start()
        for ms in range(0, CFG.T_silence + 20, 20):
            bc.on_silence(ms)
        survivors = len(played) - len(cancelled)
        assert survivors == 0
        assert events.count("backchannel_misfire") == len(cancelled) == len(played)


def test_t_bc_silence_short_gives_at_most_one():
    for i in range(N):
        clock, played, cancelled, events = FakeClock(), [], [], []
        clock.advance(100.0 * (i + 1))
        bc = make_bc(clock, played, cancelled, events)
        bc.on_speech_start()
        for ms in range(0, CFG.T_silence, 20):  # 380ms まで: T_silence 未達
            bc.on_silence(ms)
        assert len(played) - len(cancelled) <= 1
        assert cancelled == [] and "backchannel_misfire" not in events


def test_synthesize_clips_order_and_content_independent():
    clock = FakeClock()
    post = mock_tts_post(clock)
    clips = synthesize_clips(TtsClient(config=CFG, clock=clock, post=post))
    assert len(clips) >= 1
    kinds = [("audio_query" if "audio_query" in u else "synth") for u in post.calls]
    assert kinds[:2] == ["audio_query", "synth"]
    assert all(kinds[i] == ("audio_query" if i % 2 == 0 else "synth") for i in range(len(kinds)))
    assert list(inspect.signature(backchannel.synthesize_clips).parameters) == ["tts_client"]


def test_metrics_excludes_backchannel(tmp_path):
    store = Store(tmp_path / "m.db", db_backup=False)
    try:
        MetricsRecorder(store).mark_backchannel(1, 1000, 50)
        store.add_event(1000, 0, "backchannel_misfire", "{}")
        con = sqlite3.connect(tmp_path / "m.db")
        assert _g1_latencies(con) == ([], [])
        con.close()
    finally:
        store.close()


async def until(pred, timeout=3.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.005)


def make_director(tmp_path):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    d = IdleTalkDirector(store, lambda m: iter(("はい。",)), lambda t: b"RIFF....WAVE", lambda: None,
                         EnergyVad(), None, cfg=CFG, clock=clock, now_ms=lambda: 1000)
    b = MockBrowser(auto_playback=False)
    b.attach(d.core)
    d.enable_backchannel([("うん", b"RIFF....WAVE")])
    return d, b, store, clock


def test_clip_goes_through_play_and_cancel_path_and_epoch_discards(tmp_path):
    async def run():
        d, b, store, clock = make_director(tmp_path)
        try:
            clock.advance(100.0)
            assert d.backchannel.on_silence(300)
            await until(lambda: len(b.chunks) == 1)
            assert ("backchannel", "play_utterance") in d.path_calls
            assert b.chunks[0][0] == 0
            assert d.fsm.ctx.state.value == "idle"  # 相槌は FSM に触れない
            d.backchannel.resolve(CFG.T_silence)  # T_silence 到達: cancel 経路で epoch 破棄
            await asyncio.sleep(0.05)
            assert ("backchannel", "cancel") in d.path_calls
            assert d.epoch.current == 1 and b.of("cancel")[-1]["new_epoch"] == 1
            rows = store.query("SELECT kind FROM events WHERE kind='backchannel_misfire'")
            assert len(rows) == 1
            # cancel 後に旧 epoch の相槌を再生しようとしても破棄される
            n = len(b.chunks)
            d._bc_active = "うん"
            await d.pipeline.play_utterance(iter(["うん"]))  # 現 epoch では再生される
            assert all(c[0] == 1 for c in b.chunks[n:])
        finally:
            store.close()
    asyncio.run(run())


def test_stale_epoch_clip_not_played(tmp_path):
    async def run():
        d, b, store, clock = make_director(tmp_path)
        try:
            clock.advance(100.0)
            d.backchannel.on_silence(300)
            d._cancel_path("backchannel")  # 再生タスクが走る前に cancel: 旧 epoch の相槌
            await asyncio.sleep(0.1)
            assert [c for c in b.chunks if c[0] == 0] == []
        finally:
            store.close()
    asyncio.run(run())
