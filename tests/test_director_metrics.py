"""Director の M1 計測結線 (設計書 3.4 / 10 / 11.1): 段階 5 行/ターン・相槌除外・run_meta。"""
import asyncio
import itertools
import json

from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.director import IdleTalkDirector
from idle_talk.metrics import STAGES, MetricsRecorder, _g1_latencies, validate_run_meta
from idle_talk.store import Store
from idle_talk.vad import EnergyVad


def make(tmp_path, gpu1_load=False):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    tick = itertools.count(1000, 10)
    d = IdleTalkDirector(store, lambda m: iter(["今日はいい天気ですね。", "散歩に行きましょう。"]),
                         lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: next(tick),
                         metrics=MetricsRecorder(store), app_commit="abc123", gpu1_load=lambda: gpu1_load)
    b = MockBrowser()
    b.attach(d.core)
    return d, b, store


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def stage_rows(store):
    q = ",".join("?" for _ in STAGES)
    return store.query(f"SELECT ts, kind, payload_json FROM events WHERE kind IN ({q}) ORDER BY id", STAGES)


def test_full_turn_writes_five_stage_events_in_order(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        try:
            b.speak()
            await until(lambda: d.fsm.ctx.state.value == "idle" and store.count("turns") >= 2)
            rows = stage_rows(store)
            assert [r["kind"] for r in rows] == list(STAGES)
            assert [json.loads(r["payload_json"]) for r in rows] == [{"turn": 0, "gpu1_load": False}] * 5
            ts = [r["ts"] for r in rows]
            assert ts == sorted(ts)
            lat, loaded = _g1_latencies(store._read)
            assert len(lat) == 1 and lat[0] == ts[-1] - ts[0] and loaded == []
        finally:
            store.close()
    asyncio.run(run())


def test_gpu1_load_flag_recorded(tmp_path):
    async def run():
        d, b, store = make(tmp_path, gpu1_load=True)
        try:
            b.speak()
            await until(lambda: d.fsm.ctx.state.value == "idle" and store.count("turns") >= 2)
            assert all(json.loads(r["payload_json"])["gpu1_load"] is True for r in stage_rows(store))
        finally:
            store.close()
    asyncio.run(run())


def test_backchannel_not_counted_as_g1(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        try:
            b.speak()
            await until(lambda: d.fsm.ctx.state.value == "idle" and store.count("turns") >= 2)
            n_before = len(stage_rows(store))
            d.enable_backchannel([("うん", b"RIFF....WAVE")])
            n_chunks = len(b.chunks)
            d._bc_play("うん", b"RIFF....WAVE")  # 事前合成クリップも play_utterance を通る
            await until(lambda: len(b.chunks) > n_chunks)
            await asyncio.sleep(0.05)
            assert len(stage_rows(store)) == n_before == 5
            # 相槌判定された発話は新ターンとして数えず、段階行も 5 行を超えない
            d._m_mark("eot_wait")
            d._m_mark("asr_done")
            d._m_abandon()
            lat, _ = _g1_latencies(store._read)
            assert len(lat) == 1
        finally:
            store.close()
    asyncio.run(run())


def test_run_meta_on_close(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        b.speak()
        await until(lambda: d.fsm.ctx.state.value == "idle" and store.count("turns") >= 2)
        d.metrics.mark_g2(5000, 100)
        d.close()
        d.close()  # 冪等
    asyncio.run(run())
    import sqlite3
    conn = sqlite3.connect(tmp_path / "t.db")
    metas = conn.execute("SELECT payload_json FROM events WHERE kind='run_meta'").fetchall()
    assert len(metas) == 1
    meta = json.loads(metas[0][0])
    assert set(meta) == {"host", "started_at", "ended_at", "app_commit", "sample_count"}
    assert meta["app_commit"] == "abc123" and meta["ended_at"] >= meta["started_at"]
    n = conn.execute("SELECT COUNT(*) FROM events WHERE kind!='run_meta'").fetchone()[0]
    n_pb = conn.execute("SELECT COUNT(*) FROM events WHERE kind='playback'").fetchone()[0]
    assert n_pb >= 1  # 再生報告も events に記録される (G3/履歴照合用)
    assert meta["sample_count"] == n == 6 + n_pb
    validated = validate_run_meta(conn, 0)
    assert validated == meta
    conn.close()


def test_no_metrics_by_default(tmp_path):
    async def run():
        store = Store(tmp_path / "n.db", db_backup=False)
        d = IdleTalkDirector(store, lambda m: iter(["はい。"]), lambda t: b"RIFF....WAVE", lambda: None,
                             EnergyVad(), AsrClient(mock_asr_send(FakeClock())), clock=FakeClock())
        b = MockBrowser()
        b.attach(d.core)
        b.speak()
        await until(lambda: d.fsm.ctx.state.value == "idle" and store.count("turns") >= 2)
        d.close()
    asyncio.run(run())
    import sqlite3
    assert sqlite3.connect(tmp_path / "n.db").execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
