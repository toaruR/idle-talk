"""配線の試験: playback/cancel/echo_chain イベント記録、soak_sample 周期記録 (設計書 3.4 / 6.2 / 11.1 / 12.3)。"""
import asyncio
import json
import sqlite3

from test_integration_interrupt import make, state, until

from idle_talk.echo_chain import CANCEL_KIND, ECHO_CHAIN_KIND
from idle_talk.main import soak_loop
from idle_talk.metrics import MetricsRecorder, evaluate_gate, validate_run_meta
from idle_talk.resource_sampler import SOAK_KIND, ResourceSampler
from idle_talk.store import Store


def make_m(tmp_path, **kw):
    d, b, store = make(tmp_path, **kw)
    d.metrics = MetricsRecorder(store)  # 記録系イベントは M1 計測有効時のみ書く
    return d, b, store


def kinds(store, kind):
    return store.query("SELECT epoch, payload_json FROM events WHERE kind=? ORDER BY id", (kind,))


def test_consistent_session_passes_g3_and_history(tmp_path):
    async def run():
        d, b, store = make_m(tmp_path)
        try:
            b.speak()
            await until(lambda: state(d) == "idle" and store.count("turns") >= 2)
            rows = kinds(store, "playback")
            assert rows and all(set(json.loads(r["payload_json"])) ==
                                {"epoch", "utterance_id", "chunk_id", "state", "position_ms"} for r in rows)
            assert evaluate_gate(store._read, "g3").passed
            assert evaluate_gate(store._read, "history").passed
        finally:
            store.close()
    asyncio.run(run())


def test_stale_epoch_playback_fails_g3(tmp_path):
    async def run():
        d, b, store = make_m(tmp_path)
        try:
            b.speak()
            await until(lambda: state(d) == "idle" and store.count("turns") >= 2)
            stale = d.epoch.current
            d.epoch.bump()
            d._on_playback({"t": "playback", "epoch": stale, "utterance_id": 99, "chunk_id": 0,
                            "state": "started", "position_ms": 0})
            r = evaluate_gate(store._read, "g3")
            assert not r.passed and r.detail["stale"] == 1
        finally:
            store.close()
    asyncio.run(run())


def test_interrupt_records_cancel_and_not_echo_chain(tmp_path):
    async def run():
        d, b, store = make_m(tmp_path, auto_playback=False)
        try:
            b.speak()
            await until(lambda: state(d) == "ai_playing" and d._gen_done)
            d.texts_q.append("待って")
            b.speak()
            assert len(b.of("cancel")) == 1
            assert len(kinds(store, CANCEL_KIND)) == 1 and kinds(store, ECHO_CHAIN_KIND) == []
            await until(lambda d=d: d._task is not None and d._task.done())
        finally:
            store.close()
    asyncio.run(run())


def test_self_voice_transcript_records_echo_chain(tmp_path):
    async def run():
        d, b, store = make_m(tmp_path, auto_playback=False)
        try:
            b.speak()
            await until(lambda: state(d) == "ai_playing" and d._gen_done)
            d.texts_q.append("今日は、いい天気ですね")  # AI 本文の再認識 (句読点差は無視)
            b.speak()
            rows = kinds(store, ECHO_CHAIN_KIND)
            assert len(rows) == 1 and json.loads(rows[0]["payload_json"])["text"] == "今日は、いい天気ですね"
            await until(lambda d=d: d._task is not None and d._task.done())
        finally:
            store.close()
    asyncio.run(run())


def test_user_speech_outside_ai_playback_is_not_echo(tmp_path):
    async def run():
        d, b, store = make_m(tmp_path, texts=["今日はいい天気ですね"])
        try:
            b.speak()  # 利用者の最初の発話が偶然 AI 文と同文でも、AI 再生前なので連鎖ではない
            await until(lambda: state(d) == "idle" and store.count("turns") >= 2)
            assert kinds(store, ECHO_CHAIN_KIND) == []
        finally:
            store.close()
    asyncio.run(run())


def test_soak_loop_records_every_interval_with_injected_sleep(tmp_path):
    async def run():
        store = Store(tmp_path / "s.db", db_backup=False)
        try:
            t = iter(range(60_000, 10_000_000, 60_000))
            sampler = ResourceSampler(store, clock=lambda: next(t), rss_fn=lambda: 100.0, vram_fn=lambda: 2000.0)
            slept = []

            async def sleep(sec):
                slept.append(sec)
                if len(slept) > 3:
                    raise asyncio.CancelledError

            try:
                await soak_loop(sampler, 60_000, sleep)
            except asyncio.CancelledError:
                pass
            assert slept == [60.0] * 4
            rows = kinds(store, SOAK_KIND)
            assert len(rows) == 3
            assert json.loads(rows[0]["payload_json"]) == {"rss_mib": 100.0, "vram_mib": 2000.0}
        finally:
            store.close()
    asyncio.run(run())


def test_soak_loop_survives_sample_failure(tmp_path):
    async def run():
        store = Store(tmp_path / "f.db", db_backup=False)
        try:
            def bad():
                raise OSError("nvidia-smi missing")
            sampler = ResourceSampler(store, clock=lambda: 1, rss_fn=lambda: 1.0, vram_fn=bad)
            n = 0

            async def sleep(sec):
                nonlocal n
                n += 1
                if n > 2:
                    raise asyncio.CancelledError

            try:
                await soak_loop(sampler, 60_000, sleep)
            except asyncio.CancelledError:
                pass
            assert n == 3 and kinds(store, SOAK_KIND) == []
        finally:
            store.close()
    asyncio.run(run())


def test_run_meta_sample_count_includes_soak_rows(tmp_path):
    """close() の sample_count は run_meta 以外の全行 (soak_sample 含む) = validate_run_meta の M1 規則と一致。"""
    async def run():
        d, _b, store = make_m(tmp_path)
        ResourceSampler(store, clock=lambda: 1000, rss_fn=lambda: 1.0, vram_fn=lambda: 2.0).sample()
        d.close()
    asyncio.run(run())
    conn = sqlite3.connect(tmp_path / "t.db")
    assert validate_run_meta(conn, 0)["sample_count"] == 1
