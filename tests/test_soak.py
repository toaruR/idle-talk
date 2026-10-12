"""常駐試験ハーネス(G5/E12)の試験。データ・DB はすべて pytest 内で生成する。"""
import itertools
import json
import sqlite3
import tracemalloc

import pytest

from idle_talk.config import Config
from idle_talk.history import history_window
from idle_talk.metrics import ProvenanceError, validate_run_meta
from idle_talk.resource_sampler import (
    SAMPLE_INTERVAL_MS,
    SOAK_DURATION_MS,
    SOAK_MIN_SAMPLES,
    ResourceSampler,
    analyze_soak,
    load_soak_rows,
    validate_soak,
)
from idle_talk.store import Store

T0 = 1_700_000_000_000


def linear_rows(total_delta_mib, n=SOAK_MIN_SAMPLES, base=1000.0):
    """n 点を 1 分刻みで並べ、最初〜最後の差が 8h 換算で total_delta_mib になる RSS 列。"""
    rate_per_ms = total_delta_mib / SOAK_DURATION_MS
    return [{"ts": T0 + i * SAMPLE_INTERVAL_MS,
             "rss_mib": base + rate_per_ms * i * SAMPLE_INTERVAL_MS,
             "vram_mib": 2000.0} for i in range(n)]


def build_db(tmp_path, n=SOAK_MIN_SAMPLES, ts_list=None, duration_ms=SOAK_DURATION_MS,
             meta=True, sample_count=None, name="soak.db"):
    s = Store(tmp_path / name, db_backup=False)
    ts_list = ts_list if ts_list is not None else [T0 + i * SAMPLE_INTERVAL_MS for i in range(n)]
    for t in ts_list:
        s.add_event(t, 0, "soak_sample", json.dumps({"rss_mib": 1000.0, "vram_mib": 2000.0}))
    if meta:
        s.add_event(T0, 0, "run_meta", json.dumps({
            "host": "h", "started_at": T0, "ended_at": T0 + duration_ms, "app_commit": "abc",
            "sample_count": len(ts_list) if sample_count is None else sample_count}))
    s.close()
    return sqlite3.connect(str(tmp_path / name))


# --- sampler ---
def test_sampler_records_480_rows_in_8h(tmp_path):
    s = Store(tmp_path / "s.db", db_backup=False)
    now = [T0]
    sampler = ResourceSampler(s, clock=lambda: now[0], rss_fn=lambda: 500.0, vram_fn=lambda: 3000.0)
    for _ in range(SOAK_DURATION_MS // SAMPLE_INTERVAL_MS):
        sampler.sample()
        now[0] += SAMPLE_INTERVAL_MS
    rows = s.query("SELECT ts, kind, payload_json FROM events")
    s.close()
    assert len(rows) == 480
    assert all(r["kind"] == "soak_sample" for r in rows)
    p = json.loads(rows[0]["payload_json"])
    assert p == {"rss_mib": 500.0, "vram_mib": 3000.0}
    ts = [r["ts"] for r in rows]
    assert all(b - a == 60_000 for a, b in itertools.pairwise(ts))


def test_sampler_output_passes_provenance(tmp_path):
    s = Store(tmp_path / "s.db", db_backup=False)
    now = [T0]
    sampler = ResourceSampler(s, clock=lambda: now[0], rss_fn=lambda: 500.0, vram_fn=lambda: 1.0)
    for _ in range(480):
        sampler.sample()
        now[0] += SAMPLE_INTERVAL_MS
    s.add_event(T0, 0, "run_meta", json.dumps({
        "host": "h", "started_at": T0, "ended_at": T0 + SOAK_DURATION_MS,
        "app_commit": "c", "sample_count": 480}))
    s.close()
    conn = sqlite3.connect(str(tmp_path / "s.db"))
    assert validate_soak(conn)["sample_count"] == 480
    assert len(load_soak_rows(conn)) == 480
    conn.close()


# --- slope (G5) ---
def test_slope_1mib_per_h_passes():
    r = analyze_soak(linear_rows(8.0))
    assert r.passed, r.detail
    assert r.slope_mib_per_h == pytest.approx(1.0, abs=1e-6)


def test_slope_9mib_in_8h_fails():
    r = analyze_soak(linear_rows(9.0))
    assert not r.passed
    assert r.slope_mib_per_h == pytest.approx(1.125, abs=1e-6)


def test_slope_flat_passes_and_degenerate_fails():
    assert analyze_soak(linear_rows(0.0)).passed
    assert not analyze_soak(linear_rows(0.0, n=1)).passed


# --- E12 / T-E12 ---
def test_e12_1000_turns_window_bounded_and_rss_slope(tmp_path):
    cfg = Config()
    s = Store(tmp_path / "e12.db", db_backup=False)
    tracemalloc.start()
    rows = []
    try:
        for i in range(1000):
            s.add_turn(T0 + i * SAMPLE_INTERVAL_MS, "user" if i % 2 == 0 else "ai",
                       f"turn {i} " + "x" * 100, 1)
            win = history_window(s, cfg)
            assert len(win) <= cfg.N_TURNS == 8
            cur, _ = tracemalloc.get_traced_memory()
            rows.append({"ts": T0 + i * SAMPLE_INTERVAL_MS, "rss_mib": cur / (1024 * 1024)})
    finally:
        tracemalloc.stop()
        s.close()
    assert len(win) == 8
    r = analyze_soak(rows)
    assert r.passed, r.detail


# --- negative ---
def test_negative_valid_fixture_passes(tmp_path):
    conn = build_db(tmp_path)
    validate_run_meta(conn, SOAK_DURATION_MS, interval_ms=SAMPLE_INTERVAL_MS)
    validate_soak(conn)
    conn.close()


def test_negative_no_run_meta(tmp_path):
    conn = build_db(tmp_path, meta=False)
    with pytest.raises(ProvenanceError):
        validate_run_meta(conn, SOAK_DURATION_MS, interval_ms=SAMPLE_INTERVAL_MS)
    conn.close()


def test_negative_duration_one_ms_short(tmp_path):
    conn = build_db(tmp_path, duration_ms=SOAK_DURATION_MS - 1)
    with pytest.raises(ProvenanceError):
        validate_run_meta(conn, SOAK_DURATION_MS, interval_ms=SAMPLE_INTERVAL_MS)
    conn.close()


def test_negative_fewer_than_480_rows(tmp_path):
    conn = build_db(tmp_path, n=SOAK_MIN_SAMPLES - 1)
    with pytest.raises(ProvenanceError):
        validate_soak(conn)
    conn.close()


def test_negative_ts_not_monotonic(tmp_path):
    ts = [T0 + i * SAMPLE_INTERVAL_MS for i in range(480)]
    ts[100], ts[101] = ts[101], ts[100]
    conn = build_db(tmp_path, ts_list=ts)
    with pytest.raises(ProvenanceError):
        validate_run_meta(conn, SOAK_DURATION_MS, interval_ms=SAMPLE_INTERVAL_MS)
    with pytest.raises(ProvenanceError):
        validate_soak(conn)
    conn.close()


def test_negative_ts_not_strictly_increasing(tmp_path):
    ts = [T0 + i * SAMPLE_INTERVAL_MS for i in range(480)]
    ts[200] = ts[199]
    conn = build_db(tmp_path, ts_list=ts)
    with pytest.raises(ProvenanceError):
        validate_soak(conn)
    conn.close()


def test_negative_sample_count_mismatch(tmp_path):
    conn = build_db(tmp_path, sample_count=479)
    with pytest.raises(ProvenanceError):
        validate_run_meta(conn, SOAK_DURATION_MS, interval_ms=SAMPLE_INTERVAL_MS)
    conn.close()
