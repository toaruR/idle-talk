"""M1 実機試験 (-m integration)。records/m1_events.db は実機20分会話で得たアプリ DB のコピー。
ファイル欠落・出所不備・不合格はすべて失敗として扱う。"""
import itertools
import json
from pathlib import Path

import pytest

from idle_talk.metrics import (
    RUN_META_KEYS,
    evaluate_gate,
    open_readonly,
    validate_run_meta,
)

pytestmark = pytest.mark.integration

DB_PATH = Path(__file__).resolve().parent.parent / "records" / "m1_events.db"
M1_MIN_DURATION_MS = 20 * 60 * 1000  # §10 M1 の 20 分


@pytest.fixture(scope="module")
def conn():
    c = open_readonly(DB_PATH)  # 欠落は FileNotFoundError で失敗
    yield c
    c.close()


def _meta(conn):
    rows = conn.execute("SELECT payload_json FROM events WHERE kind='run_meta'").fetchall()
    assert len(rows) == 1, f"run_meta rows: {len(rows)}"
    return json.loads(rows[0][0])


def test_live_run_meta_has_five_keys(conn):
    assert set(RUN_META_KEYS) <= set(_meta(conn))


def test_live_duration_at_least_20_minutes(conn):
    m = _meta(conn)
    assert m["ended_at"] - m["started_at"] >= M1_MIN_DURATION_MS


def test_live_ts_monotonic_non_decreasing(conn):
    ts = [r[0] for r in conn.execute("SELECT ts FROM events WHERE kind!='run_meta' ORDER BY id")]
    assert ts and all(b >= a for a, b in itertools.pairwise(ts))


def test_live_sample_count_matches_events(conn):
    n = conn.execute("SELECT COUNT(*) FROM events WHERE kind!='run_meta'").fetchone()[0]
    assert _meta(conn)["sample_count"] == n


def test_live_provenance_validates(conn):
    validate_run_meta(conn, M1_MIN_DURATION_MS)


def test_live_g3_no_stale_audio(conn):
    r = evaluate_gate(conn, "g3")
    assert r.passed, r.detail


def test_live_history_consistent(conn):
    r = evaluate_gate(conn, "history")
    assert r.passed, r.detail
