"""M5 スピーカー 実機試験 (-m integration)。records/m5_events.db は 20 分運転で得たアプリ DB のコピー。
ファイル欠落・出所不備・不合格はすべて失敗として扱う。"""
import itertools
import json
from pathlib import Path

import pytest

from idle_talk.echo_chain import ECHO_MIN_DURATION_MS, evaluate_echo_chain
from idle_talk.metrics import RUN_META_KEYS, open_readonly

pytestmark = pytest.mark.integration

DB_PATH = Path(__file__).resolve().parent.parent / "records" / "m5_events.db"


@pytest.fixture(scope="module")
def conn():
    c = open_readonly(DB_PATH)  # 欠落は FileNotFoundError で失敗
    yield c
    c.close()


def _meta(conn):
    rows = conn.execute("SELECT payload_json FROM events WHERE kind='run_meta'").fetchall()
    assert len(rows) == 1, f"run_meta rows: {len(rows)}"
    return json.loads(rows[0][0])


def _ts(conn):
    return [r[0] for r in conn.execute("SELECT ts FROM events WHERE kind!='run_meta' ORDER BY id")]


def test_live_run_meta_has_five_keys(conn):
    assert set(RUN_META_KEYS) <= set(_meta(conn))


def test_live_duration_at_least_20_minutes(conn):
    m = _meta(conn)
    assert m["ended_at"] - m["started_at"] >= ECHO_MIN_DURATION_MS


def test_live_ts_monotonic_non_decreasing(conn):
    ts = _ts(conn)
    assert ts and all(b >= a for a, b in itertools.pairwise(ts))


def test_live_sample_count_matches_rows(conn):
    assert _meta(conn)["sample_count"] == len(_ts(conn))


def test_live_echo_chain_zero(conn):
    r = evaluate_echo_chain(conn)
    assert r.passed, r.detail


def test_live_cancel_at_least_one(conn):
    r = evaluate_echo_chain(conn)
    assert r.cancel_count >= 1 and r.interruptible
