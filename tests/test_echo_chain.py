"""M5 スピーカー AEC 判定 (E10/echo_chain) の試験。DB はすべて pytest 内で生成する。"""
import json
import sqlite3

import pytest

from idle_talk.echo_chain import ECHO_MIN_DURATION_MS, evaluate_echo_chain
from idle_talk.metrics import ProvenanceError
from idle_talk.store import Store

T0 = 1_700_000_000_000


def build_db(tmp_path, chains=0, cancels=1, duration_ms=ECHO_MIN_DURATION_MS, meta=True,
             reverse=False, name="m5.db"):
    s = Store(tmp_path / name, db_backup=False)
    kinds = ["echo_chain"] * chains + ["cancel"] * cancels + ["playback"]
    ts = [T0 + i * 1000 for i in range(len(kinds))]
    if reverse:
        ts[-1] = T0 - 1
    for t, k in zip(ts, kinds):
        s.add_event(t, 0, k, json.dumps({}))
    if meta:
        s.add_event(T0, 0, "run_meta", json.dumps({
            "host": "h", "started_at": T0, "ended_at": T0 + duration_ms,
            "app_commit": "abc", "sample_count": len(kinds)}))
    s.close()
    return sqlite3.connect(str(tmp_path / name))


def test_chain_zero_passes_and_interruptible(tmp_path):
    r = evaluate_echo_chain(build_db(tmp_path, chains=0, cancels=2))
    assert r.passed and r.chain_count == 0 and r.interruptible and r.recommendation == ""


def test_chain_one_fails_with_pipecat_recommendation(tmp_path):
    r = evaluate_echo_chain(build_db(tmp_path, chains=1))
    assert not r.passed and not r.interruptible
    assert "Pipecat" in r.recommendation and "Pipecat" in r.detail


def test_cancel_zero_not_interruptible(tmp_path):
    r = evaluate_echo_chain(build_db(tmp_path, chains=0, cancels=0))
    assert r.passed and not r.interruptible


def test_cancel_one_interruptible(tmp_path):
    assert evaluate_echo_chain(build_db(tmp_path, cancels=1)).interruptible


def test_calls_validate_run_meta_with_20min(tmp_path, monkeypatch):
    import idle_talk.echo_chain as ec
    seen = {}

    def fake(conn, min_duration_ms, interval_ms=None):
        seen["min"] = min_duration_ms
        raise ProvenanceError("stop")
    monkeypatch.setattr(ec, "validate_run_meta", fake)
    with pytest.raises(ProvenanceError):
        ec.evaluate_echo_chain(build_db(tmp_path))
    assert seen["min"] == 1_200_000


def test_negative_no_run_meta(tmp_path):
    with pytest.raises(ProvenanceError):
        evaluate_echo_chain(build_db(tmp_path, meta=False))


def test_negative_duration_one_ms_short(tmp_path):
    with pytest.raises(ProvenanceError):
        evaluate_echo_chain(build_db(tmp_path, duration_ms=ECHO_MIN_DURATION_MS - 1))


def test_negative_ts_reversed(tmp_path):
    with pytest.raises(ProvenanceError):
        evaluate_echo_chain(build_db(tmp_path, reverse=True))
