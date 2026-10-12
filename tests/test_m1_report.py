"""M1 計測記録・合否判定の試験。events DB は pytest 一時ディレクトリ内で生成する。"""
import json
import sqlite3

import pytest

from idle_talk.metrics import (
    STAGES,
    MetricsRecorder,
    ProvenanceError,
    evaluate_gate,
    percentile,
    validate_run_meta,
)
from idle_talk.store import Store

MIN_DUR = 1_200_000


def make_conn(tmp_path, name="m1.db"):
    s = Store(tmp_path / name, db_backup=False)
    s.close()
    return sqlite3.connect(str(tmp_path / name))


def add(conn, ts, kind, payload, epoch=0):
    conn.execute("INSERT INTO events(ts, epoch, kind, payload_json) VALUES (?,?,?,?)",
                 (ts, epoch, kind, json.dumps(payload)))
    conn.commit()


def lat_list(p50, p95, n=50):
    """nearest-rank で p50=index24, p95=index47 になる n=50 の遅延列。"""
    return [p50 - 200] * 24 + [p50] + [p95 - 100] * 22 + [p95] * 3


def fill_g1(conn, lats, gpu1_load=False, t0=1000):
    t = t0
    for i, lat in enumerate(lats):
        for stage, off in (("eot_wait", 0), ("asr_done", lat // 4), ("llm_first_clause", lat // 2),
                           ("tts_first", lat * 3 // 4), ("play_start", lat)):
            add(conn, t + off, stage, {"turn": i + t0, "gpu1_load": gpu1_load})
        t += lat + 100


def test_recorder_five_rows_per_turn(tmp_path):
    s = Store(tmp_path / "r.db", db_backup=False)
    rec = MetricsRecorder(s)
    for turn in range(3):
        for k, stage in enumerate(STAGES):
            rec.mark(turn, stage, 1000 * turn + 10 * k)
    assert len(STAGES) == 5
    for stage in STAGES:
        assert s.query("SELECT COUNT(*) FROM events WHERE kind=?", (stage,))[0][0] == 3
    for turn in range(3):
        rows = s.query("SELECT kind FROM events WHERE json_extract(payload_json,'$.turn')=?", (turn,))
        assert sorted(r[0] for r in rows) == sorted(STAGES)
    s.close()
    with pytest.raises(ValueError):
        MetricsRecorder(None).mark(0, "bogus", 1)


def test_percentile_nearest_rank():
    assert percentile(list(range(1, 51)), 0.5) == 25
    assert percentile(list(range(1, 51)), 0.95) == 48


def test_g1_pass_boundary(tmp_path):
    c = make_conn(tmp_path)
    fill_g1(c, lat_list(1200, 2000))
    r = evaluate_gate(c, "g1")
    assert r.passed and r.detail["p50"] == 1200 and r.detail["p95"] == 2000


@pytest.mark.parametrize("p50,p95", [(1201, 2000), (1200, 2001)])
def test_g1_fail_just_over(tmp_path, p50, p95):
    c = make_conn(tmp_path)
    fill_g1(c, lat_list(p50, p95))
    assert not evaluate_gate(c, "g1").passed


def test_g1_fail_insufficient_samples(tmp_path):
    c = make_conn(tmp_path)
    fill_g1(c, lat_list(1200, 2000)[:49])
    assert not evaluate_gate(c, "g1").passed


def test_g2_boundary(tmp_path):
    ok = make_conn(tmp_path, "ok.db")
    ng = make_conn(tmp_path, "ng.db")
    base = [100] * 47
    for i, v in enumerate(base + [250, 250, 250]):
        add(ok, i, "g2_sample", {"latency_ms": v})
    for i, v in enumerate(base + [251, 251, 251]):
        add(ng, i, "g2_sample", {"latency_ms": v})
    assert evaluate_gate(ok, "g2").passed
    assert not evaluate_gate(ng, "g2").passed


def playback(conn, ts, epoch, uid, cid, state, pos, cur_epoch=None):
    add(conn, ts, "playback", {"epoch": epoch, "utterance_id": uid, "chunk_id": cid,
                               "state": state, "position_ms": pos},
        epoch=epoch if cur_epoch is None else cur_epoch)


def test_g3_zero_stale_passes_and_one_fails(tmp_path):
    c = make_conn(tmp_path)
    playback(c, 1, 1, 1, 0, "started", 0)
    playback(c, 2, 1, 1, 0, "finished", 500)
    assert evaluate_gate(c, "g3").passed
    playback(c, 3, 1, 1, 1, "started", 0, cur_epoch=2)  # 旧世代が現行 epoch=2 で再生開始
    r = evaluate_gate(c, "g3")
    assert not r.passed and r.detail["stale"] == 1


def history_fixture(conn):
    playback(conn, 1, 1, 10, 0, "started", 0)
    playback(conn, 2, 1, 10, 0, "finished", 800)
    playback(conn, 3, 1, 11, 0, "started", 0)
    playback(conn, 4, 1, 11, 0, "finished", 700)
    playback(conn, 5, 1, 11, 1, "started", 0)
    playback(conn, 6, 1, 11, 1, "dropped", 300)  # 中断
    conn.execute("INSERT INTO turns(ts,role,text,completed,stop_position_ms,utterance_id) "
                 "VALUES (7,'ai','a',1,NULL,10)")
    conn.commit()


def test_history_consistent_passes(tmp_path):
    c = make_conn(tmp_path)
    history_fixture(c)
    c.execute("INSERT INTO turns(ts,role,text,completed,stop_position_ms,utterance_id) "
              "VALUES (8,'ai','b',0,300,11)")
    c.commit()
    r = evaluate_gate(c, "history")
    assert r.passed, r.detail


def test_history_one_mismatch_fails(tmp_path):
    c = make_conn(tmp_path)
    history_fixture(c)
    c.execute("INSERT INTO turns(ts,role,text,completed,stop_position_ms,utterance_id) "
              "VALUES (8,'ai','b',0,301,11)")  # 実再生位置 300 とずれる
    c.commit()
    r = evaluate_gate(c, "history")
    assert not r.passed and r.detail["mismatches"] == 1


def test_history_played_but_missing_turn_fails(tmp_path):
    c = make_conn(tmp_path)
    history_fixture(c)  # utterance 11 の finished チャンクが turns に無い
    assert not evaluate_gate(c, "history").passed


def test_e9_gpu1_load_aggregated_separately(tmp_path):
    ok = make_conn(tmp_path, "ok.db")
    fill_g1(ok, lat_list(1200, 2000), gpu1_load=True)
    assert evaluate_gate(ok, "e9").passed
    assert not evaluate_gate(ok, "g1").passed  # 負荷サンプルは g1 集計に入らない
    ng = make_conn(tmp_path, "ng.db")
    fill_g1(ng, lat_list(1201, 2000), gpu1_load=True)
    assert not evaluate_gate(ng, "e9").passed


def test_backchannel_excluded_from_g1(tmp_path):
    c = make_conn(tmp_path)
    fill_g1(c, lat_list(1200, 2000))
    base = evaluate_gate(c, "g1").detail["p50"]
    # 相槌は短遅延。含めると p50 が下がる件数を追加
    for i in range(60):
        add(c, 10**7 + i, "backchannel", {"turn": 9000 + i, "latency_ms": 100})
    r = evaluate_gate(c, "g1")
    assert r.detail["p50"] == base == 1200 and r.detail["n"] == 50
    with_bc = [x for x in lat_list(1200, 2000)] + [100] * 60
    assert percentile(with_bc, 0.5) != base  # 除外前後で p50 が変わる


def test_unknown_gate_raises(tmp_path):
    with pytest.raises(ValueError):
        evaluate_gate(make_conn(tmp_path), "g9")


# --- run_meta 出所検証 ---
def meta(**over):
    m = {"host": "h", "started_at": 0, "ended_at": MIN_DUR, "app_commit": "abc", "sample_count": 3}
    m.update(over)
    return m


def good_db(tmp_path, name="p.db", **over):
    c = make_conn(tmp_path, name)
    for ts in (10, 20, 30):
        add(c, ts, "g2_sample", {"latency_ms": 1})
    add(c, 40, "run_meta", meta(**over))
    return c


def test_run_meta_valid_no_exception(tmp_path):
    m = validate_run_meta(good_db(tmp_path), MIN_DUR, None)
    assert set(m) >= {"host", "started_at", "ended_at", "app_commit", "sample_count"}
    validate_run_meta(good_db(tmp_path, "q.db"), MIN_DUR, 1000)


def test_run_meta_requires_five_keys(tmp_path):
    c = make_conn(tmp_path)
    for ts in (10, 20, 30):
        add(c, ts, "g2_sample", {})
    m = meta()
    del m["app_commit"]
    add(c, 40, "run_meta", m)
    with pytest.raises(ProvenanceError):
        validate_run_meta(c, MIN_DUR)


def test_negative_no_run_meta_row(tmp_path):
    c = make_conn(tmp_path)
    add(c, 10, "g2_sample", {})
    with pytest.raises(ProvenanceError):
        validate_run_meta(c, MIN_DUR)


def test_negative_duration_one_ms_short(tmp_path):
    c = good_db(tmp_path, ended_at=MIN_DUR - 1)
    with pytest.raises(ProvenanceError):
        validate_run_meta(c, MIN_DUR)


def test_negative_sample_count_off_by_one(tmp_path):
    for bad in (2, 4):
        with pytest.raises(ProvenanceError):
            validate_run_meta(good_db(tmp_path, f"b{bad}.db", sample_count=bad), MIN_DUR)


def test_negative_ts_not_monotonic(tmp_path):
    c = make_conn(tmp_path)
    for ts in (10, 30, 20):
        add(c, ts, "g2_sample", {})
    add(c, 40, "run_meta", meta())
    with pytest.raises(ProvenanceError):
        validate_run_meta(c, MIN_DUR)


def test_negative_interval_exceeded(tmp_path):
    with pytest.raises(ProvenanceError):
        validate_run_meta(good_db(tmp_path), MIN_DUR, 5)
