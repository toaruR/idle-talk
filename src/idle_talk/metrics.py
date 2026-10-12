"""M1 計測記録と合否判定 (設計書 11.1 / 1.2 / 10 / 12.3 E9)。

events テーブル(§3.4)に新形式を作らず kind と payload_json で表す:
- 段階 kind: eot_wait(発話終了時刻) asr_done llm_first_clause tts_first play_start。
  payload {"turn": int, "gpu1_load": bool}。1 ターン 5 行。G1 遅延 = play_start.ts - eot_wait.ts。
- backchannel: 事前合成相槌。G1 集計から除外する。
- g2_sample: payload {"latency_ms": int} (発話開始→ダッキング/停止開始)。
- playback: payload {epoch, utterance_id, chunk_id, state, position_ms}。events.epoch は受信時の現行 epoch。
  state='started' かつ payload.epoch < events.epoch を旧世代再生(G3 違反)とする。
- run_meta: payload {host, started_at, ended_at, app_commit, sample_count}。
"""
from __future__ import annotations

import itertools
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STAGES = ("eot_wait", "asr_done", "llm_first_clause", "tts_first", "play_start")
RUN_META_KEYS = ("host", "started_at", "ended_at", "app_commit", "sample_count")

G1_P50_MS = 1200
G1_P95_MS = 2000
G2_P95_MS = 250
G1_MIN_SAMPLES = 50  # §1.2 G1 の試行数
G2_MIN_SAMPLES = 50  # §1.2 G2 の試行数
GATES = ("g1", "g2", "g3", "history", "e9")


class ProvenanceError(Exception):
    """records DB の出所(run_meta)検証失敗。"""


@dataclass
class GateResult:
    gate: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.passed


# --- 記録 ---
class MetricsRecorder:
    """段階別タイムスタンプを events に kind 別で記録する。"""

    def __init__(self, store) -> None:
        self.store = store

    def mark(self, turn: int, stage: str, ts: int, epoch: int = 0, gpu1_load: bool = False) -> int:
        if stage not in STAGES:
            raise ValueError(stage)
        payload = json.dumps({"turn": turn, "gpu1_load": bool(gpu1_load)})
        return self.store.add_event(ts, epoch, stage, payload)

    def mark_backchannel(self, turn: int, ts: int, latency_ms: int, epoch: int = 0) -> int:
        payload = json.dumps({"turn": turn, "latency_ms": latency_ms})
        return self.store.add_event(ts, epoch, "backchannel", payload)

    def mark_g2(self, ts: int, latency_ms: int, epoch: int = 0) -> int:
        return self.store.add_event(ts, epoch, "g2_sample", json.dumps({"latency_ms": latency_ms}))


def open_readonly(path: str | Path) -> sqlite3.Connection:
    """既存 DB を読取専用で開く。無ければ作らず例外 (sqlite3.OperationalError)。"""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return sqlite3.connect(f"{p.resolve().as_uri()}?mode=ro", uri=True)


# --- 集計 ---
def percentile(values: list[float], q: float) -> float:
    """最近順位法 (nearest-rank)。"""
    if not values:
        raise ValueError("empty")
    s = sorted(values)
    return s[max(1, math.ceil(q * len(s))) - 1]


def _payload(raw: str | None) -> dict:
    try:
        d = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def _g1_latencies(conn: sqlite3.Connection) -> tuple[list[int], list[int]]:
    """(通常, gpu1_load) の G1 遅延リスト。backchannel 行は kind が段階外なので含まれない。"""
    marks: dict[int, dict[str, int]] = {}
    gpu: dict[int, bool] = {}
    q = ",".join("?" for _ in STAGES)
    for ts, kind, raw in conn.execute(
            f"SELECT ts, kind, payload_json FROM events WHERE kind IN ({q}) ORDER BY id", STAGES):
        p = _payload(raw)
        turn = p.get("turn")
        if not isinstance(turn, int):
            continue
        marks.setdefault(turn, {})[kind] = ts
        if p.get("gpu1_load") is True:
            gpu[turn] = True
    normal: list[int] = []
    loaded: list[int] = []
    for turn, m in marks.items():
        if "eot_wait" in m and "play_start" in m:
            (loaded if gpu.get(turn) else normal).append(m["play_start"] - m["eot_wait"])
    return normal, loaded


def _gate_g1(conn) -> GateResult:
    lat, _ = _g1_latencies(conn)
    if len(lat) < G1_MIN_SAMPLES:
        return GateResult("g1", False, {"n": len(lat), "reason": "insufficient samples"})
    p50, p95 = percentile(lat, 0.5), percentile(lat, 0.95)
    return GateResult("g1", p50 <= G1_P50_MS and p95 <= G1_P95_MS, {"n": len(lat), "p50": p50, "p95": p95})


def _gate_e9(conn) -> GateResult:
    _, lat = _g1_latencies(conn)
    if len(lat) < G1_MIN_SAMPLES:
        return GateResult("e9", False, {"n": len(lat), "reason": "insufficient samples"})
    p50 = percentile(lat, 0.5)
    return GateResult("e9", p50 <= G1_P50_MS, {"n": len(lat), "p50": p50})


def _gate_g2(conn) -> GateResult:
    lat = []
    for (raw,) in conn.execute("SELECT payload_json FROM events WHERE kind='g2_sample' ORDER BY id"):
        v = _payload(raw).get("latency_ms")
        if isinstance(v, (int, float)):
            lat.append(v)
    if len(lat) < G2_MIN_SAMPLES:
        return GateResult("g2", False, {"n": len(lat), "reason": "insufficient samples"})
    p95 = percentile(lat, 0.95)
    return GateResult("g2", p95 <= G2_P95_MS, {"n": len(lat), "p95": p95})


def _gate_g3(conn) -> GateResult:
    stale = 0
    for ep, raw in conn.execute("SELECT epoch, payload_json FROM events WHERE kind='playback' ORDER BY id"):
        p = _payload(raw)
        pe = p.get("epoch")
        if p.get("state") == "started" and isinstance(pe, int) and ep is not None and pe < ep:
            stale += 1
    return GateResult("g3", stale == 0, {"stale": stale})


def _gate_history(conn) -> GateResult:
    """turns(ai) と playback の照合。不整合件数 0 で合格。"""
    chunks: dict[int, dict[int, tuple[str, int]]] = {}
    for raw, in conn.execute("SELECT payload_json FROM events WHERE kind='playback' ORDER BY id"):
        p = _payload(raw)
        if not all(isinstance(p.get(k), int) for k in ("utterance_id", "chunk_id", "position_ms")):
            continue
        chunks.setdefault(p["utterance_id"], {})[p["chunk_id"]] = (p.get("state"), p["position_ms"])
    turns: dict[int, tuple[int, Any]] = {}
    mismatches: list[str] = []
    for uid, completed, stop in conn.execute(
            "SELECT utterance_id, completed, stop_position_ms FROM turns WHERE role='ai' ORDER BY id"):
        if uid in turns:
            mismatches.append(f"duplicate turn for utterance {uid}")
        turns[uid] = (completed, stop)
    for uid, (completed, stop) in turns.items():
        ch = chunks.get(uid)
        if ch is None:
            mismatches.append(f"turn {uid} has no playback")
            continue
        unfinished = [c for c, (st, _) in ch.items() if st != "finished"]
        if completed == 1:
            if unfinished or not ch:
                mismatches.append(f"turn {uid} completed but playback unfinished")
        else:
            ref = max(unfinished) if unfinished else max(ch)
            if stop != ch[ref][1]:
                mismatches.append(f"turn {uid} stop_position {stop} != playback {ch[ref][1]}")
    for uid, ch in chunks.items():
        if uid not in turns and any(st == "finished" for st, _ in ch.values()):
            mismatches.append(f"utterance {uid} played but not in turns")
    return GateResult("history", not mismatches, {"mismatches": len(mismatches), "items": mismatches[:10]})


_GATES = {"g1": _gate_g1, "g2": _gate_g2, "g3": _gate_g3, "history": _gate_history, "e9": _gate_e9}


def evaluate_gate(conn: sqlite3.Connection, gate: str) -> GateResult:
    if gate not in _GATES:
        raise ValueError(f"unknown gate: {gate}")
    return _GATES[gate](conn)


# --- 出所検証 ---
def validate_run_meta(conn: sqlite3.Connection, min_duration_ms: int, interval_ms: int | None = None) -> dict:
    """kind=run_meta 行の 5 キー・期間・ts 単調・sample_count を検証し payload を返す。

    interval_ms 指定時のみ隣接標本の ts 間隔上限を検査 (M1 に間隔規定はなく既定は None)。
    """
    metas = conn.execute("SELECT payload_json FROM events WHERE kind='run_meta'").fetchall()
    if len(metas) != 1:
        raise ProvenanceError(f"run_meta rows must be exactly 1, got {len(metas)}")
    try:
        meta = json.loads(metas[0][0])
    except (TypeError, ValueError) as e:
        raise ProvenanceError(f"run_meta payload is not JSON: {e}") from e
    if not isinstance(meta, dict):
        raise ProvenanceError("run_meta payload is not an object")
    missing = [k for k in RUN_META_KEYS if k not in meta]
    if missing:
        raise ProvenanceError(f"run_meta missing keys: {missing}")
    for k in ("host", "app_commit"):
        if not isinstance(meta[k], str) or not meta[k]:
            raise ProvenanceError(f"run_meta {k} must be non-empty string")
    for k in ("started_at", "ended_at", "sample_count"):
        if isinstance(meta[k], bool) or not isinstance(meta[k], int):
            raise ProvenanceError(f"run_meta {k} must be integer")
    if meta["ended_at"] - meta["started_at"] < min_duration_ms:
        raise ProvenanceError(
            f"duration {meta['ended_at'] - meta['started_at']}ms < min {min_duration_ms}ms")
    ts_list = [r[0] for r in conn.execute("SELECT ts FROM events WHERE kind!='run_meta' ORDER BY id")]
    if meta["sample_count"] != len(ts_list):
        raise ProvenanceError(f"sample_count {meta['sample_count']} != events rows {len(ts_list)}")
    for a, b in itertools.pairwise(ts_list):
        if a is None or b is None or b < a:
            raise ProvenanceError("events.ts is not monotonic non-decreasing")
        if interval_ms is not None and b - a > interval_ms:
            raise ProvenanceError(f"ts gap {b - a}ms > interval {interval_ms}ms")
    return meta
