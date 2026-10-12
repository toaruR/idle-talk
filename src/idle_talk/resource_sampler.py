"""常駐試験(M4/G5)ハーネス (設計書 11.1 常駐 G5 / 1.2 G5 / 10 M4 / 3.4 events)。

test-harness-only: CLI なし。RSS・VRAM を 1 分ごとに events(kind='soak_sample') へ記録し、
analyze_soak が RSS の回帰傾きで G5 を判定する。出所検証は metrics.validate_run_meta を再利用する。
payload_json = {"rss_mib": float, "vram_mib": float}。
"""
from __future__ import annotations

import itertools
import json
import os
import sqlite3
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .metrics import ProvenanceError, validate_run_meta

SOAK_KIND = "soak_sample"
SAMPLE_INTERVAL_MS = 60_000                         # §11.1 '1 分ごと'
SOAK_DURATION_MS = 8 * 60 * 60 * 1000               # §1.2 G5 / §10 M4 の 8 時間
SOAK_MIN_SAMPLES = SOAK_DURATION_MS // SAMPLE_INTERVAL_MS  # 480
G5_MAX_SLOPE_MIB_PER_H = 1.0                        # §1.2 G5
_MS_PER_HOUR = 3_600_000
_EPS = 1e-9


def read_rss_mib() -> float:
    """現プロセスの RSS(MiB)。psutil があれば使い、無ければ /proc を読む。"""
    try:
        import psutil  # type: ignore
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    raise RuntimeError("RSS unavailable")


def read_vram_mib() -> float:
    """GPU の使用 VRAM 合計(MiB)。nvidia-smi 経由。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10).stdout
    return float(sum(float(x) for x in out.split() if x.strip()))


class ResourceSampler:
    """sample() 1 回で RSS・VRAM を 1 行記録する。clock は ms を返す (試験では注入)。"""

    def __init__(self, store, clock: Callable[[], int] | None = None,
                 rss_fn: Callable[[], float] = read_rss_mib,
                 vram_fn: Callable[[], float] = read_vram_mib, epoch: int = 0) -> None:
        self.store = store
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.rss_fn = rss_fn
        self.vram_fn = vram_fn
        self.epoch = epoch
        self.count = 0

    def sample(self) -> int:
        ts = int(self.clock())
        payload = {"rss_mib": float(self.rss_fn()), "vram_mib": float(self.vram_fn())}
        self.count += 1
        return self.store.add_event(ts, self.epoch, SOAK_KIND, json.dumps(payload))


@dataclass
class SoakResult:
    passed: bool
    slope_mib_per_h: float
    n: int
    detail: str


def rss_slope_mib_per_h(rows: Sequence[dict]) -> float:
    """rows: [{"ts": ms, "rss_mib": float}, ...] の最小二乗回帰傾き (MiB/h)。"""
    n = len(rows)
    if n < 2:
        raise ValueError("need >= 2 rows")
    xs = [r["ts"] / _MS_PER_HOUR for r in rows]
    ys = [float(r["rss_mib"]) for r in rows]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        raise ValueError("ts has no spread")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def analyze_soak(rows: Sequence[dict]) -> SoakResult:
    """G5: RSS 回帰傾き <= 1.0 MiB/h で合格。"""
    try:
        slope = rss_slope_mib_per_h(rows)
    except ValueError as e:
        return SoakResult(False, float("nan"), len(rows), str(e))
    ok = slope <= G5_MAX_SLOPE_MIB_PER_H + _EPS
    return SoakResult(ok, slope, len(rows),
                      f"slope {slope:.4f} MiB/h (limit {G5_MAX_SLOPE_MIB_PER_H})")


def load_soak_rows(conn: sqlite3.Connection) -> list[dict]:
    rows = []
    for ts, pj in conn.execute(
            "SELECT ts, payload_json FROM events WHERE kind=? ORDER BY id", (SOAK_KIND,)):
        p = json.loads(pj)
        rows.append({"ts": ts, "rss_mib": p["rss_mib"], "vram_mib": p["vram_mib"]})
    return rows


def validate_soak(conn: sqlite3.Connection) -> dict:
    """validate_run_meta(8h, interval_ms=60000) + ts 厳密増加 + 480 行以上。違反は ProvenanceError。"""
    meta = validate_run_meta(conn, SOAK_DURATION_MS, interval_ms=SAMPLE_INTERVAL_MS)
    ts = [r[0] for r in conn.execute("SELECT ts FROM events WHERE kind=? ORDER BY id", (SOAK_KIND,))]
    if len(ts) < SOAK_MIN_SAMPLES:
        raise ProvenanceError(f"soak_sample rows {len(ts)} < {SOAK_MIN_SAMPLES}")
    if any(b <= a for a, b in itertools.pairwise(ts)):
        raise ProvenanceError("soak_sample ts is not strictly increasing")
    return meta
