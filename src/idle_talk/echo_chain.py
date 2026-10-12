"""M5 スピーカー AEC 実測の合否判定 (設計書 6.2 / 10 M5 / 12.3 E10 / 3.4 events)。

test-harness-only: CLI なし。出所検証は metrics.validate_run_meta を再利用する。
events の kind:
- echo_chain: 自声起因の応答連鎖 1 回につき 1 行 (件数 = 連鎖回数)。
- cancel: 割り込みによる再生取消 1 回につき 1 行。
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .metrics import validate_run_meta

ECHO_CHAIN_KIND = "echo_chain"
CANCEL_KIND = "cancel"
ECHO_MIN_DURATION_MS = 1_200_000  # §12.3 E10 の 20 分
MAX_CHAINS = 0                    # §12.3 E10: 連鎖 0 回
PIPECAT_RECOMMENDATION = "Pipecat 比較を推奨 (設計書 6.2: 自分の声に応答が連鎖しないことが合格条件)"


@dataclass(frozen=True)
class EchoChainResult:
    passed: bool
    chain_count: int
    cancel_count: int
    interruptible: bool
    recommendation: str
    detail: str


def evaluate_echo_chain(conn: sqlite3.Connection) -> EchoChainResult:
    """validate_run_meta(20 分) の後に連鎖回数と cancel 件数を判定する。不備は ProvenanceError。"""
    validate_run_meta(conn, min_duration_ms=ECHO_MIN_DURATION_MS)
    chains = conn.execute("SELECT COUNT(*) FROM events WHERE kind=?", (ECHO_CHAIN_KIND,)).fetchone()[0]
    cancels = conn.execute("SELECT COUNT(*) FROM events WHERE kind=?", (CANCEL_KIND,)).fetchone()[0]
    passed = chains <= MAX_CHAINS
    interruptible = passed and cancels >= 1
    if passed:
        rec = ""
        detail = f"chain=0 cancel={cancels} interruptible={interruptible}"
    else:
        rec = PIPECAT_RECOMMENDATION
        detail = f"chain={chains} > 0: {rec}"
    return EchoChainResult(passed, chains, cancels, interruptible, rec, detail)
