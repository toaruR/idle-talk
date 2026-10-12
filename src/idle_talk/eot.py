"""終話判定インターフェースと G4 比較 (設計書 §9 / §10 M2 / §1.2 G4)。

既定は固定無音 (FixedSilence)。Smart Turn v3.2 相当のアダプタと、早切り件数での比較を提供する。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .budget import g4_pass as _budget_g4_pass
from .config import Config
from .m0_rules import eot_choice

G4_TRIALS = 50  # §1.2 G4 は 50 試行


class EndOfTurn(Protocol):
    """終話判定。silence_ms (継続無音) と任意の音声から終話か否かを返す。"""

    def is_end(self, silence_ms: int, audio: bytes | None = None) -> bool: ...


class FixedSilence:
    """固定無音: 無音が T_silence 以上 (>=) で終話。"""

    def __init__(self, t_silence: int | None = None, cfg: Config | None = None) -> None:
        self.t_silence = t_silence if t_silence is not None else (cfg or Config()).T_silence

    def is_end(self, silence_ms: int, audio: bytes | None = None) -> bool:
        return silence_ms >= self.t_silence


class SmartTurn:
    """Smart Turn v3.2 相当アダプタ。model(audio) -> 終話確率。単体テストでは偽モデルを注入する。"""

    def __init__(self, model: Callable[[bytes], float], threshold: float = 0.5) -> None:
        self.model = model
        self.threshold = threshold

    def is_end(self, silence_ms: int, audio: bytes | None = None) -> bool:
        if audio is None:
            return False
        return self.model(audio) >= self.threshold


def g4_pass(early_cuts: int) -> bool:
    """G4: 50 試行で早切り 2 件以下は合格、3 件以上は不合格。"""
    return _budget_g4_pass(early_cuts)


def compare_g4(fixed_cuts: int, smart_turn_cuts: int) -> dict:
    """両方式の早切り件数を並べ、少ない方を推奨 (同数は fixed)。T006 の eot_choice と同一。"""
    return {
        "fixed": fixed_cuts,
        "smart_turn": smart_turn_cuts,
        "fixed_pass": g4_pass(fixed_cuts),
        "smart_turn_pass": g4_pass(smart_turn_cuts),
        "recommend": eot_choice(fixed_cuts, smart_turn_cuts),
    }
