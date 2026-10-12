"""割り込み判定 (設計書 5.1)。純粋関数のみ。"""
from __future__ import annotations

import unicodedata

from .config import Config

BACKCHANNEL_WORDS: frozenset[str] = frozenset(
    {"うん", "うんうん", "へえ", "ふーん", "ええ", "はい", "なるほど", "そう", "そっか"}
)

_STRIP = " 　、。,.!！"


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().strip(_STRIP)


def is_stop_word(text: str, cfg: Config | None = None) -> bool:
    cfg = cfg or Config()
    t = normalize(text)
    return any(w == t or (w in t and len(t) <= len(w) + 2) for w in cfg.STOP_WORDS)


def initial_action(text: str, cfg: Config | None = None) -> str:
    """AI再生中の VAD/ASR 語に対する即時動作。停止語は duck を経由せず cancel。"""
    return "cancel" if is_stop_word(text, cfg) else "duck"


def classify(text: str) -> str:
    """'backchannel' | 'interrupt' | 'unknown'。疑問/訂正/判断不能は相槌にしない。"""
    raw = unicodedata.normalize("NFKC", text).strip()
    if not raw:
        return "unknown"
    if raw.endswith(("?", "？")):
        return "unknown"
    return "backchannel" if normalize(raw) in BACKCHANNEL_WORDS else "unknown"


def in_confirm_window(start_ms: int, t_ms: int, cfg: Config | None = None) -> bool:
    """半開区間 [start, start + T_confirm)。"""
    cfg = cfg or Config()
    return start_ms <= t_ms < start_ms + cfg.T_confirm


def is_strong_utt(user_volume: float, ai_avg_volume: float, continues_to_window_end: bool,
                  cfg: Config | None = None) -> bool:
    cfg = cfg or Config()
    if continues_to_window_end:
        return True
    if ai_avg_volume <= 0:
        return user_volume > 0
    return user_volume + 1e-9 >= ai_avg_volume * cfg.STRONG_UTT.volume_ratio


def judge(text: str, start_ms: int, t_ms: int, cfg: Config | None = None) -> str:
    """確認窓での最終判定。'backchannel'(resume) | 'interrupt' | 'unknown'(cancel、停止したまま)。"""
    cfg = cfg or Config()
    if is_stop_word(text, cfg):
        return "interrupt"
    if classify(text) == "backchannel" and in_confirm_window(start_ms, t_ms, cfg):
        return "backchannel"
    return "unknown"
