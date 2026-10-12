"""自発発話スケジューラ (設計書 6.1 / 8.1 / 4.3-5)。既定オフ・雑談モードのみ。

発火可否の判定と待ち時間の管理だけを持つ純ロジック。再生は Director が通常応答と同じ経路で行う。
沈黙は同意ではない: 無反応が続くたびに待ち時間を 2 倍に延ばし、決して短縮・連続発火しない。
"""
from __future__ import annotations

import datetime
from collections.abc import Callable

from .config import Config

CHAT_MODE = "chat"


def _minutes(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:5])


def local_minutes() -> int:
    n = datetime.datetime.now()
    return n.hour * 60 + n.minute


def in_quiet_hours(minute_of_day: int, start: str, end: str) -> bool:
    """[start, end) が休止時間帯。start > end は日跨ぎ。"""
    s, e = _minutes(start), _minutes(end)
    if s == e:
        return False
    if s < e:
        return s <= minute_of_day < e
    return minute_of_day >= s or minute_of_day < e


class Spontaneous:
    def __init__(self, cfg: Config | None = None, minutes: Callable[[], int] = local_minutes,
                 record: Callable[[str, dict], None] | None = None, mode: str = CHAT_MODE) -> None:
        self.cfg = cfg or Config()
        self.enabled = self.cfg.spontaneous_enabled
        self.freq_per_hour = self.cfg.freq_per_hour
        self.quiet_start = self.cfg.quiet_start
        self.quiet_end = self.cfg.quiet_end
        self.mode = mode
        self._minutes = minutes
        self._record = record or (lambda kind, payload: None)
        self.silent_count = 0  # 無反応で終わった自発発話の連続回数
        self.awaiting_reaction = False
        self.pending = False  # 準備中 (発話生成・合成中)
        self.fired = 0
        self.cancelled = 0
        self.intervals: list[float] = []  # 発火直後に設定した次回までの待ち時間 (秒)
        self.next_at: float | None = None

    # ---- 設定 ----
    def apply_settings(self, s: dict) -> None:
        """core.settings (検証済み) を取り込む。範囲外は ValueError (5006 相当)。"""
        f = s.get("freq_per_hour", self.freq_per_hour)
        if not isinstance(f, int) or isinstance(f, bool) or not 0 <= f <= 12:
            raise ValueError(f"freq_per_hour must be in [0, 12]: {f}")
        Config(freq_per_hour=f, quiet_start=s.get("quiet_start", self.quiet_start),
               quiet_end=s.get("quiet_end", self.quiet_end))  # HH:MM 検証
        self.freq_per_hour = f
        self.quiet_start = s.get("quiet_start", self.quiet_start)
        self.quiet_end = s.get("quiet_end", self.quiet_end)
        if "on" in s:
            self.enabled = bool(s["on"])
            if not self.enabled:
                self.next_at = None

    @property
    def base_interval(self) -> float:
        return 3600.0 / self.freq_per_hour if self.freq_per_hour else float("inf")

    def is_quiet(self) -> bool:
        return in_quiet_hours(self._minutes(), self.quiet_start, self.quiet_end)

    # ---- 発火判定 ----
    def should_fire(self, now: float, idle: bool) -> bool:
        if not self.enabled or self.mode != CHAT_MODE or not idle or self.pending:
            return False
        if self.freq_per_hour <= 0 or self.is_quiet():
            return False
        if self.next_at is None:  # 有効化後の初回は 1 周期待つ
            self.next_at = now + self.base_interval
            return False
        return now >= self.next_at

    def fire(self, now: float) -> None:
        """発火確定。直前の自発発話に反応が無ければ待ち時間を倍にする。"""
        if self.awaiting_reaction:
            self.silent_count += 1
        self.pending, self.awaiting_reaction = True, True
        self.fired += 1
        interval = self.base_interval * (2 ** self.silent_count)
        self.intervals.append(interval)
        self.next_at = now + interval
        self._record("spontaneous_fire", {"silent_count": self.silent_count, "interval_s": interval})

    def done(self) -> None:
        self.pending = False

    def cancel(self) -> None:
        """準備中に利用者発話があった: 取消 (再生しない)。"""
        if self.pending:
            self.pending = False
            self.cancelled += 1
            self._record("spontaneous_cancel", {})

    def on_user_speech(self) -> None:
        """利用者の反応。待ち時間を基本に戻す。"""
        self.awaiting_reaction = False
        self.silent_count = 0
