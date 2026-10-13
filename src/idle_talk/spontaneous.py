"""自発発話スケジューラ (設計書 6.1 / 8.1 / 4.3-5)。既定オフ・雑談モードのみ。

発火可否の判定と待ち時間の管理だけを持つ純ロジック。再生は Director が通常応答と同じ経路で行う。
待ち時間は「IDLE になってから (直前の発話が終わってから) の沈黙秒数」で、基本値は interval_s。
沈黙は同意ではない: 無反応が続くたびに待ち時間を 2 倍に延ばし、決して短縮・連続発火しない。
ラジオモードは返答を期待しない独り語り: 待ち時間は RADIO_GAP_S 固定で、無反応でも延ばさない (on がオフでも発火する)。
"""
from __future__ import annotations

import datetime
from collections.abc import Callable

from .config import Config

CHAT_MODE = "chat"
RADIO_GAP_S = 3.0  # ラジオモードで 1 本話し終えてから次を話し始めるまでの沈黙秒数


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
        self.interval_s = self.cfg.spont_interval_s
        self.quiet_start = self.cfg.quiet_start
        self.quiet_end = self.cfg.quiet_end
        self.radio = self.cfg.radio
        self.mode = mode
        self._minutes = minutes
        self._record = record or (lambda kind, payload: None)
        self.silent_count = 0  # 無反応で終わった自発発話の連続回数
        self.awaiting_reaction = False
        self.pending = False  # 準備中 (発話生成・合成中)
        self.fired = 0
        self.cancelled = 0
        self.intervals: list[float] = []  # 各発火で使った待ち時間 (秒)

    # ---- 設定 ----
    def apply_settings(self, s: dict) -> None:
        """core.settings (検証済み) を取り込む。範囲外は ValueError (5006 相当)。"""
        iv = s.get("interval_s", self.interval_s)
        Config(spont_interval_s=iv, quiet_start=s.get("quiet_start", self.quiet_start),
               quiet_end=s.get("quiet_end", self.quiet_end))  # 範囲・HH:MM 検証
        self.interval_s = iv
        self.quiet_start = s.get("quiet_start", self.quiet_start)
        self.quiet_end = s.get("quiet_end", self.quiet_end)
        if "on" in s:
            self.enabled = bool(s["on"])
        if "radio" in s:
            self.radio = bool(s["radio"])

    @property
    def wait_s(self) -> float:
        """次の発火までに必要な沈黙秒数。直前の自発発話が無反応なら 2 倍ずつ延びる (ラジオは固定)。"""
        if self.radio:
            return RADIO_GAP_S
        return float(self.interval_s * 2 ** (self.silent_count + int(self.awaiting_reaction)))

    def is_quiet(self) -> bool:
        return in_quiet_hours(self._minutes(), self.quiet_start, self.quiet_end)

    # ---- 発火判定 ----
    def should_fire(self, now: float, idle_since: float | None) -> bool:
        """idle_since: IDLE になった時刻 (IDLE でなければ None)。"""
        if not (self.enabled or self.radio) or self.mode != CHAT_MODE or idle_since is None or self.pending:
            return False
        if self.is_quiet():
            return False
        return now - idle_since >= self.wait_s

    def fire(self, now: float) -> None:
        """発火確定。直前の自発発話に反応が無ければ無反応回数を数える (待ち時間が倍になる)。ラジオは反応を待たない。"""
        wait = self.wait_s
        if self.awaiting_reaction and not self.radio:
            self.silent_count += 1
        self.pending, self.awaiting_reaction = True, not self.radio
        self.fired += 1
        self.intervals.append(wait)
        self._record("spontaneous_fire", {"silent_count": self.silent_count, "interval_s": wait,
                                          "radio": self.radio})

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
