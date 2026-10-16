"""画面モード (設計書 6.7)。共有した画面を interval_s ごとに見て、前に話したときから変わっていれば一言言うかを決める。

FSM には触れない純ロジック。画面の取得・LLM・再生は Director が行う。
変化の量はブラウザが送る縮小画像 (THUMB_W x THUMB_H の明るさ 1 バイトずつ) の平均差で測る。画像は保存しない。
"""
from __future__ import annotations

import base64
import binascii
import time
from collections.abc import Callable

from .config import Screen

THUMB_W, THUMB_H = 32, 18  # web/client.js と一致させる


def decode_thumb(b64: str | None) -> bytes | None:
    """ブラウザの縮小画像 (base64)。無い・壊れている・大きさ違いは None。"""
    if not b64:
        return None
    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None
    return raw if len(raw) == THUMB_W * THUMB_H else None


def thumb_diff(a: bytes, b: bytes) -> float:
    """2 枚の縮小画像の明るさの平均差 (%、0〜100)。"""
    return sum(abs(x - y) for x, y in zip(a, b)) * 100 / (255 * len(a))


class ScreenWatcher:
    def __init__(self, settings: Screen, clock: Callable[[], float] = time.monotonic) -> None:
        self.s = settings
        self.clock = clock
        self._last = clock()  # 前に画面を見に行った時刻
        self._base: bytes | None = None  # 前に LLM に見せた画面の縮小画像
        self._title: str | None = None  # 前に LLM に見せたウィンドウのタイトル (follow_window のとき)

    def due(self) -> bool:
        return self.clock() - self._last >= self.s.interval_s

    def mark_checked(self) -> None:
        """画面を見に行った (話さなかった場合も含む)。次は interval_s 後。"""
        self._last = self.clock()

    def diff(self, thumb: bytes | None, title: str | None = None) -> float:
        """前に LLM に見せた画面との差 (%)。比べられなければ (初回・縮小画像なし) とウィンドウが替わったら 100。"""
        if thumb is None or self._base is None or title != self._title:
            return 100.0
        return thumb_diff(thumb, self._base)

    def changed(self, thumb: bytes | None, title: str | None = None) -> bool:
        return self.diff(thumb, title) >= self.s.min_diff

    def mark_shown(self, thumb: bytes | None, title: str | None = None) -> None:
        """LLM に見せた (「なし」で黙った場合も含む)。以後はこの画面と比べる。"""
        self._base, self._title = thumb, title

    def start(self) -> None:
        self._base, self._title = None, None
        self._last = self.clock()

    def stop(self) -> None:
        self._base, self._title = None, None
