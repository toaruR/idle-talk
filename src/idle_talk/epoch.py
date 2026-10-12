"""epoch 管理 (設計書 3.2)。キャンセル確定時のみ +1 する単調増加 u32。"""
from __future__ import annotations

from .protocol import U32_MAX


class Epoch:
    def __init__(self, value: int = 0) -> None:
        if not 0 <= value <= U32_MAX:
            raise ValueError(f"epoch out of u32 range: {value}")
        self._value = value

    @property
    def current(self) -> int:
        return self._value

    def bump(self) -> int:
        """キャンセル確定時のみ呼ぶ。u32 上限では 0 へ巻き戻る。"""
        self._value = 0 if self._value == U32_MAX else self._value + 1
        return self._value

    def is_current(self, epoch: int) -> bool:
        """生成物の epoch が現在値と一致するか。不一致は到着時に破棄する。"""
        return epoch == self._value
