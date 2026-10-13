"""VAD アダプタ (設計書 2.1: VAD は CPU)。差し替え可能なインターフェースと CPU 実装 (エネルギー閾値)。"""
from __future__ import annotations

import array
import math
from typing import Protocol

SPEECH_START = "speech_start"
SPEECH_END = "speech_end"

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # 16kHz mono s16le = 32 bytes/ms
FRAME_MS = 20  # 上り音声の 1 フレーム (director.FRAME_BYTES)


class Vad(Protocol):
    """1 フレームを処理し SPEECH_START / SPEECH_END / None を返す。"""

    def process(self, frame: bytes) -> str | None: ...


def rms(frame: bytes) -> float:
    n = len(frame) // 2
    if n == 0:
        return 0.0
    a = array.array("h")
    a.frombytes(frame[: n * 2])
    return math.sqrt(sum(x * x for x in a) / n)


class EnergyVad:
    """CPU 実装: RMS が threshold 以上のフレームが onset_frames 連続したら発話開始、hangover 連続無音フレームで終了。

    onset_frames > 1 なら一瞬の物音 (クリック・衝突音) では発話開始にしない。開始までの onset_frames - 1 フレームは
    Ingress が直前の音として区間の頭に足す。"""

    def __init__(self, threshold: float = 500.0, hangover_frames: int = 10, onset_frames: int = 1) -> None:
        self.threshold = threshold
        self.hangover = hangover_frames
        self.onset_frames = max(1, onset_frames)
        self._speaking = False
        self._quiet = 0
        self._loud = 0

    def process(self, frame: bytes) -> str | None:
        loud = rms(frame) >= self.threshold
        if not self._speaking:
            self._loud = self._loud + 1 if loud else 0
            if self._loud >= self.onset_frames:
                self._speaking = True
                self._quiet = 0
                self._loud = 0
                return SPEECH_START
            return None
        if loud:
            self._quiet = 0
            return None
        self._quiet += 1
        if self._quiet >= self.hangover:
            self._speaking = False
            self._quiet = 0
            return SPEECH_END
        return None
