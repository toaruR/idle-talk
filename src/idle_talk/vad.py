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

# 自動校正 (adaptive): 周囲の雑音と、AI 再生中の回り込み (AEC の消し残り) の大きさを dB で追う
NOISE_WARMUP_MS = 1000     # 雑音の推定に使った非再生フレームがこの長さに満たない間は固定の threshold を使う
NOISE_DOWN_TAU_MS = 500    # 雑音の推定は静かな方へは速く追う (時定数)
NOISE_UP_DB_PER_S = 0.5    # うるさい方へはゆっくり追う (発話で雑音の推定が上がりすぎないように)
ECHO_TAU_MS = 2000         # 再生中の回り込みの平均 (時定数)。再生をまたいで保つ。ドラマの音 (bleed) も同じ時定数で別に測る


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


def _db(level: float) -> float:
    return 20 * math.log10(max(level, 1.0))


def _lin(db: float) -> float:
    return 10 ** (db / 20)


class EnergyVad:
    """CPU 実装: RMS が threshold 以上のフレームが onset_frames 連続したら発話開始、hangover 連続無音フレームで終了。

    onset_frames > 1 なら一瞬の物音 (クリック・衝突音) では発話開始にしない。開始までの onset_frames - 1 フレームは
    Ingress が直前の音として区間の頭に足す。

    adaptive なら閾値を自動で決める: 非再生中は「雑音の推定 x ratio」(下限 min_threshold)、AI 再生中 (playing) は
    さらに「回り込みの平均 x ratio」以上。雑音は静かな方へ速く・うるさい方へゆっくり追う (発話は断続的なので床に戻る)。
    推定がそろうまで (NOISE_WARMUP_MS) は threshold を使う。playing は Director が音声の送出状況から毎フレーム設定する。
    bleed はドラマモードで共有したタブの音が鳴っている間 (Director が設定)。タブの音はブラウザの AEC で消えないので
    スピーカーだと丸ごと回り込む。AI の回り込みとは大きさが違うので別に測り、同じく「x ratio」以上に閾値を上げる。"""

    def __init__(self, threshold: float = 500.0, hangover_frames: int = 10, onset_frames: int = 1,
                 adaptive: bool = False, ratio: float = 3.0, min_threshold: float = 100.0) -> None:
        self.threshold = threshold
        self.hangover = hangover_frames
        self.onset_frames = max(1, onset_frames)
        self.adaptive = adaptive
        self.ratio = ratio
        self.min_threshold = min_threshold
        self.playing = False
        self.bleed = False
        self._speaking = False
        self._quiet = 0
        self._loud = 0
        self._noise_db: float | None = None
        self._noise_ms = 0
        self._echo_db: float | None = None
        self._bleed_db: float | None = None

    def reset(self) -> None:
        """新しい接続では発話・音量の判定を取り直す。"""
        self.playing = self.bleed = self._speaking = False
        self._quiet = self._loud = 0
        self._noise_db = self._echo_db = self._bleed_db = None
        self._noise_ms = 0

    @property
    def current_threshold(self) -> float:
        if not self.adaptive or self._noise_db is None or self._noise_ms < NOISE_WARMUP_MS:
            return self.threshold
        t = max(self.min_threshold, _lin(self._noise_db) * self.ratio)
        if self.playing and self._echo_db is not None:
            t = max(t, _lin(self._echo_db) * self.ratio)
        if self.bleed and self._bleed_db is not None:
            t = max(t, _lin(self._bleed_db) * self.ratio)
        return t

    def _track(self, level: float) -> None:
        db = _db(level)
        if self.playing:  # 回り込みは再生中だけ測る。雑音の推定には混ぜない
            a = FRAME_MS / ECHO_TAU_MS
            self._echo_db = db if self._echo_db is None else self._echo_db + a * (db - self._echo_db)
            return
        if self.bleed:  # ドラマの音の回り込みも同様 (雑音の推定に混ぜると静かな場面で閾値が高すぎる)
            a = FRAME_MS / ECHO_TAU_MS
            self._bleed_db = db if self._bleed_db is None else self._bleed_db + a * (db - self._bleed_db)
            return
        self._noise_ms += FRAME_MS
        if self._noise_db is None:
            self._noise_db = db
        elif db < self._noise_db:
            self._noise_db += (db - self._noise_db) * FRAME_MS / NOISE_DOWN_TAU_MS
        else:
            self._noise_db += min(db - self._noise_db, NOISE_UP_DB_PER_S * FRAME_MS / 1000)

    def process(self, frame: bytes) -> str | None:
        level = rms(frame)
        loud = level >= self.current_threshold
        if self.adaptive:
            self._track(level)
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
