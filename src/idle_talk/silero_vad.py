"""文字起こしモードの発話検出 (設計書 6.10)。Silero VAD (silero_vad_v6.onnx、MIT) の声らしさで区間を切る。

音量の VAD は BGM と声を区別できず、途切れないテレビの音を文の途中で切ったり何文もつなげたりした。テレビの 80 クリップを
BGM つきでつなげた音 (tools/stream_bench.py) で CER 18.4% → 13.7%、BGM なしで 25.7% → 12.8% (正しい区切りなら約 10.6%)。
EnergyVad と同じく 20ms のフレームごとに process を呼ぶ。声らしさは 512 サンプル (32ms) ごとに前の状態を渡して計算する。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .vad import SPEECH_END, SPEECH_START

CHUNK = 512  # Silero の 1 回の入力 (16kHz で 32ms)
CONTEXT = 64  # 前のチャンクの末尾を頭に足す長さ (faster-whisper の SileroVADModel と同じ)
SPEECH_ON = 0.5  # 声らしさがこれ以上で発話開始
SPEECH_OFF = 0.35  # これ未満が MIN_SILENCE_MS 続いたら発話の終わり
MIN_SILENCE_MS = 300  # 150 / 300 / 500ms で CER はほぼ同じ (2026-10-28)


def load_session(path: str | Path):
    import onnxruntime as ort

    if not Path(path).is_file():
        raise FileNotFoundError(f"silero vad model not found: {path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.log_severity_level = 4
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class SileroVad:
    """Vad (process(frame) -> SPEECH_START / SPEECH_END / None)。session は load_session の戻り値 (状態は持たないので共有できる)。"""

    onset_frames = 1  # 開始の判定は声らしさで行う (Ingress は前の 300ms も ASR に渡す)
    current_threshold = 0.0  # 音量の閾値は使わない (EnergyVad と同じ名前で参照される)

    def __init__(self, session) -> None:
        self.session = session
        self.playing = self.bleed = False  # Director が設定する (使わない)
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self.prob = 0.0  # 直近の声らしさ
        self._speaking = False
        self._quiet_ms = 0

    def _run(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self._ctx, chunk])[None, :]
        out, self._h, self._c = self.session.run(None, {"input": x, "h": self._h, "c": self._c})
        self._ctx = chunk[-CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])

    def process(self, frame: bytes) -> str | None:
        self._pending = np.concatenate([self._pending, np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768])
        while len(self._pending) >= CHUNK:
            self.prob = self._run(self._pending[:CHUNK])
            self._pending = self._pending[CHUNK:]
        frame_ms = len(frame) // 32
        if not self._speaking:
            if self.prob >= SPEECH_ON:
                self._speaking, self._quiet_ms = True, 0
                return SPEECH_START
            return None
        self._quiet_ms = self._quiet_ms + frame_ms if self.prob < SPEECH_OFF else 0
        if self._quiet_ms >= MIN_SILENCE_MS:
            self._speaking = False
            return SPEECH_END
        return None
