"""合成した声のエフェクト ([voice_fx])。ロボットのような声質にする。

順に フォルマントシフト -> リングモジュレーション -> コムフィルタ / フランジャー を掛ける。
入出力は AivisSpeech の WAV (16bit モノラル)。読めない形式・失敗はそのまま返す (声が出ないより良い)。
"""
from __future__ import annotations

import io
import logging
import wave

import numpy as np

from .config import VoiceFx

log = logging.getLogger("idle_talk.voice_fx")

N_FFT = 1024   # 24kHz で約 43ms
HOP = 256
LIFTER = 30    # ケプストラムで包絡に残す次数 (声の基本周期 約 2.5ms 以上を除く)
PEAK = 0.98    # 書き出しの上限 (フルスケール比)


def is_active(fx: VoiceFx) -> bool:
    return fx.on and (fx.formant != 1.0 or (fx.ring_hz > 0 and fx.ring_mix > 0)
                      or (fx.comb_ms > 0 and fx.comb_mix > 0))


def apply(wav: bytes, fx: VoiceFx) -> bytes:
    """WAV にエフェクトを掛けた WAV を返す。無効・読めない形式ならそのまま。"""
    if not is_active(fx):
        return wav
    try:
        with wave.open(io.BytesIO(wav)) as w:
            ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            data = w.readframes(n)
        if ch != 1 or width != 2:
            log.warning("voice_fx skipped: %d ch, %d byte", ch, width)
            return wav
        x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
        y = process(x, rate, fx)
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(y, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
        return out.getvalue()
    except (wave.Error, EOFError, ValueError) as e:
        log.warning("voice_fx failed (kept dry): %s", e)
        return wav


def process(x: np.ndarray, rate: int, fx: VoiceFx) -> np.ndarray:
    """float (-1〜1) の信号に順にエフェクトを掛ける。元より大きくなれば元のピークに揃える。"""
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    y = x
    if fx.formant != 1.0:
        y = formant_shift(y, fx.formant)
    if fx.ring_hz > 0 and fx.ring_mix > 0:
        y = ring_mod(y, rate, fx.ring_hz, fx.ring_mix)
    if fx.comb_ms > 0 and fx.comb_mix > 0:
        if fx.flanger_hz > 0:
            y = flanger(y, rate, fx.comb_ms, fx.flanger_hz, fx.flanger_depth_ms, fx.comb_mix)
        else:
            y = comb(y, rate, fx.comb_ms, fx.comb_feedback, fx.comb_mix)
    out_peak = float(np.max(np.abs(y)))
    limit = min(max(peak, 1e-9), PEAK)
    if out_peak > limit:
        y = y * (limit / out_peak)
    return y


def formant_shift(x: np.ndarray, ratio: float) -> np.ndarray:
    """声の高さを変えずに響き (スペクトル包絡) だけを周波数方向に ratio 倍する。

    STFT の各フレームで包絡をケプストラムで取り出し、包絡だけを伸縮して元の微細構造 (声の高さ) に掛け直す。
    位相は元のまま。"""
    n = x.size
    pad = np.concatenate([np.zeros(N_FFT), x, np.zeros(N_FFT + HOP)])
    count = 1 + (pad.size - N_FFT) // HOP
    win = np.hanning(N_FFT)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(count)[:, None]
    spec = np.fft.rfft(pad[idx] * win, axis=1)
    mag = np.abs(spec) + 1e-9
    logmag = np.log(mag)
    cep = np.fft.irfft(logmag, n=N_FFT, axis=1)
    cep[:, LIFTER:N_FFT - LIFTER + 1] = 0.0
    env = np.fft.rfft(cep, axis=1).real
    bins = env.shape[1]
    src = np.arange(bins) / ratio
    i0 = np.clip(np.floor(src).astype(int), 0, bins - 1)
    i1 = np.clip(i0 + 1, 0, bins - 1)
    frac = np.clip(src - i0, 0.0, 1.0)
    warped = env[:, i0] * (1.0 - frac) + env[:, i1] * frac
    new = np.exp(logmag - env + warped) * np.exp(1j * np.angle(spec))
    frames = np.fft.irfft(new, n=N_FFT, axis=1) * win
    out = np.zeros(pad.size)
    norm = np.zeros(pad.size)
    np.add.at(out, idx, frames)
    np.add.at(norm, idx, np.broadcast_to(win ** 2, frames.shape))
    out = out / np.maximum(norm, 1e-6)
    return out[N_FFT:N_FFT + n]


def ring_mod(x: np.ndarray, rate: int, hz: float, mix: float) -> np.ndarray:
    """正弦波を掛け合わせる (mix=1 で完全なリングモジュレーション)。金属的な響きになる。"""
    t = np.arange(x.size) / rate
    return x * ((1.0 - mix) + mix * np.sin(2.0 * np.pi * hz * t))


def comb(x: np.ndarray, rate: int, delay_ms: float, feedback: float, mix: float) -> np.ndarray:
    """帰還つきコムフィルタ y[n] = x[n] + g * y[n - d]。筐体の中で鳴るような共鳴になる。"""
    d = max(1, int(round(rate * delay_ms / 1000.0)))
    y = x.copy()
    for k in range(d, x.size, d):  # d サンプルずつまとめて計算する (d 以上前の y だけを使う)
        e = min(k + d, x.size)
        y[k:e] += feedback * y[k - d:e - d]
    wet = y * (1.0 - feedback)  # 共鳴の山の高さ 1 / (1 - g) を戻す
    return (1.0 - mix) * x + mix * wet


def flanger(x: np.ndarray, rate: int, delay_ms: float, hz: float, depth_ms: float, mix: float) -> np.ndarray:
    """遅れを delay_ms〜delay_ms + depth_ms で揺らした音を重ねる (うねるジェット音)。"""
    n = np.arange(x.size)
    d = rate * (delay_ms + depth_ms * 0.5 * (1.0 - np.cos(2.0 * np.pi * hz * n / rate))) / 1000.0
    pos = n - d
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    valid = i0 >= 0
    i0c = np.clip(i0, 0, x.size - 1)
    i1c = np.clip(i0 + 1, 0, x.size - 1)
    delayed = np.where(valid, x[i0c] * (1.0 - frac) + x[i1c] * frac, 0.0)
    return (1.0 - mix) * x + mix * 0.5 * (x + delayed)
