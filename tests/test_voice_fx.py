import io
import os
import time
import wave

import numpy as np
import pytest

from idle_talk import voice_fx
from idle_talk.config import VoiceFx, load_voice_fx

RATE = 24000


def make_wav(x: np.ndarray, rate: int = RATE, ch: int = 1, width: int = 2) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


def read_wav(wav: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def voice_like(sec: float = 1.0, f0: float = 200.0) -> np.ndarray:
    """f0 の倍音に 800Hz 付近の山 (フォルマント) を持たせた信号。"""
    t = np.arange(int(RATE * sec)) / RATE
    x = sum(np.exp(-((k * f0 - 800) / 400) ** 2) * np.sin(2 * np.pi * k * f0 * t) for k in range(1, 40))
    return 0.5 * x / np.max(np.abs(x))


def spectrum_peak(x: np.ndarray) -> float:
    mag = np.abs(np.fft.rfft(x * np.hanning(x.size)))
    return float(np.argmax(mag) * RATE / x.size)


def test_off_returns_same_bytes():
    wav = make_wav(voice_like())
    assert voice_fx.apply(wav, VoiceFx()) is wav
    assert voice_fx.apply(wav, VoiceFx(on=True)) is wav  # 全効果が既定値
    assert voice_fx.apply(wav, VoiceFx(on=False, formant=1.2)) is wav


def test_unsupported_format_kept():
    wav = make_wav(np.zeros(100), ch=2)
    assert voice_fx.apply(wav, VoiceFx(on=True, ring_hz=50)) is wav
    assert voice_fx.apply(b"not a wav", VoiceFx(on=True, ring_hz=50)) == b"not a wav"


def test_length_rate_and_peak_preserved():
    x = voice_like()
    fx = VoiceFx(on=True, formant=1.2, ring_hz=60, ring_mix=0.6, comb_ms=3, comb_feedback=0.8, comb_mix=0.7)
    out = voice_fx.apply(make_wav(x), fx)
    with wave.open(io.BytesIO(out)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, RATE, x.size)
    y = read_wav(out)
    assert np.max(np.abs(y)) <= np.max(np.abs(x)) + 1e-3
    assert not np.allclose(y, x, atol=1e-2)


def test_formant_shift_moves_envelope_keeps_pitch():
    x = voice_like()
    up = voice_fx.formant_shift(x, 1.5)
    assert up.size == x.size
    # 包絡の山 800Hz が 1200Hz 付近へ (倍音 200Hz 刻みなので最大の倍音で見る)
    assert spectrum_peak(x) == pytest.approx(800, abs=30)
    assert spectrum_peak(up) == pytest.approx(1200, abs=210)
    # 声の高さ (倍音の間隔) は保たれる: 自己相関の周期が 1/f0
    def period(s):
        s = s[2000:-2000]
        ac = np.correlate(s, s, "full")[s.size - 1:]
        lo = RATE // 500
        return (np.argmax(ac[lo:RATE // 70]) + lo) / RATE
    assert period(up) == pytest.approx(1 / 200, rel=0.05)


def test_formant_identity_close_to_input():
    x = voice_like()
    assert np.allclose(voice_fx.formant_shift(x, 1.0), x, atol=1e-6)


def test_ring_mod_full_mix_is_product():
    x = voice_like(0.1)
    y = voice_fx.ring_mod(x, RATE, 50, 1.0)
    t = np.arange(x.size) / RATE
    assert np.allclose(y, x * np.sin(2 * np.pi * 50 * t))
    assert np.allclose(voice_fx.ring_mod(x, RATE, 50, 0.0), x)


def test_comb_matches_reference_recursion():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(1000) * 0.1
    d, g = int(round(RATE * 0.002)), 0.7
    ref = x.copy()
    for n in range(d, x.size):
        ref[n] = x[n] + g * ref[n - d]
    y = voice_fx.comb(x, RATE, 2.0, g, 1.0)
    assert np.allclose(y, ref * (1 - g))


def test_flanger_zero_depth_is_fixed_delay():
    x = np.zeros(200)
    x[0] = 1.0
    y = voice_fx.flanger(x, RATE, 1.0, 0.5, 0.0, 1.0)  # 遅れ 24 サンプル固定
    assert y[0] == pytest.approx(0.5) and y[24] == pytest.approx(0.5)
    assert np.count_nonzero(np.abs(y) > 1e-9) == 2


def test_processing_is_fast_enough():
    x = voice_like(5.0)
    fx = VoiceFx(on=True, formant=1.2, ring_hz=60, comb_ms=1, comb_feedback=0.5, comb_mix=0.5)
    start = time.perf_counter()
    voice_fx.apply(make_wav(x), fx)
    assert time.perf_counter() - start < 0.5  # 5 秒の声で 0.5 秒未満 (TTS 1 文の予算 6 秒に対して)


def test_load_voice_fx(tmp_path):
    f = tmp_path / "s.toml"
    assert load_voice_fx(f) == VoiceFx()
    f.write_text("[voice_fx]\non = true\nformant = 1.2\nring_hz = 60\n", encoding="utf-8")
    fx = load_voice_fx(f)
    assert fx.on and fx.formant == 1.2 and fx.ring_hz == 60.0 and fx.comb_ms == 0.0
    for bad in ("formant = 3.0", "comb_feedback = 1.0", "on = 1", "ring_hz = \"a\"", "unknown = 1"):
        f.write_text(f"[voice_fx]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_voice_fx(f)


def test_reloader_applies_fx_and_reloads(tmp_path):
    from idle_talk.clients import TtsClient
    from idle_talk.main import TtsSettingsReloader
    wav = make_wav(voice_like(0.3))
    f = tmp_path / "s.toml"
    f.write_text("[tts]\nspeaker = 1\n", encoding="utf-8")
    tts = TtsClient(speaker=1, post=lambda url, body, timeout: b"{}" if "audio_query" in url else wav)
    r = TtsSettingsReloader(tts, f)
    assert r.synthesize("あ") == wav
    f.write_text("[tts]\nspeaker = 1\n[voice_fx]\non = true\nring_hz = 80\nring_mix = 1.0\n", encoding="utf-8")
    os.utime(f, (1, 1))
    assert r.synthesize("あ") != wav and r.fx.ring_hz == 80.0
    f.write_text("[tts]\nspeaker = 1\n[voice_fx]\non = true\nring_hz = 9999\n", encoding="utf-8")  # 範囲外
    os.utime(f, (2, 2))
    r.synthesize("あ")
    assert r.fx.ring_hz == 80.0  # 前の値を保つ
