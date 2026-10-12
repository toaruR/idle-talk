import io
import wave
from pathlib import Path

import pytest

from idle_talk import m0_bench
from idle_talk.config import Config
from idle_talk.m0_bench import REQUIRED_KEYS, evaluate_m0, run_m0

GOOD = {
    "llm_first_s": 1.0, "llm_total_s": 4.0, "llm_think_count": 0,
    "tts_first_ms": 200.0, "tts_ratio_p95": 0.5, "asr_silence_outputs": 0,
    "llm_vram_gib": 6.0, "asr_vram_gib": 1.5, "app_ram_gib": 2.0,
}


def _wav(sec: float) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(1000)
        w.writeframes(b"\x00\x00" * int(1000 * sec))
    return b.getvalue()


def test_evaluate_good_is_empty():
    assert evaluate_m0(GOOD) == []


@pytest.mark.parametrize("key,bad", [
    ("llm_think_count", 1), ("llm_first_s", 5.1), ("llm_total_s", 20.1),
    ("tts_ratio_p95", 0.7), ("asr_silence_outputs", 1),
    ("llm_vram_gib", 9.1), ("asr_vram_gib", 3.1), ("app_ram_gib", 8.1),
])
def test_evaluate_flags_each_violation(key, bad):
    v = evaluate_m0({**GOOD, key: bad})
    assert len(v) == 1 and key in v[0]


def test_evaluate_boundaries_pass():
    m = {**GOOD, "llm_first_s": 5, "llm_total_s": 20, "tts_ratio_p95": 0.69,
         "llm_vram_gib": 9.0, "asr_vram_gib": 3.0, "app_ram_gib": 8.0}
    assert evaluate_m0(m) == []


def test_evaluate_keys_missing_reported():
    for k in REQUIRED_KEYS:
        m = dict(GOOD)
        del m[k]
        assert evaluate_m0(m) == [f"missing:{k}"]
    assert len(evaluate_m0({})) == len(REQUIRED_KEYS)


def test_keys_returned_by_run_m0():
    class Llm:
        def stream_chat(self, msgs):
            yield "こん"
            yield "にちは"

    class Tts:
        def synthesize(self, text):
            return _wav(3.0)

    ticks = iter(x * 0.1 for x in range(1000))
    m = run_m0(llm=Llm(), tts=Tts(), transcribe=lambda pcm: [], vram_probe=lambda: 5.0,
               ram_probe=lambda: 1.0, clock=lambda: next(ticks))
    assert set(REQUIRED_KEYS) <= set(m)
    assert m["llm_text"] == "こんにちは" and m["llm_think_count"] == 0
    assert m["asr_silence_outputs"] == 0 and m["asr_vram_gib"] == 0.0
    assert evaluate_m0(m) == []


def test_run_m0_counts_think_and_hallucination():
    class Llm:
        def stream_chat(self, msgs):
            yield "<think>x"

    class Tts:
        def synthesize(self, text):
            return _wav(1.0)

    m = run_m0(llm=Llm(), tts=Tts(), transcribe=lambda pcm: [{"text": "ご視聴ありがとう"}],
               vram_probe=lambda: 1.0, ram_probe=lambda: 1.0)
    v = evaluate_m0(m)
    assert any("llm_think_count" in x for x in v) and any("asr_silence_outputs" in x for x in v)


def test_defaults_limits_match_design():
    cfg = Config()
    assert (m0_bench.LLM_VRAM_MAX_GIB, m0_bench.ASR_VRAM_MAX_GIB, m0_bench.APP_RAM_MAX_GIB) == (9.0, 3.0, 8.0)
    assert cfg.TTS_RATIO_MAX == 0.7
    assert (cfg.HTTP_TIMEOUT.llm_first, cfg.HTTP_TIMEOUT.llm_total) == (5, 20)


def test_p95_and_wav_duration():
    assert m0_bench.p95([1.0] * 19 + [9.0]) == 9.0
    assert m0_bench.p95([0.5]) == 0.5
    assert m0_bench.wav_duration_s(_wav(2.0)) == pytest.approx(2.0)


def test_live_policy_m0_live_has_no_skip_tokens():
    from test_live_policy import find_violations

    src = (Path(__file__).parent / "test_m0_live.py").read_text(encoding="utf-8")
    assert find_violations(src) == []
    assert "pytest.mark.integration" in src
