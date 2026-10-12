import inspect

import pytest

from idle_talk import m0_rules as r


def test_llm_choose_both():
    assert r.llm_choose(True, 30, True) == "9B"
    assert r.llm_choose(True, 50, True) == "9B"


def test_llm_choose_one_or_none():
    assert r.llm_choose(True, 40, False) == "4B"
    assert r.llm_choose(False, 40, True) == "4B"
    assert r.llm_choose(False, 40, False) == "4B"


@pytest.mark.parametrize("n", [29, 51])
def test_llm_choose_scene_range(n):
    with pytest.raises(ValueError):
        r.llm_choose(True, n, True)


def test_asr_place():
    assert r.asr_place(300) == "GPU0"
    assert r.asr_place(301) == "GPU1"


def test_tts_place():
    assert r.tts_place(300, 0.1)["place"] == "CPU"
    assert r.tts_place(301, 0.1)["place"] == "GPU1_DirectML_compare"


def test_tts_ratio():
    assert r.tts_ratio_ok(0.69) is True
    assert r.tts_ratio_ok(0.70) is False
    assert r.tts_place(100, 0.69)["ratio_ok"] is True
    assert r.tts_place(100, 0.70)["ratio_ok"] is False


def test_no_think():
    assert r.no_think_check("<think>x</think>hi") is False
    assert r.no_think_check("こんにちは") is True


def test_resource_limits():
    g = 1024
    assert r.resource_check(9 * g, 3 * g, 8 * g) == {"llm_vram": True, "asr_vram": True, "ram": True}
    assert r.resource_check(9 * g + 1, 0, 0)["llm_vram"] is False
    assert r.resource_check(0, 3 * g + 1, 0)["asr_vram"] is False
    assert r.resource_check(0, 0, 8 * g + 1)["ram"] is False


def test_eot_choice():
    assert r.eot_choice(5, 2) == "SmartTurn"
    assert r.eot_choice(2, 5) == "FixedSilence"
    assert r.eot_choice(3, 3) == "FixedSilence"


def test_keys():
    assert r.MEASUREMENT_KEYS == {
        "llm_first_clause_ms", "tts_first_sentence_ms", "tts_ratio_p95",
        "vram_mib", "asr_vram_mib", "ram_mib", "overlap_p95"}
    names = set(inspect.signature(r.resource_check).parameters) | set(
        inspect.signature(r.asr_place).parameters) | set(
        inspect.signature(r.tts_place).parameters)
    assert {"vram_mib", "asr_vram_mib", "ram_mib", "overlap_p95",
            "tts_first_sentence_ms", "tts_ratio_p95"} <= names
