"""M0 実機試験 (-m integration)。接続不能は失敗として扱う。"""
import pytest

from idle_talk.config import Config
from idle_talk.m0_bench import evaluate_m0, run_m0

pytestmark = pytest.mark.integration

CFG = Config()


@pytest.fixture(scope="module")
def measured():
    return run_m0(cfg=CFG)


def test_m0_evaluate_has_no_violations(measured):
    assert evaluate_m0(measured, CFG) == []


def test_m0_real_output_has_no_think_tags(measured):
    assert measured["llm_text"]
    assert "<think>" not in measured["llm_text"]
    assert measured["llm_think_count"] == 0
