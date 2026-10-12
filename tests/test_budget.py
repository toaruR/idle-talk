import pytest

from idle_talk import budget as b
from idle_talk.config import Config

C = Config()


def test_g1_p50():
    assert b.g1_p50_pass(1200) is True
    assert b.g1_p50_pass(1201) is False
    assert b.g1_p50_margin() == 50 == 1200 - (C.T_silence + C.VAR_SUM)


def test_g1_p95():
    assert b.g1_k_max() == pytest.approx(2.1333, abs=1e-3)
    assert b.g1_p95_pass(1.7) is True
    assert b.g1_p95_pass(2.14) is False


def test_g2():
    assert b.g2_pass(250) is True
    assert b.g2_pass(251) is False
    assert sum((40, 60, 30)) == 130 == C.G2_SUM == b.g2_sum()
    assert C.G2_SUM <= 250


@pytest.mark.parametrize("tc,exp", [(300, 430), (400, 530), (500, 630)])
def test_g2_confirm_cancel(tc, exp):
    assert b.confirm_cancel_ms(tc) == exp


def test_g3():
    assert b.g3_pass(0) is True
    assert b.g3_pass(1) is False


def test_g4():
    assert b.g4_pass(2) is True
    assert b.g4_pass(3) is False


def test_g5():
    assert b.g5_pass(8, 8) is True
    assert b.g5_pass(9, 8) is False


def test_strong_utt():
    assert b.strong_utt(0.1, 0.2) is True
    assert b.strong_utt(0.1, 0.19) is False


def test_n_turns():
    assert b.n_turns_tokens(8) == 2500
    assert b.n_turns_fits(b.n_turns_tokens(8)) is True
    assert b.n_turns_tokens(17) == 4750
    assert b.n_turns_fits(b.n_turns_tokens(17)) is False


def test_tts_ratio():
    assert b.tts_ratio_pass(3.0, 2.0) is True
    assert b.tts_ratio_pass(3.0, 2.1) is False


def test_tts_gpu_thresh():
    assert b.tts_gpu_compare(300) is False
    assert b.tts_gpu_compare(301) is True
