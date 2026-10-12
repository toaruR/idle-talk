import pytest

from idle_talk import eot
from idle_talk.m0_rules import eot_choice


def test_fixed_boundary():
    f = eot.FixedSilence(400)
    assert not f.is_end(399)
    assert f.is_end(400)
    assert f.is_end(401)


def test_fixed_default_from_config():
    assert eot.FixedSilence().t_silence == 400


def test_smart_turn_fake_model():
    st = eot.SmartTurn(lambda a: 0.9 if a == b"done" else 0.1)
    assert st.is_end(0, b"done")
    assert not st.is_end(1000, b"more")
    assert not st.is_end(1000)


def test_g4_boundary():
    assert eot.g4_pass(0)
    assert eot.g4_pass(2)
    assert not eot.g4_pass(3)


@pytest.mark.parametrize("f,s", [(5, 2), (2, 5), (3, 3), (0, 0), (2, 1)])
def test_compare_same_as_eot_choice(f, s):
    r = eot.compare_g4(f, s)
    assert r["fixed"] == f and r["smart_turn"] == s
    assert r["recommend"] == eot_choice(f, s)


def test_compare_recommend():
    assert eot.compare_g4(5, 2)["recommend"] == "SmartTurn"
    assert eot.compare_g4(2, 5)["recommend"] == "FixedSilence"
    assert eot.compare_g4(3, 3)["recommend"] == "FixedSilence"
    assert eot.compare_g4(2, 3)["fixed_pass"] and not eot.compare_g4(2, 3)["smart_turn_pass"]
