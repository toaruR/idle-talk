import logging

import pytest

from idle_talk.config import Config
from idle_talk.history import PlaybackHistory, history_window
from idle_talk.interrupt import (
    classify,
    in_confirm_window,
    initial_action,
    is_strong_utt,
    judge,
)
from idle_talk.store import Store

CFG = Config()


@pytest.mark.parametrize("w", ["待って", "違う", "ちょっと", "ストップ"])
def test_stop_words_cancel(w):
    assert initial_action(w) == "cancel"


def test_other_word_duck():
    assert initial_action("今日は") == "duck"


@pytest.mark.parametrize("w", ["うん", "へえ"])
def test_backchannel_resume(w):
    assert classify(w) == "backchannel"
    assert judge(w, 0, 100) == "backchannel"


@pytest.mark.parametrize("w", ["うん？", "いや"])
def test_not_backchannel(w):
    assert classify(w) != "backchannel"
    assert judge(w, 0, 100) != "backchannel"


def test_unknown_cancels():
    assert judge("むにゃ", 0, 100) == "unknown"
    assert judge("", 0, 100) == "unknown"


def test_strong_utt():
    assert is_strong_utt(0.2, 0.1, False)
    assert not is_strong_utt(0.19, 0.1, False)
    assert is_strong_utt(0.01, 0.1, True)


@pytest.mark.parametrize("tc", [300, 400, 500])
def test_confirm_window_half_open(tc):
    c = Config(T_confirm=tc)
    assert in_confirm_window(1000, 1000, c)
    assert in_confirm_window(1000, 1000 + tc - 1, c)
    assert not in_confirm_window(1000, 1000 + tc, c)
    assert not in_confirm_window(1000, 999, c)


def test_backchannel_after_window_not_resume():
    assert judge("うん", 0, 399) == "backchannel"
    assert judge("うん", 0, 400) == "unknown"


def _pb(c, state, pos=0, e=1, u=1):
    return {"t": "playback", "epoch": e, "utterance_id": u, "chunk_id": c, "position_ms": pos, "state": state}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", db_backup=False)
    yield s
    s.close()


def test_history_only_finished(store):
    h = PlaybackHistory(store)
    for i, t in enumerate(["A。", "B。", "C。"]):
        h.register_chunk(1, 1, i, t)
    h.on_playback(_pb(0, "finished", 500))
    h.on_playback(_pb(1, "started", 100))
    h.on_playback(_pb(2, "dropped", 0))
    h.commit(1, 1, 10)
    rows = store.query("SELECT text, completed FROM turns")
    assert [(r[0], r[1]) for r in rows] == [("A。", 1)]


def test_history_sent_not_played_zero(store):
    h = PlaybackHistory(store)
    h.register_chunk(1, 1, 0, "X")
    h.on_playback(_pb(0, "started"))
    h.commit(1, 1, 10)
    assert store.query("SELECT COUNT(*) FROM turns WHERE completed=1")[0][0] == 0


def test_history_interrupted_stop_position(store):
    h = PlaybackHistory(store)
    h.register_chunk(1, 1, 0, "A。")
    h.register_chunk(1, 1, 1, "BBB。")
    h.on_playback(_pb(0, "finished", 500))
    h.on_playback(_pb(1, "started", 230))
    h.commit(1, 1, 10, interrupted=True)
    r = store.query("SELECT text, completed, stop_position_ms FROM turns")[0]
    assert (r[0], r[1], r[2]) == ("A。", 0, 230)
    assert "BBB" not in r[0]


def test_history_window_limit_and_log(store, caplog):
    for i in range(1000):
        store.add_turn(i, "user", f"t{i}", 1)
    with caplog.at_level(logging.INFO, logger="idle_talk.history"):
        w = history_window(store)
    assert len(w) <= CFG.N_TURNS == 8
    assert w[-1]["text"] == "t999"
    assert len([r for r in caplog.records if r.name == "idle_talk.history"]) == 1
