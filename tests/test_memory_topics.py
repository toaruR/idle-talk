import sqlite3

import pytest

from idle_talk import memory, topics
from idle_talk.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def test_expires_excluded(store):
    topics.add_card(store, "old", expires_at=100)
    assert topics.select_cards(store, now_ms=200, limit=5) == []


def test_used_excluded_and_mark_used(store):
    cid = topics.add_card(store, "h", expires_at=10_000)
    assert len(topics.select_cards(store, 200)) == 1
    topics.mark_used(store, cid, 300)
    assert store.query("SELECT used_at FROM topic_cards WHERE id=?", (cid,))[0][0] == 300
    assert topics.select_cards(store, 400) == []


def test_user_reaction_saved(store):
    cid = topics.add_card(store, "h")
    topics.set_reaction(store, cid, "liked")
    assert store.query("SELECT user_reaction FROM topic_cards WHERE id=?", (cid,))[0][0] == "liked"


def test_hook_null_rejected(store):
    with pytest.raises(ValueError):
        topics.add_card(store, None)
    with pytest.raises(sqlite3.IntegrityError):
        store.execute("INSERT INTO topic_cards(hook) VALUES (NULL)")


def test_evidence_required(store):
    with pytest.raises(ValueError):
        memory.add_memory(store, "x", None, 1)
    assert store.count("memories") == 0
    memory.add_memory(store, "x", 5, 1)
    assert store.count("memories") == 1


def test_delete_deletable_only(store):
    a = memory.add_memory(store, "a", 1, 1, deletable=1)
    b = memory.add_memory(store, "b", 1, 1, deletable=0)
    memory.delete_memory(store, a)
    with pytest.raises(PermissionError):
        memory.delete_memory(store, b)
    assert store.count("memories") == 1


def test_delete_all_memories(store):
    memory.add_memory(store, "a", 1, 1, deletable=1)
    memory.add_memory(store, "b", 1, 1, deletable=0)
    assert memory.delete_all_memories(store) == 2
    assert store.count("memories") == 0


def test_ratio_q_simulation():
    r = topics.QuestionRatio()
    qs = [r.decide(True) for _ in range(30)]
    assert sum(qs) <= 10
    for i in range(len(qs) - 2):
        assert sum(qs[i:i + 3]) <= 1
