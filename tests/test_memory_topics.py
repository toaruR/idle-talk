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


def test_positive_card_selected_before_newer_negative(store):
    topics.add_card(store, "高速道路で事故 3人死亡", published_at=300, expires_at=10_000)
    topics.add_card(store, "円相場の動き", published_at=200, expires_at=10_000)
    topics.add_card(store, "新しい水族館が開業 イルカの赤ちゃん誕生", published_at=100, expires_at=10_000)
    hooks = [r["hook"] for r in topics.select_cards(store, 0, limit=3)]
    assert hooks == ["新しい水族館が開業 イルカの赤ちゃん誕生", "円相場の動き", "高速道路で事故 3人死亡"]


def test_same_tone_keeps_newest_first(store):
    topics.add_card(store, "古い話", published_at=100)
    topics.add_card(store, "新しい話", published_at=200)
    assert topics.select_cards(store, 0)[0]["hook"] == "新しい話"


def test_card_tone_uses_facts():
    assert topics.card_tone("駅前の話", "地震で停電") < 0 < topics.card_tone("駅前の話", "桜が満開")


def test_parse_pick():
    assert topics.parse_pick("3", 4) == 2
    assert topics.parse_pick("番号は２です", 4) == 1
    assert topics.parse_pick("5", 4) is None
    assert topics.parse_pick("0", 4) is None
    assert topics.parse_pick("なし", 4) is None
    assert topics.parse_pick(None, 4) is None


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


def test_ends_with_question():
    assert topics.ends_with_question("リキはそれを温かいまま飲むと？")
    assert topics.ends_with_question("何かに集中しとるん?」")
    assert topics.ends_with_question("趣味を詳しく聞かせてほしいっちゃんね")
    assert topics.ends_with_question("また教えてね。")
    assert not topics.ends_with_question("平穏に一日が終わるのが一番の贅沢かもしれんとね。")
    assert not topics.ends_with_question("")


def test_ratio_q_from_texts():
    assert topics.QuestionRatio.from_texts([]).can_ask()
    assert not topics.QuestionRatio.from_texts(["そうなん？"]).can_ask()
    assert not topics.QuestionRatio.from_texts(["そうなん？", "いいね。"]).can_ask()  # 3 応答に 1 回以下
    assert topics.QuestionRatio.from_texts(["そうなん？", "いいね。", "よかね。"]).can_ask()
