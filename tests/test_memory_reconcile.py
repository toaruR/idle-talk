"""会話での訂正・撤回、共有知識の分離、抽出中の手動操作との競合。"""
import json

import numpy as np
import pytest

from idle_talk import memory
from idle_talk.store import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "memory.db", db_backup=False)
    yield s
    s.close()


def speaker(store, name):
    return store.execute("INSERT INTO speakers(name, embedding, n, updated_at) VALUES (?, ?, 1, 0)",
                         (name, np.array([1, 0], dtype=np.float32).tobytes()))


def operation(mid, content, evidence, kind="speaker"):
    return json.dumps(dict(kind=kind, id=mid, content=content, evidence=evidence), ensure_ascii=False)


def test_correction_replaces_in_place_with_actual_evidence_and_persists(store):
    sid = speaker(store, "太郎")
    mid = memory.add_memory(store, "利用者は横浜に住んでいる。", 1, 100, persona="a", speaker_id=sid)
    snapshot = memory.memory_snapshot(store, "a", sid)
    text = operation(mid, "利用者は大阪に住んでいる。", "横浜じゃなく大阪に住んでいる")
    batch = [(9, "横浜じゃなく大阪に住んでいるよ"), (10, "今日は暖かいね")]
    memory.remember_extracted(store, text, batch, snapshot, 200, "a", sid)
    row, = memory.list_memories(store)
    assert (row["id"], row["content"], row["updated_at"], row["evidence_utterance_id"]) == (
        mid, "利用者は大阪に住んでいる。", 200, 9)
    again = Store(store.path, db_backup=False)
    try:
        assert "大阪" in memory.memory_prompt(again, "a", sid)
        assert "横浜" not in memory.memory_prompt(again, "a", sid)
    finally:
        again.close()


def test_retraction_removes_only_target(store):
    mid = memory.add_memory(store, "利用者は来週旅行する。", 1, 1)
    memory.add_memory(store, "利用者は猫が好き。", 1, 1)
    utterance = "来週旅行するって話は間違いなので取り消して"
    memory.remember_extracted(store, operation(mid, None, utterance), [(2, utterance)],
                               memory.memory_snapshot(store), 2)
    assert [r["content"] for r in memory.list_memories(store)] == ["利用者は猫が好き。"]


def test_snapshot_and_updates_isolate_speakers_personas_and_fixed_rows(store):
    a, b = speaker(store, "太郎"), speaker(store, "花子")
    own = memory.add_memory(store, "利用者は横浜に住む。", 1, 1, persona="a", speaker_id=a)
    shared_personal = memory.add_memory(store, "利用者は猫が好き。", 1, 1, speaker_id=a)
    protected = [memory.add_memory(store, "利用者は横浜に住む。", 1, 1, persona="a", speaker_id=b),
                 memory.add_memory(store, "利用者は横浜に住む。", 1, 1, persona="b", speaker_id=a),
                 memory.add_memory(store, "利用者は固定。", 1, 1, deletable=0, persona="a", speaker_id=a),
                 memory.add_memory(store, "利用者は話者未割当。", 1, 1, persona="a")]
    snapshot = memory.memory_snapshot(store, "a", a)
    assert {r["id"] for r in snapshot} == {own, shared_personal}
    utterance = "横浜ではなく大阪に住んでいます"
    for mid in protected:
        memory.remember_extracted(store, operation(mid, "利用者は大阪に住む。", utterance), [(2, utterance)], snapshot, 2, "a", a)
    assert all(r["updated_at"] == 1 for r in memory.list_memories(store))
    memory.remember_extracted(store, operation(own, "利用者は大阪に住む。", utterance), [(2, utterance)], snapshot, 2, "a", a)
    assert "大阪" in memory.memory_prompt(store, "a", a)
    assert "大阪" not in memory.memory_prompt(store, "a", b)
    assert "大阪" not in memory.memory_prompt(store, "b", a)


@pytest.mark.parametrize("mutation", ["delete", "update", "fixed", "reuse_id"])
def test_stale_extraction_does_not_override_manual_changes_or_resurrect(store, mutation):
    old, new = "利用者は横浜に住む。", "利用者は大阪に住む。"
    mid = memory.add_memory(store, old, 1, 1)
    snapshot = memory.memory_snapshot(store)
    if mutation in ("delete", "reuse_id"):
        memory.delete_memory(store, mid)
        if mutation == "reuse_id":
            assert memory.add_memory(store, "利用者は猫が好き。", 3, 3) == mid
    elif mutation == "update":
        store.execute("UPDATE memories SET content=?, updated_at=3 WHERE id=?", ("利用者は京都に住む。", mid))
    else:
        store.execute("UPDATE memories SET deletable=0 WHERE id=?", (mid,))
    before = [tuple(row) for row in memory.list_memories(store)]
    utterance = "横浜じゃなく大阪です"
    # モデルが置換と同時に新旧の内容を新規保存しようとしても復活させない。
    text = "\n".join([operation(mid, new, utterance), operation(None, new, utterance), operation(None, old, utterance)])
    memory.remember_extracted(store, text, [(2, utterance)], snapshot, 2)
    assert [tuple(row) for row in memory.list_memories(store)] == before


@pytest.mark.parametrize("bad", [
    {"id": True}, {"id": "1"}, {"kind": "invalid"}, {"content": 5}, {"content": "利用者は" + "長" * 60},
    {"content": "利用者は猫好き。\n利用者は犬好き。"}, {"content": "本人ではない"}, {"evidence": ""},
    {"evidence": "利用者は横浜に住んでいる。"}, {"evidence": "引用されていない根拠"},
])
def test_malformed_or_ungrounded_changes_leave_memory_intact(store, bad):
    mid = memory.add_memory(store, "利用者は横浜に住んでいる。", 1, 1)
    op = dict(kind="speaker", id=mid, content="利用者は大阪に住む。", evidence="大阪に住んでいる")
    op.update(bad)
    memory.remember_extracted(store, json.dumps(op), [(2, "大阪に住んでいるよ")], memory.memory_snapshot(store), 2)
    assert tuple(memory.list_memories(store)[0]) == (mid, "利用者は横浜に住んでいる。", 1, 1, 1, None, None)


def test_duplicate_changes_to_same_target_are_ignored(store):
    mid = memory.add_memory(store, "利用者は横浜に住む。", 1, 1)
    utterance = "横浜ではなく大阪に住んでいる"
    text = "\n".join([operation(mid, "利用者は大阪に住む。", utterance), operation(mid, None, utterance)])
    memory.remember_extracted(store, text, [(2, utterance)], memory.memory_snapshot(store), 2)
    assert memory.list_memories(store)[0]["content"] == "利用者は横浜に住む。"


def test_shared_knowledge_is_separate_and_available_to_all_and_survives_speaker_deletion(store):
    a, b = speaker(store, "太郎"), speaker(store, "花子")
    memory.remember(store, ["利用者は猫が好き。"], 1, 1, "a", a)
    utterance = "この作品の舞台は横浜です"
    text = operation(None, "この作品の舞台は横浜。", utterance, "shared")
    memory.remember_extracted(store, text, [(2, utterance)], memory.memory_snapshot(store, "a", a), 2, "a", a)
    assert store.count("knowledge") == 1 and store.count("memories") == 1
    for persona, sid in (("a", a), ("b", a), ("a", b), ("b", None)):
        prompt = memory.memory_prompt(store, persona, sid, include_speaker=sid is not None)
        assert "この作品の舞台は横浜。" in prompt and "共有知識" in prompt
        if persona != "a" or sid != a:
            assert "猫が好き" not in prompt
    store.execute("DELETE FROM speakers WHERE id=?", (a,))
    assert store.count("memories") == 0 and store.count("knowledge") == 1


def test_shared_correction_uses_separate_id_namespace_and_evidence(store):
    personal = memory.add_memory(store, "利用者は横浜に住む。", 1, 1, persona="a")
    shared, = memory.remember_knowledge(store, ["この作品の舞台は横浜。"], 1, 1)
    assert personal == shared  # 同じ番号でも kind が異なる
    utterance = "作品の舞台は横浜ではなく大阪です"
    memory.remember_extracted(store, operation(shared, "この作品の舞台は大阪。", utterance, "shared"),
                               [(2, utterance)], memory.memory_snapshot(store, "b"), 2, "b")
    assert "横浜" in memory.list_memories(store)[0]["content"]
    assert "大阪" in memory.memory_prompt(store, "c", include_speaker=False)
    assert store.query("SELECT evidence_utterance_id FROM knowledge")[0][0] == 2


def test_shared_knowledge_deduplication_and_independent_capacity(store):
    memory.remember(store, ["利用者は猫が好き。"], 1, 1)
    memory.remember_knowledge(store, ["作品の舞台は横浜。"], 1, 1)
    assert memory.remember_knowledge(store, ["作品の舞台は 横浜"], 2, 2) == []
    assert store.count("knowledge") == 1
    assert store.query("SELECT updated_at, evidence_utterance_id FROM knowledge")[0][:] == (2, 2)
    for i in range(memory.MAX_KNOWLEDGE):
        memory.remember_knowledge(store, [f"作品{i}の舞台は大阪。"], 3, i + 3)
    assert store.count("knowledge") == memory.MAX_KNOWLEDGE
    assert "横浜" not in memory.memory_prompt(store)
    assert "猫が好き" in memory.memory_prompt(store)


def test_short_correction_allowed_but_short_new_facts_rejected(store):
    mid = memory.add_memory(store, "利用者は猫が好き。", 1, 1)
    utterance = "猫は嫌い"
    text = "\n".join([operation(mid, "利用者は猫が嫌い。", utterance), operation(None, "利用者は猫が嫌い。", utterance)])
    memory.remember_extracted(store, text, [(2, utterance)], memory.memory_snapshot(store), 2)
    assert [r["content"] for r in memory.list_memories(store)] == ["利用者は猫が嫌い。"]


def test_structured_personal_and_shared_additions_have_correct_source_and_scope(store):
    sid = speaker(store, "太郎")
    text = "\n".join([operation(None, "利用者は猫が好き。", "猫が好きです"),
                       operation(None, "この作品の舞台は大阪。", "この作品の舞台は大阪", "shared")])
    memory.remember_extracted(store, text, [(4, "猫が好きです"), (5, "この作品の舞台は大阪です")], [], 100, "a", sid)
    personal, = memory.list_memories(store)
    assert (personal["persona"], personal["speaker_id"], personal["evidence_utterance_id"]) == ("a", sid, 4)
    assert store.query("SELECT evidence_utterance_id FROM knowledge")[0][0] == 5


def test_personal_subject_is_not_accepted_as_shared_knowledge(store):
    utterance = "私は猫がとても好きです"
    memory.remember_extracted(store, operation(None, "利用者は猫が好き。", utterance, "shared"), [(1, utterance)], [], 1)
    assert store.count("knowledge") == 0
