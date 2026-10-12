"""記憶 (設計書 3.4 / §7)。根拠発話なしの保存は拒否、deletable=1 のみ個別削除可。"""
from __future__ import annotations

from .store import Store


class MemoryError_(Exception):
    pass


def add_memory(store: Store, content: str, evidence_utterance_id: int | None,
               updated_at: int, deletable: int = 1) -> int:
    if evidence_utterance_id is None:
        raise ValueError("evidence_utterance_id is required")
    return store.execute(
        "INSERT INTO memories(content, evidence_utterance_id, updated_at, deletable) VALUES (?,?,?,?)",
        (content, evidence_utterance_id, updated_at, 1 if deletable else 0),
    )


def list_memories(store: Store) -> list:
    return store.query("SELECT * FROM memories ORDER BY id")


def delete_memory(store: Store, memory_id: int) -> None:
    """deletable=1 のみ削除。deletable=0 や不在は例外。"""
    rows = store.query("SELECT deletable FROM memories WHERE id=?", (memory_id,))
    if not rows:
        raise KeyError(memory_id)
    if not rows[0][0]:
        raise PermissionError(f"memory {memory_id} is not deletable")
    store.execute("DELETE FROM memories WHERE id=? AND deletable=1", (memory_id,))


def delete_all_memories(store: Store) -> int:
    """§7 利用者操作の全記憶削除 API。削除件数を返す。"""
    n = store.count("memories")
    store.execute("DELETE FROM memories")
    return n
