import logging
import sqlite3
import threading

import pytest

from idle_talk.store import SCHEMA_VERSION, Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def _cols(s, t):
    return {r[1] for r in s.query(f"PRAGMA table_info({t})")}


def test_schema_columns(store):
    assert _cols(store, "events") == {"id", "ts", "epoch", "kind", "payload_json"}
    assert _cols(store, "turns") == {"id", "ts", "role", "text", "completed", "stop_position_ms", "utterance_id"}
    assert _cols(store, "topic_cards") == {
        "id", "hook", "facts", "source_url", "published_at", "expires_at", "angle", "used_at", "user_reaction"}
    assert _cols(store, "memories") == {"id", "content", "evidence_utterance_id", "updated_at", "deletable"}


def test_wal_and_user_version(store):
    assert store.query("PRAGMA journal_mode")[0][0].lower() == "wal"
    assert store.query("PRAGMA user_version")[0][0] == SCHEMA_VERSION


def test_writer_thread_differs(store):
    store.add_turn(1, "user", "hi", 1)
    store.add_event(1, 0, "k")
    assert store.write_thread_ids
    assert threading.get_ident() not in store.write_thread_ids
    assert len(set(store.write_thread_ids)) == 1


def test_schema_constraints(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.execute("INSERT INTO topic_cards(hook) VALUES (NULL)")
    store.add_turn(1, "ai", "x", 0, stop_position_ms=None)
    assert store.query("SELECT stop_position_ms FROM turns")[0][0] is None


def test_e14_backup_and_restore(tmp_path, caplog):
    p = tmp_path / "t.db"
    s = Store(p)
    for i in range(5):
        s.add_turn(i, "user", f"t{i}", 1)
    s.close()
    s = Store(p)  # 起動時に .bak 作成
    s.close()
    bak = tmp_path / "t.db.bak"
    assert bak.exists()
    s = Store(p)  # 再起動を繰り返しても世代は増えない
    s.close()
    baks = [f for f in tmp_path.iterdir() if ".bak" in f.name]
    assert baks == [bak]

    with open(p, "r+b") as f:
        f.write(b"\x00" * 100)
    with caplog.at_level(logging.ERROR, logger="idle_talk.store"):
        s = Store(p)
    try:
        assert s.restored
        assert s.count("turns") == 5
    finally:
        s.close()
    assert any("5004" in r.getMessage() for r in caplog.records)
    assert len([f for f in tmp_path.iterdir() if ".bak" in f.name]) == 1


def test_delete_all(store):
    store.add_event(1, 0, "k")
    store.add_turn(1, "user", "a", 1)
    store.execute("INSERT INTO topic_cards(hook) VALUES ('h')")
    store.execute("INSERT INTO memories(content, evidence_utterance_id, updated_at, deletable) VALUES ('c',1,1,1)")
    store.delete_all()
    for t in ("events", "turns", "topic_cards", "memories"):
        assert store.count(t) == 0
