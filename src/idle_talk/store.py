"""SQLite ストア (設計書 3.4 / 7 E13,E14)。WAL、書込は専用スレッドのキュー経由。"""
from __future__ import annotations

import logging
import os
import queue
import shutil
import sqlite3
import threading
from collections.abc import Iterable
from concurrent.futures import Future
from pathlib import Path
from typing import Any

from .errors import ERR_DB

SCHEMA_VERSION = 3  # 2: speakers (話者識別、設計書 6.4)、3: memories.persona (ペルソナごとの記憶、設計書 6.6)

log = logging.getLogger("idle_talk.store")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    ts INTEGER,
    epoch INTEGER,
    kind TEXT,
    payload_json TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    ts INTEGER,
    role TEXT,
    text TEXT,
    completed INTEGER CHECK (completed IN (0, 1)),
    stop_position_ms INTEGER NULL,
    utterance_id INTEGER
);
CREATE TABLE IF NOT EXISTS topic_cards (
    id INTEGER PRIMARY KEY,
    hook TEXT NOT NULL,
    facts TEXT,
    source_url TEXT,
    published_at INTEGER,
    expires_at INTEGER,
    angle TEXT,
    used_at INTEGER NULL,
    user_reaction TEXT NULL
);
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    content TEXT,
    evidence_utterance_id INTEGER,
    updated_at INTEGER,
    deletable INTEGER,
    persona TEXT NULL
);
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY,
    name TEXT NULL,
    embedding BLOB NOT NULL,
    n INTEGER,
    updated_at INTEGER
);
"""

TABLES = ("events", "turns", "topic_cards", "memories", "speakers")


def _integrity_ok(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        con = sqlite3.connect(str(path))
        try:
            rows = con.execute("PRAGMA integrity_check").fetchall()
        finally:
            con.close()
        return len(rows) == 1 and rows[0][0] == "ok"
    except sqlite3.DatabaseError:
        return False


class Store:
    def __init__(self, path: str | os.PathLike, db_backup: bool = True) -> None:
        self.path = Path(path)
        self.bak_path = self.path.with_name(self.path.name + ".bak")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.restored = False
        self.write_thread_ids: list[int] = []
        self._prepare_file(db_backup)

        self._q: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(target=self._writer_loop, name="store-writer", daemon=True)
        self._ready: Future = Future()
        self._thread.start()
        self._ready.result()
        self._read = sqlite3.connect(str(self.path), check_same_thread=False)
        self._read.row_factory = sqlite3.Row
        self._read_lock = threading.Lock()

    # --- 起動時: integrity_check と .bak (E13/E14) ---
    def _prepare_file(self, db_backup: bool) -> None:
        if _integrity_ok(self.path):
            if db_backup and self.path.exists():
                self._make_backup()
            return
        log.error("error %d: DB integrity_check failed: %s", ERR_DB, self.path)
        if db_backup and self.bak_path.exists():
            for suffix in ("-wal", "-shm"):
                p = self.path.with_name(self.path.name + suffix)
                if p.exists():
                    p.unlink()
            shutil.copyfile(self.bak_path, self.path)
            self.restored = True
            log.error("error %d: DB restored from %s", ERR_DB, self.bak_path)
        else:
            # 復元元なし: 破損ファイルを退避して新規作成
            self.path.replace(self.path.with_name(self.path.name + ".corrupt"))

    def _make_backup(self) -> None:
        tmp = self.bak_path.with_name(self.bak_path.name + ".tmp")
        src = sqlite3.connect(str(self.path))
        try:
            dst = sqlite3.connect(str(tmp))
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        os.replace(tmp, self.bak_path)  # 常に 1 世代のみ

    # --- 書込スレッド ---
    def _writer_loop(self) -> None:
        try:
            con = sqlite3.connect(str(self.path), check_same_thread=True, isolation_level=None)
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA synchronous=NORMAL")
            con.executescript(SCHEMA_SQL)
            if "persona" not in {r[1] for r in con.execute("PRAGMA table_info(memories)")}:
                con.execute("ALTER TABLE memories ADD COLUMN persona TEXT NULL")  # v2 以前: 既存の記憶は全ペルソナ共通
            con.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._ready.set_result(True)
        except BaseException as e:  # noqa: BLE001
            self._ready.set_exception(e)
            return
        while True:
            item = self._q.get()
            if item is None:
                break
            fn, fut = item
            self.write_thread_ids.append(threading.get_ident())
            try:
                fut.set_result(fn(con))
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            con.close()

    def _submit(self, fn) -> Any:
        if self._closed:
            raise RuntimeError("store closed")
        fut: Future = Future()
        self._q.put((fn, fut))
        return fut.result()

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        """書込 SQL を書込スレッドで実行し lastrowid を返す。"""
        params = tuple(params)

        def _do(con: sqlite3.Connection) -> int:
            cur = con.execute(sql, params)
            return cur.lastrowid

        return self._submit(_do)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._read_lock:
            return self._read.execute(sql, tuple(params)).fetchall()

    # --- 簡易 API ---
    def add_event(self, ts: int, epoch: int, kind: str, payload_json: str = "{}") -> int:
        return self.execute(
            "INSERT INTO events(ts, epoch, kind, payload_json) VALUES (?,?,?,?)",
            (ts, epoch, kind, payload_json),
        )

    def add_turn(self, ts: int, role: str, text: str, completed: int,
                 stop_position_ms: int | None = None, utterance_id: int | None = None) -> int:
        return self.execute(
            "INSERT INTO turns(ts, role, text, completed, stop_position_ms, utterance_id) VALUES (?,?,?,?,?,?)",
            (ts, role, text, completed, stop_position_ms, utterance_id),
        )

    def count(self, table: str) -> int:
        if table not in TABLES:
            raise ValueError(table)
        return self.query(f"SELECT COUNT(*) FROM {table}")[0][0]

    def delete_all(self) -> None:
        """§7 全削除: 全テーブルの全行を削除する。"""
        def _do(con: sqlite3.Connection) -> None:
            con.execute("BEGIN")
            try:
                for t in TABLES:
                    con.execute(f"DELETE FROM {t}")
                con.execute("COMMIT")
            except BaseException:
                con.execute("ROLLBACK")
                raise

        self._submit(_do)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._q.put(None)
        self._thread.join()
        self._read.close()
