"""RSS Topic Worker (T021 / 設計書 2.1, 7, 10)。ネットワークは使わない。"""
import socket
import threading
from pathlib import Path

import pytest

from idle_talk.store import Store
from idle_talk.topic_worker import TopicWorker, parse_feed

FEED = (Path(__file__).parent / "fixtures" / "sample_feed.xml").read_bytes()


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", db_backup=False)
    yield s
    s.close()


def test_parse_feed_three_items():
    items = parse_feed(FEED)
    assert len(items) == 3
    assert all(isinstance(i["published_at"], int) for i in items)
    assert parse_feed(b"<not xml") == []


def test_parse_and_store_three_cards(store):
    w = TopicWorker(store, ["http://feed.test/rss"], fetch=lambda u: FEED)
    w.start()
    assert w.poll_once() == 3
    rows = store.query("SELECT * FROM topic_cards")
    assert len(rows) == 3
    for r in rows:
        assert r["hook"] and r["source_url"]
        assert isinstance(r["published_at"], int)
        assert r["expires_at"] > r["published_at"]
    assert w.poll_once() == 0  # 重複なし
    assert store.count("topic_cards") == 3


def test_store_write_via_writer_thread(store):
    names = []
    orig = store.execute

    def spy(sql, params=()):
        names.append(threading.current_thread().name)
        return orig(sql, params)

    store.execute = spy
    w = TopicWorker(store, ["u"], fetch=lambda u: FEED)
    w.start()
    w.poll_once()
    assert len(names) == 3  # 書込は Store.execute (書込専用スレッド) 経由のみ


def test_stopped_no_network(store, monkeypatch):
    calls = []
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: calls.append(a))
    w = TopicWorker(store, ["http://feed.test/rss"])  # 既定の http_fetch
    assert w.poll_once() == 0  # 未開始
    w.start()
    w.on_state("rest")  # rest で停止
    assert w.running is False
    assert w.poll_once() == 0
    assert calls == []
    assert store.count("topic_cards") == 0


def test_interrupt_on_user_speaking(store):
    started = threading.Event()

    def slow(item, cancel):
        started.set()
        cancel.wait(5)
        return item["title"]

    w = TopicWorker(store, ["u"], fetch=lambda u: FEED, generate=slow)
    w.start()
    w.on_state("idle")
    t = threading.Thread(target=w.poll_once)
    t.start()
    assert started.wait(2)
    w.on_state("user_speaking")
    t.join(5)
    assert not t.is_alive()
    assert w.interrupted == 1
    assert store.count("topic_cards") == 0
