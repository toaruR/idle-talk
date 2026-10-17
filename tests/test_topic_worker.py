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


def test_resumes_after_rest(store):
    w = TopicWorker(store, ["u"], fetch=lambda u: FEED)
    w.start()
    w.on_state("rest")
    assert w.poll_once() == 0
    w.on_state("idle")
    assert w.running is True
    assert w.poll_once() == 3


def test_short_summary_filled_from_page(store):
    page = '<html><head><meta property="og:description" content="記事ページの説明文です。&amp; 詳細はこちら。三十文字を超えるよう長めに書いておきます。"></head></html>'
    feed = ('<rss><channel><item><title>T</title><link>https://example.test/a</link><description>短い</description>'
            '</item><item><title>U</title><link>https://example.test/b</link><description>'
            + "十分に長い概要の文章です。" * 3 + '</description></item></channel></rss>').encode()
    fetched = []

    def fetch(u):
        fetched.append(u)
        return feed if u == "feed" else page.encode()

    w = TopicWorker(store, ["feed"], fetch=fetch)
    w.start()
    assert w.poll_once() == 2
    facts = {r["hook"]: r["facts"] for r in store.query("SELECT hook, facts FROM topic_cards")}
    assert facts["T"].startswith("記事ページの説明文です。& 詳細")
    assert facts["U"].startswith("十分に長い概要")
    assert fetched == ["feed", "https://example.test/a"]  # 概要が十分ならページは読まない


def test_page_fetch_failure_keeps_feed_summary(store):
    def fetch(u):
        if u != "feed":
            raise OSError("down")
        return FEED

    w = TopicWorker(store, ["feed"], fetch=fetch)
    w.start()
    assert w.poll_once() == 3
    assert min(r[0] for r in store.query("SELECT facts FROM topic_cards")) == "First summary text."


def test_page_description_shift_jis():
    from idle_talk.topic_worker import page_description
    html = '<html><head><meta http-equiv="Content-Type" content="text/html; charset=Shift_JIS">' \
           '<meta name="description" content="便利になるからこそ、人の力が問われますね。"></head></html>'
    assert page_description(html.encode("shift_jis")) == "便利になるからこそ、人の力が問われますね。"
    assert page_description(b"<html></html>") is None


def test_no_duplicate_cards_across_restart(store):
    pages = []

    def fetch(u):
        if u != "feed":
            pages.append(u)
        return FEED

    first =TopicWorker(store, ["feed"], fetch=fetch)
    first.start()
    assert first.poll_once() == 3
    pages.clear()
    again = TopicWorker(store, ["feed"], fetch=fetch)  # 再起動相当 (起動中の既読集合は空)
    again.start()
    assert again.poll_once() == 0
    assert store.count("topic_cards") == 3
    assert pages == []  # 保存済みの記事はページも読まない


# ---- 在庫切れの補充 (フィードの取り直し + ニュース検索) ----

class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


NEWS = [{"title": "新型ロケット打ち上げ成功", "snippet": "宇宙機関は 14 日、新型ロケットの打ち上げに成功したと発表した。",
         "url": "https://news.test/space/1"}]


def test_request_refill_throttled_and_ignored_when_stopped(store):
    from idle_talk.topic_worker import REFILL_MIN_S
    clock = Clock()
    w = TopicWorker(store, [], fetch=lambda u: FEED, clock=clock)
    assert not w.request_refill() and not w.wake.is_set()  # 停止中
    w.start()
    assert w.request_refill() and w.wake.is_set()
    w.wake.clear()
    clock.t = REFILL_MIN_S - 1
    assert not w.request_refill() and not w.wake.is_set()  # 取得先に負荷をかけない
    clock.t = REFILL_MIN_S
    assert w.request_refill()


def test_refill_uses_feed_first_then_news_search_rotating_queries(store):
    queries = []
    w = TopicWorker(store, ["http://feed.test/rss"], fetch=lambda u: FEED,
                    news=lambda q: queries.append(q) or NEWS, news_queries=("A", "B"))
    w.start()
    assert w.refill() == 3 and queries == []  # フィードに新着があればニュース検索はしない
    assert w.refill() == 1 and queries == ["A"]
    row = store.query("SELECT * FROM topic_cards WHERE source_url=?", ("https://news.test/space/1",))[0]
    assert row["hook"] == "新型ロケット打ち上げ成功" and "打ち上げに成功" in row["facts"]
    assert row["expires_at"] > row["published_at"]
    assert w.refill() == 0 and queries == ["A", "B"]  # 同じ記事は重複保存しない。ジャンルは順に回す


def test_poll_stores_x_trends_with_angle_and_short_ttl(store):
    from idle_talk.topic_worker import TREND_TTL_MS
    trends = [{"title": "裸の勇者", "snippet": "X で話題 (エンタメ)。投稿: 「公開されるの嬉しすぎます」",
               "url": "https://search.yahoo.co.jp/realtime/search?p=x", "angle": "x_trend"}]
    w = TopicWorker(store, ["http://feed.test/rss"], fetch=lambda u: FEED, now_ms=lambda: 1000, trends=lambda: trends)
    w.start()
    assert w.poll_once() == 4  # フィード 3 件 + トレンド 1 件
    row = store.query("SELECT * FROM topic_cards WHERE angle='x_trend'")[0]
    assert row["hook"] == "裸の勇者" and "嬉しすぎます" in row["facts"] and row["expires_at"] == 1000 + TREND_TTL_MS
    assert w.poll_once() == 0  # 同じトレンドは重複保存しない


def test_refill_without_news_is_poll_only(store):
    w = TopicWorker(store, ["http://feed.test/rss"], fetch=lambda u: FEED)
    w.start()
    assert w.refill() == 3 and w.refill() == 0


def test_topic_loop_wakes_on_refill_request(store):
    import asyncio

    from idle_talk.main import topic_loop
    calls = []
    w = TopicWorker(store, [], fetch=lambda u: FEED)
    w.poll_once = lambda: calls.append("poll") or 0
    w.refill = lambda: calls.append("refill") or 0

    async def run():
        t = asyncio.get_running_loop().create_task(topic_loop(w, 3600))
        for _ in range(20):
            await asyncio.sleep(0.01)
        assert calls == ["poll"]  # 起動直後の 1 回。次は 1 時間後
        w.start()
        assert w.request_refill()
        for _ in range(20):
            await asyncio.sleep(0.01)
        t.cancel()
    asyncio.run(run())
    assert calls == ["poll", "refill"]
