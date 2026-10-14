"""RSS Topic Worker (M4 / 設計書 2.1, 7, 10)。

ニュース取得はこのワーカーだけがネットワークを使う (§7)。停止中は一切接続しない。
カードは Store の書込専用スレッド経由 (topics.add_card) で保存する (§3.4)。
"""
from __future__ import annotations

import asyncio
import email.utils
import logging
import re
import threading
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from html.parser import HTMLParser

from .store import Store
from .topics import add_card, has_card

TOPIC_TTL_MS = 24 * 60 * 60 * 1000  # カード有効期間 (published_at から)
FETCH_TIMEOUT_S = 10.0
REFILL_MIN_S = 300  # 在庫切れによる取り直しの最短間隔 (秒)。取得先に負荷をかけない
SUMMARY_MIN_CHARS = 30  # フィードの概要 (タグ除去後) がこれより短ければ記事ページの説明文で補う
log = logging.getLogger("idle_talk.topic_worker")
_ATOM = "{http://www.w3.org/2005/Atom}"


def _text(el: ET.Element | None) -> str:
    return (el.text or "").strip() if el is not None else ""


def _parse_date_ms(s: str) -> int | None:
    if not s:
        return None
    try:
        return int(email.utils.parsedate_to_datetime(s).timestamp() * 1000)
    except (TypeError, ValueError):
        pass
    try:
        import datetime
        return int(datetime.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


def parse_feed(xml_bytes: bytes | str) -> list[dict]:
    """RSS 2.0 / Atom を [{title, summary, url, published_at(epoch ms)}] にする。壊れた入力は []。"""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return []
    out: list[dict] = []
    for it in root.iter("item"):
        out.append({"title": _text(it.find("title")), "summary": _text(it.find("description")),
                    "url": _text(it.find("link")) or None,
                    "published_at": _parse_date_ms(_text(it.find("pubDate")))})
    for en in root.iter(_ATOM + "entry"):
        link = en.find(_ATOM + "link")
        out.append({"title": _text(en.find(_ATOM + "title")),
                    "summary": _text(en.find(_ATOM + "summary")),
                    "url": link.get("href") if link is not None else None,
                    "published_at": _parse_date_ms(_text(en.find(_ATOM + "published"))
                                                   or _text(en.find(_ATOM + "updated")))})
    return [i for i in out if i["title"]]


class _MetaDescription(HTMLParser):
    """記事ページの og:description (無ければ description) を拾う。"""

    def __init__(self) -> None:
        super().__init__()
        self.found: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "meta":
            return
        a = dict(attrs)
        key = (a.get("property") or a.get("name") or "").lower()
        if key in ("og:description", "description") and a.get("content"):
            self.found.setdefault(key, a["content"].strip())


def _decode_html(html_bytes: bytes) -> str:
    """<meta charset> / http-equiv の charset で復号する (ITmedia の記事ページは Shift_JIS)。無指定・未知なら UTF-8。"""
    m = re.search(rb"charset=[\"']?([A-Za-z0-9_-]+)", html_bytes[:4096])
    try:
        return html_bytes.decode(m.group(1).decode() if m else "utf-8", "replace")
    except LookupError:
        return html_bytes.decode("utf-8", "replace")


def page_description(html_bytes: bytes) -> str | None:
    p = _MetaDescription()
    p.feed(_decode_html(html_bytes))
    return p.found.get("og:description") or p.found.get("description") or None


def http_fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT_S) as r:
        return r.read()


class TopicWorker:
    """start() 中のみ fetch する。rest で停止、user_speaking で進行中の背景生成を中断する。"""

    def __init__(self, store: Store, feed_urls: list[str],
                 fetch: Callable[[str], bytes] = http_fetch,
                 generate: Callable[[dict, threading.Event], str | None] | None = None,
                 now_ms: Callable[[], int] | None = None, ttl_ms: int = TOPIC_TTL_MS,
                 news: Callable[[str], list[dict]] | None = None, news_queries: tuple[str, ...] = (),
                 clock: Callable[[], float] | None = None) -> None:
        import time
        self.store = store
        self._news = news  # 在庫切れでフィードに新着も無いときのニュース検索 (query -> [{title, snippet, url}])
        self.news_queries = news_queries
        self._news_i = 0
        self._clock = clock or time.monotonic
        self._last_refill: float | None = None
        self.wake = asyncio.Event()  # request_refill で立つ。topic_loop が待ち合わせる
        self.feed_urls = list(feed_urls)
        self._fetch = fetch
        self._generate = generate or (lambda item, cancel: item["title"])
        self._now = now_ms or (lambda: int(time.time() * 1000))
        self.ttl_ms = ttl_ms
        self.running = False
        self.interrupted = 0
        self.saved = 0
        self._cancel = threading.Event()
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._state = "idle"

    # ---- 状態連動 ----
    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False
        self._cancel.set()

    def on_state(self, state: str) -> None:
        """Director の状態変化を受ける。rest→停止、rest から復帰→再開、(idle→)user_speaking→背景生成中断。"""
        prev, self._state = self._state, state
        if state == "rest":
            self.stop()
        elif prev == "rest":
            self.start()
        elif state == "user_speaking" and prev != "user_speaking":
            self._cancel.set()

    # ---- 取得 ----
    def poll_once(self) -> int:
        """全フィードを 1 回取得してカードを保存。保存件数を返す。停止中は何もしない。"""
        if not self.running:
            return 0
        self._cancel.clear()
        n = 0
        for url in self.feed_urls:
            if not self.running or self._cancel.is_set():
                break
            try:
                items = parse_feed(self._fetch(url))
            except (OSError, ValueError) as e:  # 取得失敗は次回に持ち越す (会話に影響させない)
                log.debug("feed fetch failed: %s", e)
                continue
            for item in items:
                if not self.running or self._cancel.is_set():
                    break
                n += self._store_item(item)
        return n

    # ---- 在庫切れの補充 ----
    def request_refill(self) -> bool:
        """未使用の話題カードが尽きたとき Director が呼ぶ。REFILL_MIN_S 以内の再要求と停止中は無視する。"""
        now = self._clock()
        if not self.running or (self._last_refill is not None and now - self._last_refill < REFILL_MIN_S):
            return False
        self._last_refill = now
        self.wake.set()
        return True

    def refill(self) -> int:
        """フィードを取り直し、新着が無ければニュース検索 (ジャンルを順に回す) の記事をカードにする。保存件数を返す。"""
        n = self.poll_once()
        if n or self._news is None or not self.news_queries or not self.running:
            return n
        query = self.news_queries[self._news_i % len(self.news_queries)]
        self._news_i += 1
        for r in self._news(query):
            if not self.running or self._cancel.is_set():
                break
            n += self._store_item({"title": r["title"], "summary": r.get("snippet") or "",
                                   "url": r.get("url"), "published_at": None})
        log.info("topic cards from news search %r: %d", query, n)
        return n

    def _store_item(self, item: dict) -> int:
        key = item.get("url") or item["title"]
        with self._lock:
            if key in self._seen:
                return 0
        if has_card(self.store, item.get("url"), item["title"]):  # 前回起動までに保存済み (ページは読まない)
            with self._lock:
                self._seen.add(key)
            return 0
        summary = item.get("summary") or ""
        if item.get("url") and len(re.sub(r"<[^>]+>", "", summary).strip()) < SUMMARY_MIN_CHARS:
            summary = self._describe(item["url"]) or summary
            if self._cancel.is_set():
                self.interrupted += 1
                return 0
        hook = self._generate(item, self._cancel)
        if self._cancel.is_set():
            self.interrupted += 1
            return 0
        if not hook:
            return 0
        published = item["published_at"] if item["published_at"] is not None else self._now()
        add_card(self.store, hook, facts=summary or None, source_url=item.get("url"),
                 published_at=published, expires_at=published + self.ttl_ms)
        with self._lock:
            self._seen.add(key)
            self.saved += 1
        return 1

    def _describe(self, url: str) -> str | None:
        """記事ページの説明文。取得・解析に失敗したら None (フィードの概要のまま保存する)。"""
        try:
            return page_description(self._fetch(url))
        except (OSError, ValueError) as e:
            log.debug("page fetch failed: %s", e)
            return None
