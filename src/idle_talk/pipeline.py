"""応答パイプライン (設計書 3.3 / 4.3)。LLM 節分割 -> TTS 直列 -> 再生。cancel は LLM/ASR 完了を待たない。"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from collections.abc import Callable, Iterable
from typing import Any

from .clause import ClauseSplitter, is_synthesizable
from .clients import ClientError
from .epoch import Epoch
from .errors import ERR_LLM_TIMEOUT, ERR_TTS_FAILED
from .llm_supervisor import FALLBACK_PHRASE
from .web_search import (LINK_NOTICE, SEARCH_INCOMPLETE_NOTICE, SEARCH_MAX_ROUNDS, SEARCH_NOTICE,
                         SEARCH_WAIT_NOTICE, SEARCH_WAIT_S, find_mark, link_items, parse_query,
                         research_messages, with_link_rule, with_search_rule)

log = logging.getLogger("idle_talk.pipeline")
_END = object()


class ResponsePipeline:
    """synthesize(text)->wav / llm_stream(messages)->tokens / cancel_tts() は同期ブロッキング関数。
    emit(msg) は WS 向けメッセージ、on_audio(epoch, utt_id, chunk_id, text, wav) は音声チャンク出力。"""

    def __init__(self, llm_stream: Callable[[list[dict]], Iterable[str]],
                 synthesize: Callable[[str], bytes], cancel_tts: Callable[[], None],
                 on_audio: Callable[[int, int, int, str, bytes], None],
                 emit: Callable[[dict], None], epoch: Epoch | None = None) -> None:
        self._llm = llm_stream
        self._tts = synthesize
        self._cancel_tts = cancel_tts
        self._on_audio = on_audio
        self._emit = emit
        self.epoch = epoch or Epoch()
        self._tts_lock = asyncio.Lock()
        self._utt = 0
        self.discarded = 0
        self.links: list[dict] = []  # 直近の検索結果の [{title, url}]。「URL を教えて」で画面に出す

    # ---- cancel (不変条件 1: LLM/ASR の完了を待たない) ----
    async def cancel(self) -> int:
        """TTS 中止を 1 回送り、epoch を +1 して cancel を通知する。進行中の LLM は await しない。"""
        self._cancel_tts()
        new = self.epoch.bump()
        self._emit({"t": "cancel", "new_epoch": new})
        return new

    # ---- 再生の唯一の入口 (不変条件 3) ----
    async def play_utterance(self, clauses: Iterable[str] | Any) -> int:
        """通常応答・自発発話・相槌クリップはすべてこの関数を通る。clauses は同期イテラブル (ブロッキング可)。"""
        ep = self.epoch.current
        self._utt += 1
        utt, chunk = self._utt, 0
        it = iter(clauses)
        while True:
            text = await asyncio.to_thread(next, it, _END)
            if text is _END:
                return utt
            if not self.epoch.is_current(ep):
                self._discard(ep, text)
                return utt
            async with self._tts_lock:  # TTS は直列 (E8)
                try:
                    wav = await asyncio.to_thread(self._tts, text)
                except ClientError as e:
                    if self.epoch.is_current(ep):
                        self._emit({"t": "error", "code": ERR_TTS_FAILED, "message": str(e)[:200]})
                        self._emit({"t": "text", "epoch": ep, "utterance_id": utt, "text": text})
                    continue
            if not self.epoch.is_current(ep):  # 遅れて返った結果は破棄
                self._discard(ep, text)
                return utt
            self._on_audio(ep, utt, chunk, text, wav)
            chunk += 1

    def _discard(self, ep: int, text: str) -> None:
        self.discarded += 1
        log.info("discard stale epoch=%d current=%d text=%r", ep, self.epoch.current, text)

    def _llm_clauses(self, messages: list[dict], search: Callable[[str], list[dict]] | None = None,
                     emit: Callable[[dict], None] | None = None):
        """LLM 出力を節にする (別スレッドで回る)。emit はイベントループへ渡すメッセージ送出。"""
        emit = emit or self._emit
        sp = ClauseSplitter()
        ep = self.epoch.current
        try:
            links = list(self.links) if search is not None else []
            first = messages
            if search is not None:
                first = with_search_rule(with_link_rule(messages) if links else messages)
            toks = iter(self._llm(first))
            if search is None:
                for tok in toks:
                    yield from sp.feed(tok)
                yield from sp.flush()
                return
            # 「検索:」「リンク:」は返答の先頭とは限らない (末尾に書くこともある)。節ごとに探し、前の文だけ読み上げる
            for tok in toks:
                for clause in sp.feed(tok):
                    kind, after = yield from self._before_mark(clause, links)
                    if kind is not None:
                        getattr(toks, "close", lambda: None)()
                        yield from self._mark_clauses(kind, after, messages, links, search, ep, emit)
                        return
            for clause in sp.flush():
                kind, after = yield from self._before_mark(clause, links)
                if kind is not None:
                    yield from self._mark_clauses(kind, after, messages, links, search, ep, emit)
        except (ClientError, OSError, ValueError, TimeoutError) as e:  # LlmTimeout 等/接続断。5001 + 定型句にフォールバック
            log.warning("llm failed: %s", e)
            emit({"t": "error", "code": ERR_LLM_TIMEOUT, "message": str(e)[:200]})
            yield FALLBACK_PHRASE

    @staticmethod
    def _before_mark(clause: str, links: list[dict]):
        """節に指示があれば前の文を読み上げ (yield)、(種類, 指示以降) を返す。無ければ節を読み上げて (None, "")。"""
        before, kind, after = find_mark(clause, links=bool(links))
        if is_synthesizable(before):
            yield before
        return kind, after

    def _mark_clauses(self, kind: str, after: str, messages: list[dict], links: list[dict],
                      search: Callable[[str], list[dict]], ep: int, emit: Callable[[dict], None]):
        if kind == "links":  # 話したニュース・調べたページの URL は読み上げず画面の URL 欄に出す
            log.info("show links: %d", len(links))
            emit({"t": "links", "items": links})
            yield LINK_NOTICE
            return
        yield from self._search_clauses(messages, parse_query(after), search, ep, emit)

    def _search_clauses(self, messages: list[dict], query: str, search: Callable[[str], list[dict]], ep: int,
                        emit: Callable[[dict], None]):
        """結果が不足すれば skill に従って再検索。回数と重複を制限し、取消後は進めない。"""
        query = query or next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")[:100]
        searches: list[tuple[str, list[dict]]] = []
        seen: set[str] = set()
        links: dict[str, dict] = {}
        wait_notified = False
        while self.epoch.is_current(ep):
            key = " ".join(query.split()).casefold()
            if not key or key in seen or len(searches) >= SEARCH_MAX_ROUNDS:
                yield SEARCH_INCOMPLETE_NOTICE
                return
            seen.add(key)
            log.info("web search round=%d: %r", len(searches) + 1, query)
            emit({"t": "search", "query": query[:200], "active": True})
            try:
                if not searches:
                    yield SEARCH_NOTICE
                if not self.epoch.is_current(ep):
                    return
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="search") as ex:
                    fut = ex.submit(search, query)
                    try:
                        results = fut.result(timeout=SEARCH_WAIT_S)
                    except FutureTimeout:
                        if self.epoch.is_current(ep) and not wait_notified:
                            wait_notified = True
                            yield SEARCH_WAIT_NOTICE
                        results = fut.result()
            finally:
                emit({"t": "search", "query": query[:200], "active": False})
            searches.append((query, results))
            for item in link_items(results):
                links[item["url"]] = item
            if links:  # 取消されても調べた結果は残す
                self.links = list(links.values())
            if not self.epoch.is_current(ep):
                return
            next_query = yield from self._research_answer(messages, searches, ep)
            if next_query is None:
                return
            query = next_query

    def _research_answer(self, messages: list[dict], searches: list[tuple[str, list[dict]]], ep: int):
        """追加検索のマーカーは読み上げず呼出元へ返す。通常の回答は逐次読み上げる。"""
        sp = ClauseSplitter()
        toks = iter(self._llm(research_messages(messages, searches, SEARCH_MAX_ROUNDS - len(searches))))
        try:
            for tok in toks:
                if not self.epoch.is_current(ep):
                    return None
                for clause in sp.feed(tok):
                    kind, after = yield from self._before_mark(clause, [])
                    if kind == "search":
                        return parse_query(after)
            for clause in sp.flush():
                if not self.epoch.is_current(ep):
                    return None
                kind, after = yield from self._before_mark(clause, [])
                if kind == "search":
                    return parse_query(after)
        finally:
            getattr(toks, "close", lambda: None)()
        return None

    async def respond(self, messages: list[dict], search: Callable[[str], list[dict]] | None = None,
                      tail: str = "") -> int:
        """search を渡すと、LLM が「検索: 語」と答えたときに Web 検索してから答える。tail は返答の後に必ず読む定型文。"""
        loop = asyncio.get_running_loop()
        emit = lambda m: loop.call_soon_threadsafe(self._emit, m)  # noqa: E731  節の生成は別スレッドで回る
        clauses = self._llm_clauses(messages, search, emit)
        if tail:
            clauses = _then(clauses, tail)
        return await self.play_utterance(clauses)


def _then(clauses, tail: str):
    yield from clauses
    yield tail
