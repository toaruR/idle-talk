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
from .web_search import (LINK_NOTICE, SEARCH_NOTICE, SEARCH_WAIT_NOTICE, SEARCH_WAIT_S, answer_messages, find_mark,
                         link_items, parse_query, with_link_rule, with_search_rule)

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
        """検索中であることを声で伝えてから検索し、結果を添えて答え直させる。取消後は 2 回目の LLM を呼ばない。"""
        query = query or next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")[:100]
        log.info("web search: %r", query)
        emit({"t": "search", "query": query[:200], "active": True})
        yield SEARCH_NOTICE
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="search") as ex:
            fut = ex.submit(search, query)
            try:
                results = fut.result(timeout=SEARCH_WAIT_S)
            except FutureTimeout:
                if self.epoch.is_current(ep):
                    yield SEARCH_WAIT_NOTICE
                results = fut.result()
        emit({"t": "search", "query": query[:200], "active": False})
        if results:  # 取消されても調べた結果は残す (後で「URL を教えて」と言われたら出す)
            self.links = link_items(results)
        if not self.epoch.is_current(ep):
            return
        sp = ClauseSplitter()
        for tok in self._llm(answer_messages(messages, query, results)):
            yield from sp.feed(tok)
        yield from sp.flush()

    async def respond(self, messages: list[dict], search: Callable[[str], list[dict]] | None = None) -> int:
        """search を渡すと、LLM が「検索: 語」と答えたときに Web 検索してから答える。"""
        loop = asyncio.get_running_loop()
        emit = lambda m: loop.call_soon_threadsafe(self._emit, m)  # noqa: E731  節の生成は別スレッドで回る
        return await self.play_utterance(self._llm_clauses(messages, search, emit))
