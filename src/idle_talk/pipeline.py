"""応答パイプライン (設計書 3.3 / 4.3)。LLM 節分割 -> TTS 直列 -> 再生。cancel は LLM/ASR 完了を待たない。"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from collections.abc import Callable, Iterable
from typing import Any

from .clause import ClauseSplitter
from .clients import ClientError
from .epoch import Epoch
from .errors import ERR_LLM_TIMEOUT, ERR_TTS_FAILED
from .llm_supervisor import FALLBACK_PHRASE
from .web_search import (SEARCH_NOTICE, SEARCH_WAIT_NOTICE, SEARCH_WAIT_S, answer_messages, mark_state,
                         parse_query, with_search_rule)

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
            toks = iter(self._llm(with_search_rule(messages) if search else messages))
            if search is not None:
                head, verdict = "", None
                for tok in toks:  # 先頭が「検索:」か判断できるまで節分割に回さない
                    head += tok
                    verdict = mark_state(head)
                    if verdict is not None:
                        break
                if verdict == "yes":
                    while "\n" not in head.lstrip():  # 検索語の行を最後まで受け取る (以降の出力は使わない)
                        tok = next(toks, None)
                        if tok is None:
                            break
                        head += tok
                    getattr(toks, "close", lambda: None)()
                    yield from self._search_clauses(messages, parse_query(head), search, ep, emit)
                    return
                yield from sp.feed(head)
            for tok in toks:
                yield from sp.feed(tok)
            yield from sp.flush()
        except (ClientError, OSError, ValueError, TimeoutError) as e:  # LlmTimeout 等/接続断。5001 + 定型句にフォールバック
            log.warning("llm failed: %s", e)
            emit({"t": "error", "code": ERR_LLM_TIMEOUT, "message": str(e)[:200]})
            yield FALLBACK_PHRASE

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
