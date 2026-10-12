"""応答パイプライン (設計書 3.3 / 4.3)。LLM 節分割 -> TTS 直列 -> 再生。cancel は LLM/ASR 完了を待たない。"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from typing import Any

from .clause import ClauseSplitter
from .clients import ClientError
from .epoch import Epoch
from .errors import ERR_LLM_TIMEOUT, ERR_TTS_FAILED
from .llm_supervisor import FALLBACK_PHRASE

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

    def _llm_clauses(self, messages: list[dict]):
        sp = ClauseSplitter()
        try:
            for tok in self._llm(messages):
                yield from sp.feed(tok)
            yield from sp.flush()
        except (ClientError, OSError, ValueError, TimeoutError) as e:  # LlmTimeout 等/接続断。5001 + 定型句にフォールバック
            log.warning("llm failed: %s", e)
            self._emit({"t": "error", "code": ERR_LLM_TIMEOUT, "message": str(e)[:200]})
            yield FALLBACK_PHRASE

    async def respond(self, messages: list[dict]) -> int:
        return await self.play_utterance(self._llm_clauses(messages))
