"""音声入力経路 (設計書 2.1 / 4.2 / 12.3 E4,E6)。

VAD -> fsm イベント (1 対 1) / 区間化 / ASR へ共有メモリで PCM 受け渡し / 無音フレーム連続検知。
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from multiprocessing import shared_memory

from .clients import AsrClient, AsrTimeout
from .errors import ERR_MIC_LOST
from .fsm import Director, Ev, Event
from .vad import BYTES_PER_MS, SPEECH_END, SPEECH_START, Vad, rms

log = logging.getLogger(__name__)


class Ingress:
    def __init__(self, director: Director, vad: Vad, asr: AsrClient | None = None,
                 on_transcript: Callable[[int, str], None] | None = None,
                 on_error: Callable[[int, str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 silent_rms: float = 1.0,
                 on_speech_end: Callable[[], None] | None = None) -> None:
        self.director = director
        self.cfg = director.cfg
        self.vad = vad
        self.asr = asr
        self.on_transcript = on_transcript
        self.on_error = on_error
        self.on_speech_end = on_speech_end  # MIN_SEG 以上の発話終了通知 (M1 計測 eot_wait)
        self.clock = clock
        self.silent_rms = silent_rms
        self.frames_by_state: Counter[str] = Counter()
        self.noise_log: list[int] = []
        self.device_lost = False
        self._buf: bytearray | None = None
        self._uid = 0
        self._silent_since: float | None = None
        self._shm: dict[str, shared_memory.SharedMemory] = {}

    @property
    def shm_count(self) -> int:
        return len(self._shm)

    # ---- 共有メモリ ----
    def _put_shm(self, pcm: bytes) -> str:
        name = f"idle_talk_{uuid.uuid4().hex[:16]}"
        shm = shared_memory.SharedMemory(name=name, create=True, size=max(len(pcm), 1))
        shm.buf[: len(pcm)] = pcm
        self._shm[name] = shm
        return name

    def release(self, name: str) -> None:
        shm = self._shm.pop(name, None)
        if shm is None:
            return
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass

    # ---- フレーム処理 ----
    def feed(self, frame: bytes) -> None:
        """全状態で同一に VAD を回す (不変条件 2)。fsm 状態は計数のみに使う。"""
        self.frames_by_state[self.director.ctx.state.value] += 1
        self._check_silence(frame)
        if self._buf is not None:
            self._buf += frame
        ev = self.vad.process(frame)
        if ev == SPEECH_START:
            self._buf = bytearray(frame)
            self.director.enqueue(Event(Ev.VAD_START))
        elif ev == SPEECH_END:
            pcm = bytes(self._buf[: len(self._buf) - len(frame)]) if self._buf is not None else b""
            self._buf = None
            self._end_segment(pcm)
        self.director.pump()

    def _end_segment(self, pcm: bytes) -> None:
        seg_ms = len(pcm) // BYTES_PER_MS
        self.director.enqueue(Event(Ev.VAD_END, seg_ms=seg_ms))
        if seg_ms < self.cfg.MIN_SEG:
            self.noise_log.append(seg_ms)
            log.info("noise segment discarded: %dms < MIN_SEG=%d", seg_ms, self.cfg.MIN_SEG)
            return
        if self.on_speech_end:
            self.on_speech_end()
        self._asr(pcm, seg_ms)

    def _asr(self, pcm: bytes, seg_ms: int) -> None:
        if self.asr is None:
            return
        self._uid += 1
        name = self._put_shm(pcm)
        try:
            res = self.asr.transcribe(self._uid, name, seg_ms / 1000)
        except AsrTimeout:
            log.warning("asr timeout: utterance %d discarded", self._uid)
            return
        finally:
            self.release(name)
        if res is None:  # 幻聴 (E4): 応答トリガを送らない
            log.info("asr hallucination discarded: utterance %d", self._uid)
            return
        if self.on_transcript:
            self.on_transcript(self._uid, res["text"])
        self.director.enqueue(Event(Ev.SILENCE, silence_ms=self.cfg.T_silence, eot=True))

    # ---- E6: 無音フレーム連続 ----
    def _check_silence(self, frame: bytes) -> None:
        now = self.clock()
        if rms(frame) > self.silent_rms:
            self._silent_since = None
            self.device_lost = False
            return
        if self._silent_since is None:
            self._silent_since = now
        if not self.device_lost and now - self._silent_since >= self.cfg.E_PARAMS.device_lost_s:
            self.device_lost = True
            self.director.enqueue(Event(Ev.FAILURES))
            if self.on_error:
                self.on_error(ERR_MIC_LOST, "microphone device lost")
