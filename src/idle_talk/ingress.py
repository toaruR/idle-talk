"""音声入力経路 (設計書 2.1 / 4.2 / 12.3 E4,E6)。

VAD -> fsm イベント (1 対 1) / 区間化 / ASR へ共有メモリで PCM 受け渡し / 無音フレーム連続検知。
"""
from __future__ import annotations

import logging
import time
import uuid
from collections import Counter, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import shared_memory

from .clients import AsrClient
from .errors import ERR_MIC_LOST
from .fsm import Director, Ev, Event
from .vad import BYTES_PER_MS, SPEECH_END, SPEECH_START, Vad, rms

log = logging.getLogger(__name__)
_FAILED = object()  # ASR 失敗 (発話破棄)

# offload(work, done): work (ブロッキングの ASR 呼び出し) を別スレッドで走らせ、結果を done で呼び出し元のループへ返す
Offload = Callable[[Callable[[], object], Callable[[object], None]], None]


class Ingress:
    def __init__(self, director: Director, vad: Vad, asr: AsrClient | None = None,
                 on_transcript: Callable[[int, str], bool] | None = None,
                 on_error: Callable[[int, str], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 silent_rms: float = 1.0,
                 on_speech_end: Callable[[], None] | None = None,
                 offload: Offload | None = None,
                 embed: Callable[[bytes], object] | None = None,
                 on_voice: Callable[[int, object], None] | None = None,
                 voice_min_ms: int = 0) -> None:
        self.director = director
        self.cfg = director.cfg
        self.vad = vad
        self.asr = asr
        self.on_transcript = on_transcript  # False を返すと認識結果を捨てる (雑音・自声)
        self.on_error = on_error
        self.on_speech_end = on_speech_end  # MIN_SEG 以上の発話終了通知 (M1 計測 eot_wait)
        self.offload = offload  # None なら ASR を呼び出し元で同期実行する
        self.asr_pending = 0  # 認識待ちの区間数
        # 話者識別 (設計書 6.4): voice_min_ms 以上の区間の声の特徴を ASR と並行して計算し、認識結果の直前に on_voice へ渡す
        self.embed = embed
        self.on_voice = on_voice
        self.voice_min_ms = voice_min_ms
        self._embed_pool: ThreadPoolExecutor | None = None
        self.clock = clock
        self.silent_rms = silent_rms
        self.frames_by_state: Counter[str] = Counter()
        self.noise_log: list[int] = []
        self.device_lost = False
        # E6 を無音で判定するか。Python で録るマイク (PC と話すモード) は静かな部屋で RMS 0.5 程度になり、
        # 抜けたと誤判定したので、読み取りの失敗で判定する (Director が切り替える)
        self.silence_check = True
        self._buf: bytearray | None = None
        # VAD が発話開始を判定するまでの onset_frames - 1 フレーム (発話の頭) を区間に含めるための直前フレーム
        self._pre: deque[bytes] = deque(maxlen=getattr(vad, "onset_frames", 1) - 1)
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
        if self.silence_check:
            self._check_silence(frame)
        if self._buf is not None:
            self._buf += frame
        ev = self.vad.process(frame)
        if ev == SPEECH_START:
            self._buf = bytearray(b"".join(self._pre) + frame)
            self._pre.clear()
            self.director.enqueue(Event(Ev.VAD_START))
        elif ev == SPEECH_END:
            pcm = bytes(self._buf[: len(self._buf) - len(frame)]) if self._buf is not None else b""
            self._buf = None
            self._end_segment(pcm)
        elif self._buf is None:
            self._pre.append(frame)
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
        uid = self._uid
        name = self._put_shm(pcm)
        self.asr_pending += 1
        voice = self._start_embed(pcm) if seg_ms >= self.voice_min_ms else None

        def work() -> tuple[object, object]:
            try:
                res = self.asr.transcribe(uid, name, seg_ms / 1000)
            except Exception as e:  # noqa: BLE001  認識の失敗で会話を止めない (該当発話だけ捨てる)
                log.warning("asr failed: utterance %d discarded: %s", uid, e)
                res = _FAILED
            finally:
                self.release(name)
            return res, _embed_result(voice, uid)

        if self.offload is None:
            self._asr_done(uid, *work())
        else:
            self.offload(work, lambda r: self._asr_done(uid, *r))

    def _start_embed(self, pcm: bytes) -> Future | None:
        if self.embed is None:
            return None
        if self._embed_pool is None:
            self._embed_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice")
        return self._embed_pool.submit(self.embed, pcm)

    def _asr_done(self, uid: int, res: object, voice: object = None) -> None:
        self.asr_pending -= 1
        if voice is not None and self.on_voice and isinstance(res, dict):
            self.on_voice(uid, voice)
        latest = uid == self._uid  # 後続区間の認識待ちがあれば、終話・破棄はそちらの結果で確定させる
        if res is None:  # 幻聴 (E4): 応答トリガを送らない
            log.info("asr hallucination discarded: utterance %d", uid)
        if res is _FAILED or res is None or (self.on_transcript and self.on_transcript(uid, res["text"]) is False):
            if latest and self._buf is None:  # 発話は終わっている: 待機 (AI 再生中なら再生再開) へ戻す
                self.director.enqueue(Event(Ev.DISCARDED))
            return
        if latest:
            self.director.enqueue(Event(Ev.SILENCE, silence_ms=self.cfg.T_silence, eot=True))

    # ---- E6: 無音フレーム連続 ----
    def lost(self) -> None:
        """マイクが読めなくなった (E6)。休止して 5005 を出す。"""
        if self.device_lost:
            return
        self.device_lost = True
        self.director.enqueue(Event(Ev.FAILURES))
        if self.on_error:
            self.on_error(ERR_MIC_LOST, "microphone device lost")

    def _check_silence(self, frame: bytes) -> None:
        now = self.clock()
        if rms(frame) > self.silent_rms:
            self._silent_since = None
            self.device_lost = False
            return
        if self._silent_since is None:
            self._silent_since = now
        if now - self._silent_since >= self.cfg.E_PARAMS.device_lost_s:
            self.lost()


def _embed_result(fut: Future | None, uid: int) -> object:
    """声の特徴。計算していない・失敗したら None (話者識別だけ諦め、認識結果は使う)。"""
    if fut is None:
        return None
    try:
        return fut.result()
    except Exception as e:  # noqa: BLE001
        log.warning("voice embedding failed: utterance %d: %s", uid, e)
        return None
