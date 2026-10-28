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
from .vad import BYTES_PER_MS, FRAME_MS, SPEECH_END, SPEECH_START, Vad, rms

log = logging.getLogger(__name__)
_FAILED = object()  # ASR 失敗 (発話破棄)
PRE_ROLL_MS = 300   # 開始判定より前の音も ASR に渡す長さ (閾値に届かない小声の出だし・子音)
# 語なしで捨てた区間 (物音・1 字だけ等) の終わりからこの間隔以内に始まった区間は、捨てた区間を頭に足して認識し直す。
# 発話が VAD で割れ、出だしの「え」だけの区間が捨てられて最初の何語かが欠けた (実機 2026-10-19)
JOIN_GAP_MS = 1000
SPLIT_LOOK_MS = 2000  # 区間を切るときは、切る長さに達した時点の直近この長さで一番静かなフレームの後で切る
SPLIT_NEAR_MS = 500  # 話者の替わり目で切るときは、その前後この長さで一番静かなフレームの後で切る
SPLIT_MIN_MS = 1000  # 話者の替わり目で切るとき、切った前の区間・残りがこれより短くなるなら切らない

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
        # 認識の失敗・幻聴で区間を捨てたときに呼ぶ (文字起こしモードは仮の文字で補う。Director が設定)。
        # 真を返したら補えたので、捨てた区間を次の区間の頭に足さない (同じ言葉が 2 回出る)
        self.on_discard: Callable[[int], bool] | None = None
        self.voice_min_ms = voice_min_ms
        self._embed_pool: ThreadPoolExecutor | None = None
        self.clock = clock
        self.silent_rms = silent_rms
        self.frames_by_state: Counter[str] = Counter()
        self.noise_log: list[int] = []
        self.device_lost = False
        # E6 を無音で判定するか。Python で録るマイク (ブラウザ補正なしモード) は静かな部屋で RMS 0.5 程度になり、
        # 抜けたと誤判定したので、読み取りの失敗で判定する (Director が切り替える)
        self.silence_check = True
        # 区間を切る長さ (ms、0 なら切らない)。文字起こしモードで動画の音を拾うと、テレビ・ドラマは音が途切れず発話が
        # 終わらなかった (実機 2026-10-27)。この長さに達したら直近 SPLIT_LOOK_MS で一番静かな所 (台詞の切れ目・息継ぎ)
        # で切って認識に回し、残りは次の区間の頭にする (決まった長さで切ると語の途中で切れた)。Director が切り替える
        self.max_seg_ms = 0
        self.voice_max_ms = 0  # 声の特徴を測る長さの上限 (ms、0 なら区間全体)。長い区間は複数人の声が混ざる
        self._buf: bytearray | None = None
        # VAD が発話開始を判定するまでの onset_frames - 1 フレーム (発話の頭) を区間に含めるための直前フレーム。
        # さらに前の PRE_ROLL_MS は区間の長さに数えず、ASR に渡す音にだけ足す (_lead)
        self._onset_head = getattr(vad, "onset_frames", 1) - 1
        self._pre: deque[bytes] = deque(maxlen=self._onset_head + PRE_ROLL_MS // FRAME_MS)
        self._lead = b""
        self._n = 0  # 受けたフレーム数 (区間の間隔を測る)
        self._start_n = 0
        self._segs: dict[int, tuple[bytes, int]] = {}  # 認識待ちの区間: uid -> (ASR に渡した音, 終わりのフレーム番号)
        self.seg_span: dict[int, tuple[int, int]] = {}  # 認識待ちの区間: uid -> (始まり, 終わり) のフレーム番号 (文字起こしの時刻)
        self._head: tuple[bytes, int] | None = None  # 語なしで捨てた区間 (次の区間の頭に足す候補)
        self._uid = 0
        self._discarded_until = 0
        self._silent_since: float | None = None
        self._shm: dict[str, shared_memory.SharedMemory] = {}

    @property
    def shm_count(self) -> int:
        return len(self._shm)

    def reset(self) -> None:
        """接続前の音と認識結果を捨てる。処理中の共有メモリはワーカーが解放する。"""
        self._discarded_until = self._uid
        self._buf = None
        self._pre.clear()
        self._lead = b""
        self._head = None
        self._segs.clear()
        self._silent_since = None
        self.device_lost = False
        reset = getattr(self.vad, "reset", None)
        if reset is not None:
            reset()

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
        self._n += 1
        if self.silence_check:
            self._check_silence(frame)
        if self._buf is not None:
            self._buf += frame
        ev = self.vad.process(frame)
        if ev == SPEECH_START:
            pre = list(self._pre)
            k = max(0, len(pre) - self._onset_head)
            self._lead = b"".join(pre[:k])
            self._buf = bytearray(b"".join(pre[k:]) + frame)
            self._start_n = self._n
            self._pre.clear()
            self.director.enqueue(Event(Ev.VAD_START))
        elif ev == SPEECH_END:
            pcm = bytes(self._buf[: len(self._buf) - len(frame)]) if self._buf is not None else b""
            self._buf = None
            self._end_segment(pcm)
        elif self._buf is None:
            self._pre.append(frame)
        elif self.max_seg_ms and len(self._buf) >= self.max_seg_ms * BYTES_PER_MS:
            self._split()
        self.director.pump()

    def set_vad(self, vad: Vad) -> None:
        """発話検出を差し替える (文字起こしモードの Silero VAD)。溜めている区間はそこで終えて認識に回す。"""
        if self._buf is not None:
            pcm, self._buf = bytes(self._buf), None
            self._end_segment(pcm)
        self.vad = vad
        self._onset_head = getattr(vad, "onset_frames", 1) - 1
        self._pre = deque(maxlen=self._onset_head + PRE_ROLL_MS // FRAME_MS)

    def split_at(self, frame_n: int) -> bool:
        """今溜めている区間を、フレーム番号 frame_n (話者の替わり目) の前後 SPLIT_NEAR_MS で一番静かな所で切る。
        区間が無い・前後が SPLIT_MIN_MS より短くなるなら切らずに False。"""
        if self._buf is None:
            return False
        fb = FRAME_MS * BYTES_PER_MS
        n = len(self._buf) // fb
        at = frame_n - (self._n - n + 1)  # 区間の中の位置 (先頭のフレームが 0)
        near, least = SPLIT_NEAR_MS // FRAME_MS, SPLIT_MIN_MS // FRAME_MS
        lo, hi = max(least, at - near), min(n - least, at + near)
        if lo >= hi:
            return False
        self._split(lo, hi, "speaker change")
        return True

    def _split(self, lo: int | None = None, hi: int | None = None, why: str = "") -> None:
        """発話の途中で区間を切る。[lo, hi) のフレーム (既定は直近 SPLIT_LOOK_MS) で一番静かなフレームまでを認識に回し、
        残りを次の区間として溜める。"""
        fb = FRAME_MS * BYTES_PER_MS
        n = len(self._buf) // fb
        if lo is None:
            lo, hi = n - min(n - 1, SPLIT_LOOK_MS // FRAME_MS), n
        levels = [rms(self._buf[i * fb:(i + 1) * fb]) for i in range(lo, hi)]
        cut = (lo + levels.index(min(levels)) + 1) * fb
        pcm, rest = bytes(self._buf[:cut]), self._buf[cut:]
        self._buf = bytearray()
        log.info("segment split at %dms (%s)", cut // BYTES_PER_MS, why or f"quietest of last {SPLIT_LOOK_MS}ms")
        self._end_segment(pcm)
        self._buf = bytearray(rest)
        self._lead = b""
        self._start_n = self._n - len(rest) // fb + 1 + self._onset_head  # 残りの最初のフレームが区間の頭
        self.director.enqueue(Event(Ev.VAD_START))  # 発話は続いている: 終話確認からユーザー発話中へ戻す

    def partial_audio(self) -> bytes | None:
        """今溜めている区間の音 (前の音を含む)。発話中でなければ None。文字起こしの仮の文字に使う。"""
        return None if self._buf is None else self._lead + bytes(self._buf)

    def _end_segment(self, pcm: bytes) -> None:
        seg_ms = len(pcm) // BYTES_PER_MS
        self.director.enqueue(Event(Ev.VAD_END, seg_ms=seg_ms))
        audio, head, self._head = self._lead + pcm, self._head, None
        if head is not None and (self._start_n - head[1]) * FRAME_MS <= JOIN_GAP_MS:
            log.info("joined discarded head: %dms + %dms", len(head[0]) // BYTES_PER_MS, len(audio) // BYTES_PER_MS)
            audio = head[0] + audio
        if seg_ms < self.cfg.MIN_SEG:
            self.noise_log.append(seg_ms)
            log.info("noise segment discarded: %dms < MIN_SEG=%d", seg_ms, self.cfg.MIN_SEG)
            self._head = (audio, self._n)
            return
        if self.on_speech_end:
            self.on_speech_end()
        self._asr(pcm, audio)

    def hold(self, uid: int) -> None:
        """uid の区間を語なしで捨てた (Director が物音の聞き違いと判断)。次の区間が JOIN_GAP_MS 以内に始まれば頭に足す。
        自声・ドラマの回り込みは足さない (AI やドラマの言葉が利用者の発話に混ざる) ので、呼ぶのは語なしの破棄だけ。"""
        seg = self._segs.get(uid)
        if seg is not None and uid == self._uid:  # 後の区間をもう認識に出していたら足せない
            self._head = seg

    def _asr(self, pcm: bytes, audio: bytes) -> None:
        """pcm は VAD の区間 (声の特徴に使う)、audio は ASR に渡す音 (前の音・捨てた区間を頭に足したもの)。"""
        if self.asr is None:
            return
        self._uid += 1
        uid = self._uid
        name = self._put_shm(audio)
        self._segs[uid] = (audio, self._n)
        start = self._start_n - self._onset_head  # 途中で切った区間の終わりは今のフレームではないので、長さから出す
        self.seg_span[uid] = (start, start + len(pcm) // (FRAME_MS * BYTES_PER_MS))
        self.asr_pending += 1
        voice = self._start_embed(pcm) if len(pcm) // BYTES_PER_MS >= self.voice_min_ms else None
        audio_s = len(audio) / BYTES_PER_MS / 1000

        def work() -> tuple[object, object]:
            try:
                if uid <= self._discarded_until:
                    return _FAILED, None
                res = self.asr.transcribe(uid, name, audio_s)
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
        if self.voice_max_ms:
            pcm = pcm[:self.voice_max_ms * BYTES_PER_MS]
        return self._embed_pool.submit(self.embed, pcm)

    def _asr_done(self, uid: int, res: object, voice: object = None) -> None:
        try:
            self._settle(uid, res, voice)
        finally:
            self._segs.pop(uid, None)
            self.seg_span.pop(uid, None)

    def _settle(self, uid: int, res: object, voice: object) -> None:
        self.asr_pending -= 1
        if uid <= self._discarded_until:
            return
        if voice is not None and self.on_voice and isinstance(res, dict):
            self.on_voice(uid, voice)
        latest = uid == self._uid  # 後続区間の認識待ちがあれば、終話・破棄はそちらの結果で確定させる
        if res is None:  # 幻聴 (E4): 応答トリガを送らない
            log.info("asr hallucination discarded: utterance %d", uid)
            self.hold(uid)
        if (res is _FAILED or res is None) and self.on_discard is not None and self.on_discard(uid):
            self._head = None
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
