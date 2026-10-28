"""文字起こしモードの話者の切り替わり (設計書 6.10)。認識の区間とは別に、マイクの音を WINDOW_MS の窓で HOP_MS ずつ
ずらして声の特徴を計算し、話者が替わった時刻を出す。

区間ごとに 1 回だけ声を見ると、BGM で区間が長くなったときに何人もの声が 1 区間に混ざり、どこで替わったか分からなかった
(実機 2026-10-28)。窓の照合は VoiceBook (Director がイベントループで呼ぶ) に任せ、ここは窓を切り出して声の特徴を
別スレッドで計算するだけにする。計算が前の窓に追いつかなければその窓は飛ばす (遅れを溜めない)。
"""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from .vad import FRAME_MS

log = logging.getLogger("idle_talk.speaker_track")

WINDOW_MS = 1500  # 声の特徴を測る窓 (話者識別の MIN_VOICE_MS と同じ。これより短いと埋め込みが不安定)
HOP_MS = 500  # 窓をずらす間隔
VOICED_MIN = 0.5  # 窓のうち VAD の閾値を超えたフレームがこの割合未満なら測らない (無音・BGM だけの窓)
CONFIRM = 2  # 別の話者がこの回数続いたら替わったとみなす (1 窓だけの揺れで替わったことにしない)


class SpeakerTrack:
    """feed にマイクの 20ms フレームを渡す。窓がそろうと embed を別スレッドで回し、done(終わりのフレーム番号, 窓の音,
    声の特徴) を呼ぶ (呼ぶスレッドは submit が決める。Director はイベントループに戻す)。"""

    def __init__(self, embed: Callable[[bytes], object], done: Callable[[int, bytes, object], None],
                 submit: Callable[[Callable[[], None]], None] | None = None) -> None:
        self.embed = embed
        self.done = done
        self._frames: deque[tuple[bytes, bool]] = deque(maxlen=WINDOW_MS // FRAME_MS)
        self._since_hop = 0
        self._pool: ThreadPoolExecutor | None = None
        self._busy: Future | None = None
        self._submit = submit

    def reset(self) -> None:
        self._frames.clear()
        self._since_hop = 0

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None

    def feed(self, frame: bytes, n: int, voiced: bool) -> None:
        """n はそのフレームの番号 (Ingress._n)。voiced はそのフレームが声か (窓に声があるかの判定に使う)。"""
        self._frames.append((frame, voiced))
        self._since_hop += 1
        if len(self._frames) < self._frames.maxlen or self._since_hop < HOP_MS // FRAME_MS:
            return
        self._since_hop = 0
        if sum(v for _, v in self._frames) < VOICED_MIN * len(self._frames):
            return
        if self._busy is not None and not self._busy.done():
            log.debug("speaker window skipped (embedding busy)")
            return
        pcm = b"".join(f for f, _ in self._frames)
        if self._submit is not None:  # テスト・同期実行
            self._submit(lambda: self.done(n, pcm, self.embed(pcm)))
            return
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="speaker-track")
        self._busy = self._pool.submit(self.embed, pcm)
        self._busy.add_done_callback(lambda f: self._finish(f, n, pcm))

    def _finish(self, fut: Future, n: int, pcm: bytes) -> None:
        if fut.cancelled():
            return
        try:
            emb = fut.result()
        except Exception as e:  # noqa: BLE001  1 窓の失敗で文字起こしを止めない
            log.warning("speaker window embedding failed: %s", e)
            return
        if emb is not None:
            self.done(n, pcm, emb)


class ChangeDetector:
    """窓ごとの話者 id から、替わった時点を決める。同じ別の話者が CONFIRM 回続いたら、その最初の窓で替わったとする。"""

    def __init__(self) -> None:
        self.current: int | None = None
        self._cand: int | None = None
        self._count = 0
        self._first: int | None = None

    def reset(self) -> None:
        self.current, self._cand, self._count, self._first = None, None, 0, None

    def push(self, sid: int | None, n: int) -> tuple[int | None, int, int] | None:
        """替わったら (前の話者, 新しい話者, 替わった窓の終わりのフレーム番号)。sid が None (知らない声) は候補を崩さない。"""
        if sid is None:
            return None
        if sid == self.current:
            self._cand, self._count = None, 0
            return None
        if sid != self._cand:
            self._cand, self._count, self._first = sid, 0, n
        self._count += 1
        if self._count < CONFIRM:
            return None
        prev, self.current, self._cand, self._count = self.current, sid, None, 0
        return prev, sid, self._first
