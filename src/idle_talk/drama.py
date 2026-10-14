"""ドラマモード (設計書 6.5)。共有したタブの音を区間化して台詞を溜め、場面の切れ目にたまに一言言うかを決める。

FSM には触れない純ロジック (ドラマの台詞は利用者の発話ではない)。認識と再生は Director が行う。
台詞はメモリにだけ置き、window_s を過ぎたものとモードを止めたときに捨てる (保存も外部送信もしない)。
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from difflib import SequenceMatcher

from .config import Drama
from .vad import BYTES_PER_MS, SPEECH_END, SPEECH_START, Vad

# マイクの認識結果のうち、これだけの割合 (かつ ECHO_MIN_CHARS 字以上) が直近の台詞と連続して一致したら回り込みとみなす
ECHO_RATIO = 0.6
ECHO_MIN_CHARS = 4


def _alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


class DramaListener:
    def __init__(self, settings: Drama, vad: Vad, min_seg_ms: int = 200,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.s = settings
        self.vad = vad
        self.min_seg_ms = min_seg_ms
        self.clock = clock
        self.lines: deque[tuple[float, str]] = deque()  # (認識した時刻, 台詞)
        self._buf: bytearray | None = None
        self._last_end: float | None = None  # 最後の音声区間が終わった時刻
        self._last_react = clock()
        self._new_chars = 0  # 前の一言から溜まった台詞の字数

    # ---- 音声 ----
    def feed(self, frame: bytes) -> bytes | None:
        """1 フレームを処理し、認識に回す区間が確定したらその PCM を返す。"""
        if self._buf is not None:
            self._buf += frame
        ev = self.vad.process(frame)
        if ev == SPEECH_START:
            self._buf = bytearray(frame)
        elif ev == SPEECH_END and self._buf is not None:
            pcm = bytes(self._buf[: len(self._buf) - len(frame)])
            self._buf, self._last_end = None, self.clock()
            return pcm if len(pcm) // BYTES_PER_MS >= self.min_seg_ms else None
        if self._buf is not None and len(self._buf) >= self.s.max_seg_s * 1000 * BYTES_PER_MS:
            pcm, self._buf = bytes(self._buf), bytearray()  # BGM 等で終わらない区間: 切って認識に回し、続きを溜める
            return pcm
        return None

    def add_line(self, text: str) -> None:
        text = text.strip()
        if not _alnum(text):
            return
        self.lines.append((self.clock(), text))
        self._new_chars += len(text)
        self._prune()

    def _prune(self) -> None:
        old = self.clock() - self.s.window_s
        while self.lines and self.lines[0][0] < old:
            self.lines.popleft()

    # ---- 一言の判定 ----
    def quiet_ms(self) -> float:
        """台詞が途切れてからの経過 (ms)。区間の途中なら 0、まだ何も聞いていなければ無限大。"""
        if self._buf is not None:
            return 0.0
        if self._last_end is None:
            return float("inf")
        return (self.clock() - self._last_end) * 1000

    def should_react(self) -> bool:
        return (self.quiet_ms() >= self.s.gap_ms and self.clock() - self._last_react >= self.s.interval_s
                and self._new_chars >= self.s.min_chars)

    def mark_reacted(self) -> None:
        """一言の発火 (黙ると決めた場合も含む)。次は interval_s 空けて、新しい台詞が溜まってから。"""
        self._last_react, self._new_chars = self.clock(), 0

    def context_text(self, since: float | None = None) -> str:
        """直近の台詞 (since 以降) を古い順に改行で連結する。ctx_chars を超える分は古い方から捨てる。"""
        self._prune()
        out: list[str] = []
        n = 0
        for t, text in reversed(self.lines):
            if since is not None and t < since:
                break
            n += len(text) + 1
            if n > self.s.ctx_chars:
                break
            out.append(text)
        return "\n".join(reversed(out))

    def scene_text(self) -> str:
        """前の一言 (または開始) 以降の台詞。ctx_chars を超える分は古い方から捨てる。"""
        return self.context_text(since=self._last_react)

    def contains(self, text: str) -> bool:
        """マイクの認識結果がドラマの台詞 (スピーカーからの回り込み) か。両方とも認識結果なので完全一致は求めない。"""
        mic = _alnum(text)
        if len(mic) < ECHO_MIN_CHARS:
            return False
        self._prune()
        lines = _alnum("".join(t for _, t in self.lines))
        m = SequenceMatcher(None, mic, lines, autojunk=False).find_longest_match(0, len(mic), 0, len(lines))
        return m.size >= max(ECHO_MIN_CHARS, ECHO_RATIO * len(mic))

    def start(self) -> None:
        self.stop()
        self._last_react = self.clock()

    def stop(self) -> None:
        """モードを止めた: 台詞と区間を捨てる。"""
        self.lines.clear()
        self._buf, self._last_end, self._new_chars = None, None, 0
