"""ドラマモード (設計書 6.5)。共有したタブの音を区間化して台詞を溜め、場面の切れ目にたまに一言言うかを決める。

FSM には触れない純ロジック (ドラマの台詞は利用者の発話ではない)。認識と再生は Director が行う。
台詞はメモリにだけ置き、window_s を過ぎたものとモードを止めたときに捨てる (保存も外部送信もしない)。
window_s より前の話は、裏で要約したあらすじメモ (notes) として覚えておき、長い無音での感想に使う。
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from difflib import SequenceMatcher

from .config import Drama
from .vad import BYTES_PER_MS, SPEECH_END, SPEECH_START, Vad, rms

# マイクの認識結果のうち、これだけの割合 (かつ ECHO_MIN_CHARS 字以上) が直近の台詞と連続して一致したら回り込みとみなす
ECHO_RATIO = 0.6
ECHO_MIN_CHARS = 4
ECHO_SHORT_CHARS = 2  # タブの音と重なった発話はこれ以上なら短くても比べる (「二年前」「十六歳」が通った。実機 2026-10-19)
# タブの音が閾値を超えてからこの間は「鳴っている」とみなす (スピーカーからマイクへの遅れと部屋の残響のぶん)
BLEED_HOLD_S = 0.3
NOTES_MAX_CHARS = 1200  # あらすじメモの上限字数。超えたら古いメモから捨てる (ctx 4k に収める)
# ドラマの台詞のうちこれだけの割合 (SELF_BLOCK 字以上の一致の合計) が直近の AI 発話と一致したら、共有した音に入った AI の声とみなす
SELF_RATIO = 0.6
SELF_BLOCK = 3  # これより短い一致は偶然とみなして数えない (AI の本文は長く、1〜2 字はどこかに必ずある)


# 音量の比で回り込みを見分ける (BleedMeter)。スピーカーからマイクへの回り込みはタブの音量にほぼ比例する
LEVEL_WINDOW_S = 0.3   # マイクの 1 フレームと比べるタブの音の範囲 (この間の最大)。送られる順・スピーカーの遅れ・残響のぶん
LEVEL_HISTORY = 1500   # 回り込みの比 (マイク - タブ, dB) を測る直近のフレーム数 (30 秒)。中央値を使う (話しかけは少ない)
LEVEL_MIN_FRAMES = 50  # 比をこれだけ測るまでは音量で判定しない
LEVEL_MARGIN_DB = 6.0  # 見積もった回り込みよりこれだけ大きいフレームを利用者の声とみなす
LEVEL_VOICE_RATIO = 0.3  # 発話のうち利用者の声のフレームがこの割合未満なら回り込み


def _db(level: float) -> float:
    return 20 * math.log10(max(level, 1.0))


class BleedMeter:
    """タブの音量からマイクに回り込むはずの音量を見積もり、マイクの発話がそれよりはっきり大きいかを測る。
    Chrome の外のアプリ (動画ビュワー等) の音はエコー除去で消えず、文字の一致では認識の違いで漏れた (実機 2026-10-19)。"""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._tab: deque[tuple[float, float]] = deque()  # (時刻, タブの音量 dB)
        self._diffs: deque[float] = deque(maxlen=LEVEL_HISTORY)
        self._seg: list[int] | None = None  # 今の発話の [比べたフレーム数, 声のフレーム数]

    def tab(self, level: float) -> None:
        now = self.clock()
        self._tab.append((now, _db(level)))
        while self._tab and self._tab[0][0] < now - LEVEL_WINDOW_S:
            self._tab.popleft()

    def gain(self) -> float | None:
        """回り込みの比 (dB) の中央値。測り足りなければ None。"""
        if len(self._diffs) < LEVEL_MIN_FRAMES:
            return None
        return sorted(self._diffs)[len(self._diffs) // 2]

    def mic(self, level: float, speaking: bool, sounding: bool) -> None:
        """マイクの 1 フレーム。sounding はタブの音が鳴っている間 (DramaListener.sounding)。"""
        tab = None
        if sounding:
            now = self.clock()
            tab = max((db for t, db in self._tab if t >= now - LEVEL_WINDOW_S), default=None)
        db = _db(level)
        if tab is not None:
            self._diffs.append(db - tab)
        if not speaking:
            return
        if self._seg is None:
            self._seg = [0, 0]
        g = self.gain()
        if tab is not None and g is not None:
            self._seg[0] += 1
            self._seg[1] += db > tab + g + LEVEL_MARGIN_DB

    def take(self) -> float | None:
        """終わった発話のうち利用者の声とみなしたフレームの割合。比べられたフレームが無ければ None。"""
        seg, self._seg = self._seg, None
        if not seg or not seg[0]:
            return None
        return seg[1] / seg[0]

    def reset(self) -> None:
        self._tab.clear()
        self._diffs.clear()
        self._seg = None


def _alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def is_ai_voice(line: str, ai_text: str) -> bool:
    """ドラマの台詞が AI の声 (画面全体の共有でシステム音声に入った) か。認識の誤りがあるので完全一致は求めない。"""
    heard, ai = _alnum(line), _alnum(ai_text)
    if len(heard) < ECHO_MIN_CHARS or not ai:
        return False
    blocks = SequenceMatcher(None, heard, ai, autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= SELF_BLOCK) >= SELF_RATIO * len(heard)


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
        self.notes: list[str] = []  # あらすじメモ (古い順)
        self.undigested: list[str] = []  # まだメモに要約していない台詞 (window_s で捨てない)
        self._last_line: float | None = None  # 最後に台詞を認識した時刻
        self._heard_chars = 0  # 前の感想 (または開始) から聞いた台詞の字数
        self._loud_at: float | None = None  # タブの音が最後に閾値を超えた時刻 (マイクへの回り込みの判定用)
        self.closed = 0  # 閉じた区間の数 (短くて捨てたもの・max_seg_s で切ったものも含む)。Director が認識の待ち合わせに使う

    # ---- 音声 ----
    def feed(self, frame: bytes) -> bytes | None:
        """1 フレームを処理し、認識に回す区間が確定したらその PCM を返す。"""
        if self._buf is not None:
            self._buf += frame
        threshold = getattr(self.vad, "current_threshold", None)
        if threshold is not None and rms(frame) >= threshold:  # 台詞に限らず笑い声・音楽も含む
            self._loud_at = self.clock()
        ev = self.vad.process(frame)
        if ev == SPEECH_START:
            self._buf = bytearray(frame)
        elif ev == SPEECH_END and self._buf is not None:
            pcm = bytes(self._buf[: len(self._buf) - len(frame)])
            self._buf, self._last_end = None, self.clock()
            self.closed += 1
            return pcm if len(pcm) // BYTES_PER_MS >= self.min_seg_ms else None
        if self._buf is not None and len(self._buf) >= self.s.max_seg_s * 1000 * BYTES_PER_MS:
            pcm, self._buf = bytes(self._buf), bytearray()  # BGM 等で終わらない区間: 切って認識に回し、続きを溜める
            self.closed += 1
            return pcm
        return None

    def sounding(self) -> bool:
        """タブの音が今鳴っているか (BLEED_HOLD_S 以内に閾値を超えた)。マイクの VAD が回り込みを測るのに使う。"""
        return self._loud_at is not None and self.clock() - self._loud_at <= BLEED_HOLD_S

    def add_line(self, text: str) -> None:
        text = text.strip()
        if not _alnum(text):
            return
        self.lines.append((self.clock(), text))
        self._new_chars += len(text)
        self._heard_chars += len(text)
        self._last_line = self.clock()
        self.undigested.append(text)
        while sum(map(len, self.undigested)) > 4 * self.s.digest_chars:  # 要約が追いつかない: 古い方から捨てる
            self.undigested.pop(0)
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

    # ---- あらすじメモと感想 ----
    def should_digest(self) -> bool:
        return sum(map(len, self.undigested)) >= self.s.digest_chars

    def add_note(self, note: str, used: int) -> None:
        """要約に使った先頭 used 件の台詞を消し、メモを足す (空なら足さない)。NOTES_MAX_CHARS を超えたら古いメモから捨てる。"""
        del self.undigested[:used]
        if note:
            self.notes.append(note)
        while sum(map(len, self.notes)) > NOTES_MAX_CHARS:
            self.notes.pop(0)

    def should_finale(self) -> bool:
        """台詞が finale_gap_s 聞こえず、前の感想から finale_min_chars 以上聞いていれば、それまでの感想を話す。"""
        return (self._last_line is not None and self.clock() - self._last_line >= self.s.finale_gap_s
                and self._heard_chars >= max(1, self.s.finale_min_chars))

    def finale_material(self) -> tuple[str, str]:
        """(あらすじメモ, メモにしていない最後の台詞)。台詞は ctx_chars を超える分を古い方から捨てる。"""
        rest = "\n".join(self.undigested)[-self.s.ctx_chars:]
        return "\n".join(self.notes), rest

    def mark_finale(self) -> None:
        """感想を話した (黙ると決めた場合も含む): メモを捨て、次の話は新しく覚え直す。"""
        self.notes.clear()
        self.undigested.clear()
        self._heard_chars = 0
        self.mark_reacted()

    def seg_open(self) -> bool:
        """区間の途中か (まだ認識に回していない台詞がある)。"""
        return self._buf is not None

    def contains(self, text: str, overlapped: bool = False) -> bool:
        """マイクの認識結果がドラマの台詞 (スピーカーからの回り込み) か。両方とも認識結果なので完全一致は求めない。
        overlapped (タブの音と重なった発話) なら、ECHO_MIN_CHARS 未満でも台詞にそのまま含まれれば回り込みとする。"""
        mic = _alnum(text)
        self._prune()
        lines = _alnum("".join(t for _, t in self.lines))
        if len(mic) < ECHO_MIN_CHARS:
            return overlapped and len(mic) >= ECHO_SHORT_CHARS and mic in lines
        m = SequenceMatcher(None, mic, lines, autojunk=False).find_longest_match(0, len(mic), 0, len(lines))
        return m.size >= max(ECHO_MIN_CHARS, ECHO_RATIO * len(mic))

    def start(self) -> None:
        self.stop()
        self._last_react = self.clock()

    def stop(self) -> None:
        """モードを止めた: 台詞・メモと区間を捨てる。"""
        self.lines.clear()
        self.notes.clear()
        self.undigested.clear()
        self._buf, self._last_end, self._new_chars = None, None, 0
        self._last_line, self._heard_chars, self._loud_at = None, 0, None
