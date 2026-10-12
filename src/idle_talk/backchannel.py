"""AI 相槌 (設計書 6.1 / 8.1 SIL_BACKCHANNEL・BC_COOLDOWN / 12.3 BC)。

内容非依存の事前合成クリップを、無音が SIL_BACKCHANNEL 継続したとき 1 発話につき最大 1 回、BC_COOLDOWN を置いて再生する。
無音が T_silence に達した時点で相槌が未完なら取消し、誤発火として全件記録する。
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable

from .clients import ClientError
from .config import Config

log = logging.getLogger("idle_talk.backchannel")

# 発話内容に依存しない固定の相槌語 (引数に発話内容を取らない)。
BACKCHANNEL_PHRASES: tuple[str, ...] = ("うん", "はい", "なるほど")
MISFIRE_KIND = "backchannel_misfire"


def synthesize_clips(tts_client) -> list[tuple[str, bytes]]:
    """固定語を tts_client.synthesize (/audio_query -> /synthesis) で事前合成する。
    失敗した語は除外する。(text, wav) のリストを返す。"""
    clips: list[tuple[str, bytes]] = []
    for text in BACKCHANNEL_PHRASES:
        try:
            wav = tts_client.synthesize(text)
        except (ClientError, OSError) as e:  # 合成失敗は相槌が減るだけ。致命ではない
            log.debug("backchannel synth failed: %s", e)
            continue
        if wav:
            clips.append((text, wav))
    return clips


class BackchannelGate:
    """発火判定 (純ロジック)。時刻は秒 (clock 注入)、無音は ms。"""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._fired = False
        self._last_fire: float | None = None

    def begin_utterance(self) -> None:
        self._fired = False

    def should_fire(self, silence_ms: float, now: float) -> bool:
        """SIL_BACKCHANNEL <= silence < T_silence、未発火、クールダウン経過で True。発火を記録する。"""
        c = self.cfg
        if self._fired or not c.SIL_BACKCHANNEL <= silence_ms < c.T_silence:
            return False
        if self._last_fire is not None and now - self._last_fire < c.BC_COOLDOWN:
            return False
        self._fired = True
        self._last_fire = now
        return True


class Backchannel:
    """gate + 再生/取消/記録コールバック。play(text, wav) / cancel() は通常応答と同一の経路を呼ぶ。"""

    def __init__(self, clips: list[tuple[str, bytes]], play: Callable[[str, bytes], None],
                 cancel: Callable[[], None], record: Callable[[str, dict], None],
                 cfg: Config | None = None, clock: Callable[[], float] | None = None) -> None:
        import time

        self.cfg = cfg or Config()
        self.gate = BackchannelGate(self.cfg)
        self.clips = list(clips)
        self._play, self._cancel, self._record = play, cancel, record
        self._clock = clock or time.monotonic
        self._i = 0
        self.pending = False  # 発火済みで T_silence 未到達
        self.fired = 0
        self.misfires = 0

    def on_speech_start(self) -> None:
        self.gate.begin_utterance()
        self.pending = False

    def on_silence(self, silence_ms: float) -> bool:
        """無音継続を通知する。発火したら True。T_silence 以上なら未完の相槌を取消して誤発火を記録。"""
        if silence_ms >= self.cfg.T_silence:
            self.resolve(silence_ms)
            return False
        if not self.clips or not self.gate.should_fire(silence_ms, self._clock()):
            return False
        text, wav = self.clips[self._i % len(self.clips)]
        self._i += 1
        self.pending = True
        self.fired += 1
        self._play(text, wav)
        self._record("backchannel", {"text": text, "silence_ms": silence_ms})
        return True

    def resolve(self, silence_ms: float | None = None) -> bool:
        """終話確認成立 (T_silence 到達): 発火済みなら取消して backchannel_misfire を記録。"""
        if not self.pending:
            return False
        self.pending = False
        self.misfires += 1
        self._cancel()
        self._record(MISFIRE_KIND, {"reason": "t_silence_reached", "silence_ms": silence_ms})
        return True


def event_payload(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)
