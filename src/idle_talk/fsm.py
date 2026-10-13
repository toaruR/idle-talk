"""Director 状態機械 (設計書 4)。遷移は副作用なしの純関数、Director は enqueue + pump で直列化。"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum

from .config import Config
from .protocol import U32_MAX


class State(str, Enum):
    """値は §12.1 の state enum (wire 名) と一致する。"""

    IDLE = "idle"
    USER_SPEAKING = "user_speaking"
    EOT_CHECK = "eot_check"
    PREPARING = "preparing"
    AI_PLAYING = "ai_playing"
    INTERRUPT_CHECK = "interrupt_check"
    REST = "rest"


class Ev(str, Enum):
    VAD_START = "vad_start"
    VAD_END = "vad_end"  # seg_ms
    SILENCE = "silence"  # silence_ms, eot (終話判定結果)
    FIRST_CLAUSE = "first_clause"  # epoch
    PLAYBACK_DONE = "playback_done"  # epoch
    JUDGED = "judged"  # verdict: backchannel|interrupt|unknown, elapsed_ms
    DISCARDED = "discarded"  # 発話区間の認識結果が捨てられた (ASR 失敗・幻聴・語なし・自声)
    MUTE = "mute"
    IDLE_TIMEOUT = "idle_timeout"
    FAILURES = "failures"
    WAKE = "wake"  # power_save_released


@dataclass(frozen=True)
class Event:
    kind: Ev
    seg_ms: int = 0
    silence_ms: int = 0
    eot: bool = False
    epoch: int | None = None
    verdict: str = ""
    elapsed_ms: int = 0
    strong: bool = False  # 強い発話 / 停止語
    power_save_released: bool = False


@dataclass(frozen=True)
class Action:
    """副作用の記述のみ。実行は呼び出し側 (await 不要・即時)。"""

    name: str
    epoch: int | None = None


@dataclass(frozen=True)
class Ctx:
    state: State = State.IDLE
    epoch: int = 0
    muted: bool = False


def _bump(e: int) -> int:
    return 0 if e == U32_MAX else e + 1


def _A(name: str, epoch: int | None = None) -> Action:
    return Action(name, epoch)


_REST_EVENTS = (Ev.MUTE, Ev.IDLE_TIMEOUT, Ev.FAILURES)
_IN_FLIGHT = (State.PREPARING, State.AI_PLAYING, State.INTERRUPT_CHECK)


def transition(ctx: Ctx, ev: Event, cfg: Config) -> tuple[Ctx, tuple[Action, ...]]:
    """純関数。(新 ctx, アクション列) を返す。該当なしは (ctx, ())。"""
    s = ctx.state
    k = ev.kind

    # 任意状態 -> 休止 (行12)
    if k in _REST_EVENTS:
        if s is State.REST:
            return (replace(ctx, muted=True) if k is Ev.MUTE else ctx), ()
        new = replace(ctx, state=State.REST, muted=ctx.muted or k is Ev.MUTE)
        acts: list[Action] = []
        if s in _IN_FLIGHT:
            new = replace(new, epoch=_bump(ctx.epoch))
            acts.append(_A("cancel", new.epoch))
        acts.append(_A("stop_background"))
        return new, tuple(acts)

    # 休止 -> 待機 (行13)
    if s is State.REST:
        if k is Ev.WAKE:
            acts = [_A("warmup")] if ev.power_save_released else []
            return replace(ctx, state=State.IDLE, muted=False), tuple(acts)
        return ctx, ()

    if s is State.IDLE:
        if k is Ev.VAD_START and not ctx.muted:  # 行1
            return replace(ctx, state=State.USER_SPEAKING), (_A("start_asr"), _A("abort_background"))
        return ctx, ()

    if k is Ev.DISCARDED:  # 行14: 雑音・自声だった発話は無かったことにする
        if s in (State.USER_SPEAKING, State.EOT_CHECK):
            return replace(ctx, state=State.IDLE), (_A("discard_segment"),)
        if s is State.INTERRUPT_CHECK:
            return replace(ctx, state=State.AI_PLAYING), (_A("resume", ctx.epoch),)
        return ctx, ()

    if s is State.USER_SPEAKING:
        if k is Ev.VAD_END:
            if ev.seg_ms >= cfg.MIN_SEG:  # 行2
                return replace(ctx, state=State.EOT_CHECK), (_A("start_eot_timer"),)
            return replace(ctx, state=State.IDLE), (_A("discard_segment"), _A("record_noise"))  # 行3
        return ctx, ()

    if s is State.EOT_CHECK:
        if k is Ev.SILENCE and ev.silence_ms >= cfg.T_silence and ev.eot:  # 行4
            return replace(ctx, state=State.PREPARING), (_A("finalize_asr"), _A("call_llm", ctx.epoch))
        if k is Ev.VAD_START:  # 行5
            return replace(ctx, state=State.USER_SPEAKING), (_A("cancel_eot_timer"),)
        return ctx, ()

    if s is State.PREPARING:
        if k is Ev.FIRST_CLAUSE and ev.epoch == ctx.epoch:  # 行6 (旧 epoch は無視)
            return replace(ctx, state=State.AI_PLAYING), (_A("tts_enqueue", ctx.epoch),)
        if k is Ev.VAD_START:  # 行7
            n = _bump(ctx.epoch)
            return replace(ctx, state=State.USER_SPEAKING, epoch=n), (_A("cancel_generation", n), _A("cancel", n))
        return ctx, ()

    if s is State.AI_PLAYING:
        if k is Ev.VAD_START:  # 行8
            if ev.strong:
                n = _bump(ctx.epoch)
                return replace(ctx, state=State.USER_SPEAKING, epoch=n), (_A("cancel", n), _A("discard_old"), _A("stop_tts"))
            return replace(ctx, state=State.INTERRUPT_CHECK), (_A("duck", ctx.epoch),)
        if k is Ev.PLAYBACK_DONE and ev.epoch == ctx.epoch:  # 行9
            return replace(ctx, state=State.IDLE), (_A("commit_history"),)
        return ctx, ()

    if s is State.INTERRUPT_CHECK:
        if k is Ev.VAD_END and ev.seg_ms < cfg.MIN_SEG:  # 行15: 短い物音は認識に回らない。再生を続ける
            return replace(ctx, state=State.AI_PLAYING), (_A("resume", ctx.epoch),)
        if k is Ev.JUDGED:
            if ev.verdict == "backchannel" and ev.elapsed_ms <= cfg.T_confirm:  # 行10
                return replace(ctx, state=State.AI_PLAYING), (_A("resume", ctx.epoch),)
            n = _bump(ctx.epoch)  # 行11: 割り込み / 判断不能 / T_confirm 超過の相槌 (語が認識された場合のみ)
            return replace(ctx, state=State.USER_SPEAKING, epoch=n), (_A("cancel", n), _A("discard_old"), _A("stop_tts"))
        return ctx, ()

    return ctx, ()


class Director:
    """単一コルーチンで直列化。外部コールバックは enqueue のみ、遷移は pump() でのみ起きる。"""

    def __init__(self, cfg: Config | None = None, ctx: Ctx | None = None,
                 on_actions: Callable[[tuple[Action, ...]], None] | None = None) -> None:
        self.cfg = cfg or Config()
        self.ctx = ctx or Ctx()
        self._q: deque[Event] = deque()
        self._on_actions = on_actions

    def enqueue(self, ev: Event) -> None:
        self._q.append(ev)

    def pump(self) -> list[Action]:
        out: list[Action] = []
        while self._q:
            self.ctx, acts = transition(self.ctx, self._q.popleft(), self.cfg)
            if acts:
                out.extend(acts)
                if self._on_actions:
                    self._on_actions(acts)
        return out
