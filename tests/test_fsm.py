import asyncio
import inspect
import re

import pytest

from idle_talk.config import Config
from idle_talk.epoch import Epoch
from idle_talk.fsm import Action, Ctx, Director, Ev, Event, State, transition
from idle_talk.protocol import MESSAGE_SCHEMA, U32_MAX

CFG = Config()
S = State


def E(kind, **kw):
    return Event(kind, **kw)


# (id, 現状態, ctx追加, event, 次状態, epoch増分, 含むアクション)
ROWS = [
    ("r01_idle_vad_start", S.IDLE, {}, E(Ev.VAD_START), S.USER_SPEAKING, 0, {"start_asr", "abort_background"}),
    ("r02_speaking_end_ge_min_seg", S.USER_SPEAKING, {}, E(Ev.VAD_END, seg_ms=200), S.EOT_CHECK, 0, {"start_eot_timer"}),
    ("r03_speaking_end_lt_min_seg", S.USER_SPEAKING, {}, E(Ev.VAD_END, seg_ms=199), S.IDLE, 0, {"discard_segment", "record_noise"}),
    ("r04_eot_silence_final", S.EOT_CHECK, {}, E(Ev.SILENCE, silence_ms=400, eot=True), S.PREPARING, 0, {"finalize_asr", "call_llm"}),
    ("r05_eot_vad_resume", S.EOT_CHECK, {}, E(Ev.VAD_START), S.USER_SPEAKING, 0, {"cancel_eot_timer"}),
    ("r06_preparing_first_clause", S.PREPARING, {"epoch": 3}, E(Ev.FIRST_CLAUSE, epoch=3), S.AI_PLAYING, 0, {"tts_enqueue"}),
    ("r07_preparing_vad_start", S.PREPARING, {"epoch": 3}, E(Ev.VAD_START), S.USER_SPEAKING, 1, {"cancel_generation", "cancel"}),
    ("r08_playing_vad_start", S.AI_PLAYING, {}, E(Ev.VAD_START), S.INTERRUPT_CHECK, 0, {"duck"}),
    ("r09_playing_done", S.AI_PLAYING, {"epoch": 2}, E(Ev.PLAYBACK_DONE, epoch=2), S.IDLE, 0, {"commit_history"}),
    ("r10_check_backchannel", S.INTERRUPT_CHECK, {}, E(Ev.JUDGED, verdict="backchannel", elapsed_ms=400), S.AI_PLAYING, 0, {"resume"}),
    ("r11_check_interrupt", S.INTERRUPT_CHECK, {}, E(Ev.JUDGED, verdict="interrupt", elapsed_ms=100), S.USER_SPEAKING, 1, {"cancel", "discard_old", "stop_tts"}),
    ("r12_any_to_rest", S.USER_SPEAKING, {}, E(Ev.MUTE), S.REST, 0, {"stop_background"}),
    ("r13_rest_wake", S.REST, {}, E(Ev.WAKE, power_save_released=True), S.IDLE, 0, {"warmup"}),
]


def _doc_row_count():
    import pathlib

    doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "design-idle-talk.md"
    text = doc.read_text(encoding="utf-8")
    sec = text.split("### 4.2 遷移表")[1].split("### 4.3")[0]
    rows = [ln for ln in sec.splitlines() if ln.startswith("|") and not ln.startswith("|---")]
    return len(rows) - 1  # ヘッダ除く


def test_table_rows_count_matches_design():
    ids = [r[0] for r in ROWS]
    assert len(ids) == 13 == len(set(ids))
    assert _doc_row_count() == 13


@pytest.mark.parametrize("row", ROWS, ids=[r[0] for r in ROWS])
def test_table_rows(row):
    _, st, extra, ev, nxt, de, acts = row
    ctx = Ctx(state=st, **extra)
    new, actions = transition(ctx, ev, CFG)
    assert new.state is nxt
    assert new.epoch == ctx.epoch + de
    assert acts <= {a.name for a in actions}


def test_epoch_only_on_confirmed_cancel():
    # 発話開始のみの割り込み確認中は +0
    new, _ = transition(Ctx(state=S.AI_PLAYING, epoch=5), E(Ev.VAD_START), CFG)
    assert new.state is S.INTERRUPT_CHECK and new.epoch == 5
    # 確定 cancel で +1、アクションに新 epoch が載る
    new2, acts = transition(new, E(Ev.JUDGED, verdict="unknown"), CFG)
    assert new2.epoch == 6
    assert [a.epoch for a in acts if a.name == "cancel"] == [6]
    # 返答準備中の VAD 開始で +1
    new3, _ = transition(Ctx(state=S.PREPARING, epoch=9), E(Ev.VAD_START), CFG)
    assert new3.epoch == 10


def test_epoch_stale_completion_ignored():
    ctx = Ctx(state=S.PREPARING, epoch=4)
    new, acts = transition(ctx, E(Ev.FIRST_CLAUSE, epoch=3), CFG)
    assert new == ctx and acts == ()
    play = Ctx(state=S.AI_PLAYING, epoch=4)
    new, acts = transition(play, E(Ev.PLAYBACK_DONE, epoch=3), CFG)
    assert new == play and acts == ()
    # キャンセルと完了の競合: cancel 後に旧 epoch の完了が来ても履歴確定しない
    d = Director(ctx=Ctx(state=S.AI_PLAYING, epoch=7))
    d.enqueue(E(Ev.VAD_START, strong=True))
    d.enqueue(E(Ev.PLAYBACK_DONE, epoch=7))
    acts = d.pump()
    assert "commit_history" not in [a.name for a in acts]
    assert d.ctx.state is S.USER_SPEAKING and d.ctx.epoch == 8


def test_epoch_class():
    e = Epoch()
    assert e.current == 0 and e.bump() == 1 and e.is_current(1) and not e.is_current(0)
    assert Epoch(U32_MAX).bump() == 0


def test_boundary_t_silence():
    c = Ctx(state=S.EOT_CHECK)
    n, _ = transition(c, E(Ev.SILENCE, silence_ms=CFG.T_silence, eot=True), CFG)
    assert n.state is S.PREPARING
    n, _ = transition(c, E(Ev.SILENCE, silence_ms=CFG.T_silence - 1, eot=True), CFG)
    assert n.state is S.EOT_CHECK
    n, _ = transition(c, E(Ev.SILENCE, silence_ms=500, eot=False), CFG)
    assert n.state is S.EOT_CHECK


def test_boundary_min_seg():
    c = Ctx(state=S.USER_SPEAKING)
    n, a = transition(c, E(Ev.VAD_END, seg_ms=CFG.MIN_SEG - 1), CFG)
    assert n.state is S.IDLE and "record_noise" in {x.name for x in a}
    n, _ = transition(c, E(Ev.VAD_END, seg_ms=CFG.MIN_SEG), CFG)
    assert n.state is S.EOT_CHECK


def test_boundary_t_confirm():
    c = Ctx(state=S.INTERRUPT_CHECK)
    n, _ = transition(c, E(Ev.JUDGED, verdict="backchannel", elapsed_ms=CFG.T_confirm), CFG)
    assert n.state is S.AI_PLAYING
    n, _ = transition(c, E(Ev.JUDGED, verdict="backchannel", elapsed_ms=CFG.T_confirm + 1), CFG)
    assert n.state is S.USER_SPEAKING and n.epoch == 1


def test_muted_vad_start_stays_idle():
    c = Ctx(state=S.IDLE, muted=True)
    n, a = transition(c, E(Ev.VAD_START), CFG)
    assert n.state is S.IDLE and a == ()


def test_strong_utterance_cancels_immediately():
    n, a = transition(Ctx(state=S.AI_PLAYING), E(Ev.VAD_START, strong=True), CFG)
    assert n.epoch == 1 and "cancel" in {x.name for x in a}


def test_enqueue_only_state_unchanged_before_pump():
    d = Director()
    d.enqueue(E(Ev.VAD_START))
    assert d.ctx.state is S.IDLE
    d.pump()
    assert d.ctx.state is S.USER_SPEAKING


def test_wire_names_match_state_enum():
    enum = next(o for o in MESSAGE_SCHEMA["oneOf"] if o["properties"]["t"].get("const") == "state")["properties"]["state"]["enum"]
    assert {s.value for s in State} == set(enum) and len(State) == 7


def test_transition_pure_and_no_await_for_cancel():
    assert not inspect.iscoroutinefunction(transition)
    ctx = Ctx(state=S.INTERRUPT_CHECK, epoch=1)
    before = ctx
    r1 = transition(ctx, E(Ev.JUDGED, verdict="interrupt"), CFG)
    r2 = transition(ctx, E(Ev.JUDGED, verdict="interrupt"), CFG)
    assert r1 == r2 and ctx == before
    _, acts = r1
    assert any(a.name == "cancel" for a in acts)
    for a in acts:
        assert isinstance(a, Action)
        assert not inspect.isawaitable(a) and not asyncio.iscoroutine(a)


@pytest.mark.parametrize("st", list(State))
@pytest.mark.parametrize("kind", [Ev.MUTE, Ev.IDLE_TIMEOUT, Ev.FAILURES])
def test_rest_from_any_state(st, kind):
    new, acts = transition(Ctx(state=st), E(kind), CFG)
    assert new.state is S.REST
    if st is not S.REST:
        assert "stop_background" in {a.name for a in acts}
    if kind is Ev.MUTE:
        assert new.muted


def test_rest_wake_unmutes():
    new, acts = transition(Ctx(state=S.REST, muted=True), E(Ev.WAKE), CFG)
    assert new.state is S.IDLE and not new.muted and acts == ()


def test_action_names_regex():
    for r in ROWS:
        for a in r[6]:
            assert re.fullmatch(r"[a-z_]+", a)
