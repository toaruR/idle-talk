import contextlib
import logging
import struct
from multiprocessing import shared_memory

from mocks import FakeClock

from idle_talk.clients import AsrClient
from idle_talk.config import Config
from idle_talk.errors import ERR_MIC_LOST
from idle_talk.fsm import Ctx, Director, State
from idle_talk.ingress import Ingress
from idle_talk.vad import SPEECH_END, SPEECH_START, EnergyVad

FRAME = bytes(struct.pack("<h", 1000)) * 16  # 1ms, 非無音


class ScriptVad:
    """frame 番号 -> イベントの模擬検出器。"""

    def __init__(self, script):
        self.script, self.i = script, 0

    def process(self, frame):
        ev = self.script.get(self.i)
        self.i += 1
        return ev


def run(ing, n):
    for _ in range(n):
        ing.feed(FRAME)


def make(script, state=State.IDLE, asr=None, **kw):
    d = Director(Config(), Ctx(state=state))
    return d, Ingress(d, ScriptVad(script), asr, **kw)


def asr_ok(seen=None, hallu=False):
    def send(req, timeout):
        if seen is not None:
            shm = shared_memory.SharedMemory(name=req["pcm_ref"])
            seen.append(bytes(shm.buf))
            shm.close()
        return 200, {"utterance_id": req["utterance_id"], "text": "やあ",
                     "no_speech_prob": 0.9 if hallu else 0.1, "compression_ratio": 1.0}
    return AsrClient(send)


def test_vad_events_one_to_one():
    d, ing = make({0: SPEECH_START, 300: SPEECH_END})
    seen = []
    d._on_actions = lambda a: seen.extend(x.name for x in a)
    run(ing, 301)
    assert "start_asr" in seen and "start_eot_timer" in seen
    assert d.ctx.state is State.EOT_CHECK


def test_min_seg_199_noise_200_segment(caplog):
    caplog.set_level(logging.INFO)
    sent = []
    d, ing = make({0: SPEECH_START, 199: SPEECH_END}, asr=asr_ok(sent))
    run(ing, 200)
    assert d.ctx.state is State.IDLE and ing.noise_log == [199] and sent == []
    assert "noise segment" in caplog.text
    d, ing = make({0: SPEECH_START, 200: SPEECH_END}, asr=asr_ok(sent))
    run(ing, 201)
    assert d.ctx.state is State.PREPARING and len(sent) == 1 and ing.noise_log == []


def test_shm_roundtrip_and_release():
    sent = []
    _d, ing = make({0: SPEECH_START, 250: SPEECH_END}, asr=asr_ok(sent))
    pcm = FRAME * 250
    run(ing, 251)
    assert sent[0][: len(pcm)] == pcm
    assert ing.shm_count == 0


def test_shm_released_on_asr_failure():
    def send(req, timeout):
        raise RuntimeError("x")
    _d, ing = make({0: SPEECH_START, 250: SPEECH_END}, asr=AsrClient(send))
    with contextlib.suppress(RuntimeError):
        run(ing, 251)
    assert ing.shm_count == 0


def test_vad_during_playback_same_frames():
    script = {0: SPEECH_START, 300: SPEECH_END}
    _, a = make(script, State.IDLE)
    _, b = make(script, State.AI_PLAYING)
    run(a, 400)
    run(b, 400)
    assert a.vad.i == b.vad.i == 400
    assert sum(a.frames_by_state.values()) == sum(b.frames_by_state.values()) == 400


def test_energy_vad_cpu_impl():
    v = EnergyVad(threshold=500, hangover_frames=3)
    quiet = b"\x00\x00" * 16
    assert v.process(quiet) is None
    assert v.process(FRAME) == SPEECH_START
    assert [v.process(quiet) for _ in range(3)] == [None, None, SPEECH_END]


def silent_ing(clock, errs):
    d = Director(Config())
    return d, Ingress(d, ScriptVad({}), clock=clock, on_error=lambda c, m: errs.append((c, m, clock())))


def test_device_lost_3s_within_5s():
    clk, errs = FakeClock(), []
    d, ing = silent_ing(clk, errs)
    silent = b"\x00\x00" * 320
    for _ in range(250):  # 20ms x 250 = 5s
        ing.feed(silent)
        clk.advance(0.02)
    assert errs and errs[0][0] == ERR_MIC_LOST and errs[0][2] <= 5.0
    assert d.ctx.state is State.REST
    assert len(errs) == 1


def test_no_device_lost_at_2_9s():
    clk, errs = FakeClock(), []
    d, ing = silent_ing(clk, errs)
    silent = b"\x00\x00" * 320
    for _ in range(145):  # 2.9s
        ing.feed(silent)
        clk.advance(0.02)
    assert errs == [] and d.ctx.state is State.IDLE


def test_hallucination_sends_no_trigger():
    d, ing = make({0: SPEECH_START, 300: SPEECH_END}, asr=asr_ok(hallu=True))
    trig = []
    d._on_actions = lambda a: trig.extend(x.name for x in a if x.name in ("finalize_asr", "call_llm"))
    got = []
    ing.on_transcript = lambda u, t: got.append(t)
    run(ing, 301)
    assert trig == [] and got == []
    assert d.ctx.state is State.EOT_CHECK
