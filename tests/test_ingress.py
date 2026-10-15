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
    assert d.ctx.state is State.IDLE  # 捨てた発話は無かったことにして待機へ戻す (DISCARDED)


def test_transcript_rejected_by_callback_is_discarded():
    d, ing = make({0: SPEECH_START, 300: SPEECH_END}, asr=asr_ok())
    trig = []
    d._on_actions = lambda a: trig.extend(x.name for x in a)
    ing.on_transcript = lambda u, t: False  # 雑音・自声として捨てる
    run(ing, 301)
    assert "call_llm" not in trig and "discard_segment" in trig
    assert d.ctx.state is State.IDLE


def test_onset_frames_ignore_short_click_and_keep_speech_head():
    v = EnergyVad(threshold=500, hangover_frames=3, onset_frames=3)
    quiet = b"\x00\x00" * 320
    loud = struct.pack("<h", 1000) * 320
    assert [v.process(f) for f in (loud, loud, quiet)] == [None, None, None]  # 2 フレームの物音は発話にしない
    assert [v.process(loud) for _ in range(3)] == [None, None, SPEECH_START]
    d = Director(Config())
    ing = Ingress(d, EnergyVad(threshold=500, hangover_frames=3, onset_frames=3))
    for f in (quiet, loud, loud, loud):
        ing.feed(f)
    assert d.ctx.state is State.USER_SPEAKING and bytes(ing._buf) == loud * 3  # 開始判定前の 2 フレームも区間に含む


def _tone(amp):
    return struct.pack("<h", amp) * 320


def test_adaptive_threshold_follows_noise_floor():
    v = EnergyVad(threshold=500, hangover_frames=3, adaptive=True, ratio=3.0, min_threshold=100)
    assert v.current_threshold == 500  # 推定がそろうまでは固定値
    for _ in range(100):  # 2 秒の雑音 (RMS 200)
        assert v.process(_tone(200)) is None
    assert abs(v.current_threshold - 600) < 1  # 200 x 3
    assert v.process(_tone(500)) is None  # 固定値なら発話扱いの音も、雑音が大きい部屋では発話にしない
    assert v.process(_tone(2000)) == SPEECH_START
    v2 = EnergyVad(threshold=500, adaptive=True, ratio=3.0, min_threshold=100)
    for _ in range(100):  # 静かな部屋では下限まで下がる
        v2.process(_tone(10))
    assert v2.current_threshold == 100


def test_adaptive_noise_floor_rises_slowly_during_speech():
    v = EnergyVad(threshold=500, hangover_frames=1000, adaptive=True)
    for _ in range(100):
        v.process(_tone(100))
    for _ in range(250):  # 5 秒話し続けても雑音の推定は 2.5dB しか上がらない
        v.process(_tone(3000))
    assert v.current_threshold < 300 * 10 ** (3 / 20)


def test_adaptive_playing_threshold_uses_echo_level():
    v = EnergyVad(threshold=500, hangover_frames=3, adaptive=True, ratio=3.0, min_threshold=100)
    for _ in range(100):
        v.process(_tone(50))
    assert abs(v.current_threshold - 150) < 1
    v.playing = True
    for _ in range(300):  # 再生中の回り込み (RMS 400) は雑音の推定に混ぜない
        v.process(_tone(400))
    assert abs(v.current_threshold - 1200) < 1  # 400 x 3
    v.playing = False
    assert abs(v.current_threshold - 150) < 1


def test_adaptive_bleed_threshold_separate_from_ai_echo():
    v = EnergyVad(threshold=500, hangover_frames=3, adaptive=True, ratio=3.0, min_threshold=100)
    for _ in range(100):
        v.process(_tone(50))
    v.playing = True
    for _ in range(300):  # AI の回り込み (AEC で小さい)
        v.process(_tone(100))
    v.playing = False
    v.bleed = True
    for _ in range(300):  # ドラマの音の回り込み (AEC が効かず大きい) は別に測る
        v.process(_tone(1000))
    assert abs(v.current_threshold - 3000) < 1  # 1000 x 3
    assert v.process(_tone(2000)) is None  # 回り込みより少し大きい程度の音 (笑い声等) では発話にしない
    v.bleed = False
    assert abs(v.current_threshold - 150) < 1  # 鳴り止んだら雑音の推定に戻る (雑音に混ぜていない)
    v.playing = True
    assert abs(v.current_threshold - 300) < 1  # AI の回り込みは別のまま
