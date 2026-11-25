"""文字起こしモードの Silero VAD (設計書 6.10): 声で区間を始めて終え、BGM だけでは始めない。モード中だけ差し替える。"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.config import Transcribe
from idle_talk.director import LEVEL_LOG_S, IdleTalkDirector
from idle_talk.main import SILERO_FILE
from idle_talk.silero_vad import CHUNK, CONTEXT, SPEECH_OFF, SileroVad, load_session
from idle_talk.store import Store
from idle_talk.vad import SPEECH_END, SPEECH_START, EnergyVad

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "data" / SILERO_FILE
CLIP = ROOT / "data/asr_bench_tv/tv009.wav"
need_model = pytest.mark.skipif(not (MODEL.is_file() and CLIP.is_file()), reason="silero model / clip not available")


def frames(pcm: np.ndarray):
    b = pcm.astype(np.int16).tobytes()
    return [b[i:i + 640] for i in range(0, len(b) - 639, 640)]


def bgm(seconds: float, amp: int = 800) -> np.ndarray:
    t = np.arange(int(16000 * seconds)) / 16000
    return amp * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 330 * t))


@need_model
@pytest.mark.parametrize("idle_reset", [False, True])
def test_speech_starts_and_ends_but_bgm_alone_does_not(idle_reset):
    import soundfile as sf

    voice, _ = sf.read(CLIP, dtype="int16")
    vad = SileroVad(load_session(MODEL), idle_reset=idle_reset)
    events = [vad.process(f) for f in frames(np.concatenate([bgm(3), voice + bgm(len(voice) / 16000), bgm(3)]))]
    starts = [i for i, e in enumerate(events) if e == SPEECH_START]
    ends = [i for i, e in enumerate(events) if e == SPEECH_END]
    assert starts and starts[0] >= 3 * 50 - 5  # BGM だけの 3 秒では始めない
    assert ends and ends[-1] > starts[0] and not vad._speaking  # 声の後の BGM で終わる


class FakeSilero:
    def __init__(self):
        self.inputs = []
        self.prob = 0.1

    def run(self, _, inputs):
        self.inputs.append(inputs)
        return np.array([self.prob]), inputs["h"] + 1, inputs["c"] + 1


def test_idle_reset_counts_inferences_preserving_pending_audio_and_context():
    session = FakeSilero()
    vad = SileroVad(session, idle_reset=True)
    pcm = np.arange(320 * 101, dtype=np.int16)
    data = frames(pcm)
    for frame in data[:99]:
        assert vad.process(frame) is None
    assert len(session.inputs) == 61 and vad.state_resets == 0

    assert vad.process(data[99]) is None
    assert len(session.inputs) == 62 and vad.state_resets == 1 and vad.prob == 0.1
    assert np.all(session.inputs[61]["h"] == 61) and np.all(session.inputs[61]["c"] == 61)
    assert vad.process(data[100]) is None
    received = session.inputs[62]
    assert not received["h"].any() and not received["c"].any()
    np.testing.assert_array_equal(received["input"][0], pcm[62 * CHUNK - CONTEXT:63 * CHUNK] / 32768)

    for frame in data:
        vad.process(frame)
    assert vad.state_resets == 2
    vad.reset()
    assert vad.state_resets == 0
    for frame in data[:99]:
        vad.process(frame)
    assert vad.state_resets == 0


def test_idle_reset_requires_uninterrupted_low_confidence_outside_speech():
    session = FakeSilero()
    vad = SileroVad(session, idle_reset=True)
    frame = bytes(CHUNK * 2)
    for _ in range(61):
        assert vad.process(frame) is None
    session.prob = SPEECH_OFF
    assert vad.process(frame) is None
    session.prob = 0.1
    for _ in range(61):
        assert vad.process(frame) is None
    assert vad.state_resets == 0

    session.prob = 0.9
    assert vad.process(frame) == SPEECH_START
    for _ in range(80):
        assert vad.process(frame) is None
    session.prob = 0.1
    for _ in range(9):
        assert vad.process(frame) is None
    assert vad.process(frame) == SPEECH_END
    for _ in range(61):
        assert vad.process(frame) is None
    assert vad.state_resets == 0
    vad.process(frame)
    assert vad.state_resets == 1


def test_onset_after_reset_requires_two_distinct_inferences():
    session = FakeSilero()
    vad = SileroVad(session, idle_reset=True)
    frame = bytes(640)
    for _ in range(100):
        assert vad.process(frame) is None
    assert vad.state_resets == 1
    session.prob = 0.51
    assert vad.process(frame) is None  # 1回目の高い判定
    assert vad.process(frame) is None  # 同じ判定を再利用しても開始しない
    session.prob = 0.1
    assert vad.process(frame) is None
    session.prob = 0.9
    assert vad.process(frame) is None  # 連続性が切れたので、再び1回目
    assert vad.process(frame) is None  # 次の推論がまだ来ていない
    assert vad.process(frame) == SPEECH_START


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "none.onnx")


@pytest.mark.parametrize("idle_reset", [False, True])
def test_transcribe_swaps_vad_only_during_mode(tmp_path, caplog, idle_reset):
    async def run():
        clock = FakeClock()
        store = Store(tmp_path / "t.db", db_backup=False)
        energy = EnergyVad()
        d = IdleTalkDirector(store, lambda m: iter(()), lambda t: b"", lambda: None, energy,
                             AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
        d.silero = FakeSilero()
        d.transcribe = Transcribe(silero_idle_reset=idle_reset)
        b = MockBrowser()
        b.attach(d.core)
        try:
            b.core.on_text(b, json.dumps({"t": "ui", "transcribe": True}))
            assert isinstance(d.ingress.vad, SileroVad) and d.ingress._onset_head == 0
            with caplog.at_level("INFO", logger="idle_talk.director"):
                for i in range(LEVEL_LOG_S * 50):
                    d.ingress.asr_pending = 3 if i == 7 else 0
                    b.send_frame(bytes(640))
            log_line = next(message for message in caplog.messages if message.startswith("transcribe level:"))
            assert f"state_resets={2 if idle_reset else 0} asr_pending_max=3" in log_line
            b.core.on_text(b, json.dumps({"t": "ui", "transcribe": False}))
            assert d.ingress.vad is energy and d._energy_vad is None
            b.core.on_text(b, json.dumps({"t": "ui", "transcribe": True}))
            assert d.ingress.vad.state_resets == 0
        finally:
            store.close()
    asyncio.run(run())
