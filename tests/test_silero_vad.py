"""文字起こしモードの Silero VAD (設計書 6.10): 声で区間を始めて終え、BGM だけでは始めない。モード中だけ差し替える。"""
import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.director import IdleTalkDirector
from idle_talk.main import SILERO_FILE
from idle_talk.silero_vad import SileroVad, load_session
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
def test_speech_starts_and_ends_but_bgm_alone_does_not():
    import soundfile as sf

    voice, _ = sf.read(CLIP, dtype="int16")
    vad = SileroVad(load_session(MODEL))
    events = [vad.process(f) for f in frames(np.concatenate([bgm(3), voice + bgm(len(voice) / 16000), bgm(3)]))]
    starts = [i for i, e in enumerate(events) if e == SPEECH_START]
    ends = [i for i, e in enumerate(events) if e == SPEECH_END]
    assert starts and starts[0] >= 3 * 50 - 5  # BGM だけの 3 秒では始めない
    assert ends and ends[-1] > starts[0] and not vad._speaking  # 声の後の BGM で終わる


def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_session(tmp_path / "none.onnx")


@need_model
def test_transcribe_swaps_vad_only_during_mode(tmp_path):
    async def run():
        clock = FakeClock()
        store = Store(tmp_path / "t.db", db_backup=False)
        energy = EnergyVad()
        d = IdleTalkDirector(store, lambda m: iter(()), lambda t: b"", lambda: None, energy,
                             AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
        d.silero = load_session(MODEL)
        b = MockBrowser()
        b.attach(d.core)
        try:
            b.core.on_text(b, json.dumps({"t": "ui", "transcribe": True}))
            assert isinstance(d.ingress.vad, SileroVad) and d.ingress._onset_head == 0
            b.core.on_text(b, json.dumps({"t": "ui", "transcribe": False}))
            assert d.ingress.vad is energy and d._energy_vad is None
        finally:
            store.close()
    asyncio.run(run())
