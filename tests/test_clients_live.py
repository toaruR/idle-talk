"""実サービス統合試験 (-m integration)。接続不能は失敗として扱う。"""
import json
import os
import time

import pytest

from idle_talk.asr_worker import transcribe_pcm
from idle_talk.clients import (
    DONE,
    LlmClient,
    TtsClient,
    _http_stream,
    build_llm_request,
)
from idle_talk.config import Config, load_ports, load_speaker

pytestmark = pytest.mark.integration

CFG = Config()
PORTS = load_ports()
LLAMA = os.environ.get("LLAMA_URL", f"http://127.0.0.1:{PORTS.llm}")
AIVIS = os.environ.get("AIVIS_URL", f"http://127.0.0.1:{PORTS.tts}")
SPEAKER = int(os.environ.get("AIVIS_SPEAKER", load_speaker()))


@pytest.fixture(scope="module", autouse=True)
def _tts_warm():
    """本番 (main.build_director) と同じく、初回合成の遅延を計測前に済ませる。"""
    assert TtsClient(AIVIS, SPEAKER, CFG).warmup(), "tts warmup failed"


MSGS = [{"role": "user", "content": "こんにちは。一言で挨拶して。"}]


def _stream_lines():
    body = json.dumps(build_llm_request(MSGS)).encode()
    return _http_stream(f"{LLAMA}/v1/chat/completions", body, CFG.HTTP_TIMEOUT.llm_total)


def test_llm_first_token_within_llm_first():
    t0 = time.monotonic()
    got = False
    for line in _stream_lines():
        s = line.decode().strip()
        if s.startswith("data:") and DONE not in s:
            got = True
            break
    assert got, "no token received"
    assert time.monotonic() - t0 <= CFG.HTTP_TIMEOUT.llm_first


def test_llm_done_within_llm_total():
    t0 = time.monotonic()
    seen = [ln.decode().strip() for ln in _stream_lines()]
    assert "data: " + DONE in seen
    assert time.monotonic() - t0 <= CFG.HTTP_TIMEOUT.llm_total


def test_llm_no_think_tags():
    text = "".join(LlmClient(LLAMA, CFG).stream_chat(MSGS))
    assert text and "<think>" not in text


def test_tts_returns_riff_wav_within_tts_sentence():
    t0 = time.monotonic()
    wav = TtsClient(AIVIS, SPEAKER, CFG).synthesize("こんにちは")
    assert time.monotonic() - t0 <= CFG.HTTP_TIMEOUT.tts_sentence
    assert wav[:4] == b"RIFF" and wav[8:12] == b"WAVE"


def test_asr_silence_gives_zero_outputs():
    pcm = b"\x00\x00" * (16000 * CFG.E_PARAMS.silence_input_s)
    assert transcribe_pcm(pcm, cfg=CFG) == []
