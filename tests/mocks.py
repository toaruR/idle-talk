"""模擬 LLM/TTS/ASR (遅延・失敗注入)。時間は FakeClock を進めて表現する。"""
from __future__ import annotations

import json
import urllib.parse


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


def sse(content: str) -> str:
    return "data: " + json.dumps({"choices": [{"delta": {"content": content}}]}) + "\n"


def mock_llm_stream(clock: FakeClock, tokens, delays=None, done=True, captured=None):
    """delays[i] 秒進めてから i 番目の行を返す stream 関数。"""
    def stream(url, body, timeout):
        if captured is not None:
            captured.append((url, json.loads(body)))
        for i, tok in enumerate(tokens):
            clock.advance((delays or [0] * len(tokens))[i])
            yield sse(tok)
            yield "\n"
        if done:
            yield "data: [DONE]\n"
    return stream


def mock_tts_post(clock: FakeClock, delay=0.0, fail_texts=(), wav=b"RIFF....WAVE"):
    calls = []

    def post(url, body, timeout):
        calls.append(url)
        if any(urllib.parse.quote(f) in url for f in fail_texts):
            raise OSError("synthesis failed")
        clock.advance(delay)
        return b"{}" if "audio_query" in url else wav
    post.calls = calls
    return post


def mock_asr_send(clock: FakeClock, delay=0.0, body=None, status=200):
    def send(req, timeout):
        clock.advance(delay)
        return status, body if body is not None else {
            "utterance_id": req["utterance_id"], "text": "こんにちは",
            "no_speech_prob": 0.1, "compression_ratio": 1.0}
    return send
