"""外部サービスクライアント (設計書 3.3 / 12.1)。標準ライブラリのみ。時計と転送は注入可能。"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Iterator

from jsonschema import Draft7Validator

from .asr_worker import ASR_REQ_SCHEMA, ASR_RES_SCHEMA, is_hallucination
from .config import DEFAULT_SPEAKER, Config
from .errors import ERR_ASR_FAILED, ERR_LLM_TIMEOUT, ERR_TTS_FAILED

LLM_REQ_SCHEMA = {"type":"object","required":["model","messages","stream","max_tokens"],"additionalProperties":False,"properties":{"model":{"const":"local"},"messages":{"type":"array","items":{"type":"object","required":["role","content"],"properties":{"role":{"enum":["system","user","assistant"]},"content":{"type":"string"}}}},"stream":{"const":True},"max_tokens":{"const":120}}}
LLM_CHUNK_SCHEMA = {"type":"object","required":["choices"],"properties":{"choices":{"type":"array","items":{"type":"object","required":["delta"],"properties":{"delta":{"type":"object","properties":{"content":{"type":["string","null"]}}}}}}}}

_LLM_REQ = Draft7Validator(LLM_REQ_SCHEMA)
_LLM_CHUNK = Draft7Validator(LLM_CHUNK_SCHEMA)
_ASR_REQ = Draft7Validator(ASR_REQ_SCHEMA)
_ASR_RES = Draft7Validator(ASR_RES_SCHEMA)

DONE = "[DONE]"
WARMUP_TEXT = "あ"
WARMUP_TIMEOUT_S = 60  # 初回合成は DirectML 初期化で十数秒かかる


class ClientError(Exception):
    code = 0


class LlmTimeout(ClientError):
    code = ERR_LLM_TIMEOUT


class TtsTimeout(ClientError):
    code = ERR_TTS_FAILED


class TtsFailed(ClientError):
    code = ERR_TTS_FAILED


class AsrTimeout(ClientError):
    """ASR 超過: 該当発話を破棄する。"""
    code = ERR_ASR_FAILED


class AsrFailed(ClientError):
    code = ERR_ASR_FAILED


def _is_timeout(e: Exception) -> bool:
    return isinstance(e, TimeoutError) or "timed out" in str(e)


# ---------------- LLM ----------------

def build_llm_request(messages: list[dict]) -> dict:
    req = {"model": "local", "messages": messages, "stream": True, "max_tokens": 120}
    errs = list(_LLM_REQ.iter_errors(req))
    if errs:
        raise ValueError(f"invalid llm request: {errs[0].message}")
    return req


def parse_sse(lines: Iterable[str | bytes]) -> Iterator[str]:
    """SSE の data 行から content を順に返す。`[DONE]` で終了。"""
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == DONE:
            return
        obj = json.loads(data)
        if not _LLM_CHUNK.is_valid(obj):
            raise ValueError("invalid sse chunk")
        for ch in obj["choices"]:
            c = ch["delta"].get("content")
            if c:
                yield c


def _http_stream(url: str, body: bytes, timeout: float) -> Iterator[bytes]:
    r = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        yield from resp


def _http_post(url: str, body: bytes | None, timeout: float) -> bytes:
    r = urllib.request.Request(url, data=body or b"", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return resp.read()


class LlmClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", config: Config | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 stream: Callable[[str, bytes, float], Iterable[str | bytes]] = _http_stream):
        self.base_url = base_url.rstrip("/")
        self.cfg = config or Config()
        self.clock = clock
        self._stream = stream

    def stream_chat(self, messages: list[dict]) -> Iterator[str]:
        """content を逐次返す。最初のトークン/全体の超過で LlmTimeout(5001)。"""
        req = build_llm_request(messages)
        t = self.cfg.HTTP_TIMEOUT
        start = self.clock()
        state = {"first": False}
        body = json.dumps(req).encode("utf-8")
        try:
            lines = self._stream(f"{self.base_url}/v1/chat/completions", body, t.llm_total)
            for tok in parse_sse(self._checked(lines, start, state)):
                state["first"] = True
                yield tok
        except OSError as e:
            if _is_timeout(e):
                raise LlmTimeout(str(e)) from e
            raise
        if self.clock() - start > t.llm_total:
            raise LlmTimeout("llm total timeout")

    def _checked(self, lines, start, state) -> Iterator:
        t = self.cfg.HTTP_TIMEOUT
        for line in lines:
            el = self.clock() - start
            if el > t.llm_total:
                raise LlmTimeout(f"llm total > {t.llm_total}s")
            if not state["first"] and el > t.llm_first:
                raise LlmTimeout(f"llm first token > {t.llm_first}s")
            yield line


# ---------------- TTS ----------------

class TtsClient:
    def __init__(self, base_url: str = "http://127.0.0.1:10101", speaker: int = DEFAULT_SPEAKER,
                 config: Config | None = None, clock: Callable[[], float] = time.monotonic,
                 post: Callable[[str, bytes | None, float], bytes] = _http_post):
        self.base_url = base_url.rstrip("/")
        self.speaker = speaker
        self.cfg = config or Config()
        self.clock = clock
        self._post = post

    def synthesize(self, text: str) -> bytes:
        """/audio_query -> /synthesis。1 文 tts_sentence 秒超過で TtsTimeout(5002)。

        AivisSpeech 1.2.0 は /cancellable_synthesis が 501 のため /synthesis を使う。
        中止は epoch 不一致での破棄で代替し、切断後も計算は最大 1 文ぶん続く。"""
        limit = self.cfg.HTTP_TIMEOUT.tts_sentence
        start = self.clock()
        try:
            q = self._post(
                f"{self.base_url}/audio_query?text={urllib.parse.quote(text)}&speaker={self.speaker}",
                None, limit)
            if self.clock() - start > limit:
                raise TtsTimeout(f"tts > {limit}s")
            wav = self._post(f"{self.base_url}/synthesis?speaker={self.speaker}", q, limit)
        except ClientError:
            raise
        except OSError as e:
            if _is_timeout(e):
                raise TtsTimeout(str(e)) from e
            raise TtsFailed(str(e)) from e
        if self.clock() - start > limit:
            raise TtsTimeout(f"tts > {limit}s")
        return wav

    def warmup(self, text: str = WARMUP_TEXT, timeout: float = WARMUP_TIMEOUT_S) -> bool:
        """起動時に 1 回合成して初回の遅延 (GPU 初期化) を済ませる。失敗は False で、致命ではない。"""
        try:
            q = self._post(
                f"{self.base_url}/audio_query?text={urllib.parse.quote(text)}&speaker={self.speaker}",
                None, timeout)
            self._post(f"{self.base_url}/synthesis?speaker={self.speaker}", q, timeout)
            return True
        except OSError:
            return False

    def synthesize_sentences(self, sentences: Iterable[str]) -> list[tuple[str, bytes | None]]:
        """失敗した文は (text, None) とし、次文へ続行する (テキストのみ UI 表示用)。"""
        out: list[tuple[str, bytes | None]] = []
        for s in sentences:
            try:
                out.append((s, self.synthesize(s)))
            except ClientError:
                out.append((s, None))
        return out


# ---------------- ASR ----------------

class AsrClient:
    """ASR ワーカーへの要求。send(req, timeout_s) -> (http_status, body)。"""

    def __init__(self, send: Callable[[dict, float], tuple[int, dict]], config: Config | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._send = send
        self.cfg = config or Config()
        self.clock = clock

    def transcribe(self, utterance_id: int, pcm_ref: str, segment_s: float) -> dict | None:
        """採用結果 dict、ハルシネーション破棄なら None。超過は AsrTimeout (発話破棄)。"""
        req = {"utterance_id": utterance_id, "pcm_ref": pcm_ref, "lang": "ja"}
        if not _ASR_REQ.is_valid(req):
            raise ValueError("invalid asr request")
        limit = self.cfg.HTTP_TIMEOUT.asr(segment_s)
        start = self.clock()
        status, body = self._send(req, limit)
        if self.clock() - start > limit:
            raise AsrTimeout(f"asr > {limit}s")
        if status != 200 or not _ASR_RES.is_valid(body):
            raise AsrFailed(f"asr status={status}")
        if is_hallucination(body["no_speech_prob"], body["compression_ratio"], self.cfg):
            return None
        return body
