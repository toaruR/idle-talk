"""ASR ワーカー (設計書 12.1)。req 検証・ハルシネーション除外・HTTP ハンドラ・faster-whisper 呼び出し。"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import shared_memory

from jsonschema import Draft7Validator

from .config import Config
from .errors import ERR_ASR_FAILED

ASR_REQ_SCHEMA = {"type":"object","required":["utterance_id","pcm_ref","lang"],"additionalProperties":False,"properties":{"utterance_id":{"type":"integer","minimum":0,"maximum":4294967295},"pcm_ref":{"type":"string","minLength":1},"lang":{"const":"ja"}}}
ASR_RES_SCHEMA = {"type":"object","required":["utterance_id","text","no_speech_prob","compression_ratio"],"additionalProperties":False,"properties":{"utterance_id":{"type":"integer","minimum":0},"text":{"type":"string"},"no_speech_prob":{"type":"number","minimum":0,"maximum":1},"compression_ratio":{"type":"number","minimum":0}}}

_REQ = Draft7Validator(ASR_REQ_SCHEMA)


def is_hallucination(no_speech_prob: float, compression_ratio: float, cfg: Config | None = None) -> bool:
    """no_speech_prob > 0.6 または compression_ratio > 2.4 で破棄 (境界値は採用)。"""
    h = (cfg or Config()).ASR_HALLU
    return no_speech_prob > h.no_speech_prob or compression_ratio > h.compression_ratio


def err_body(message: str) -> dict:
    return {"error": {"code": ERR_ASR_FAILED, "message": message[:200]}}


def handle_request(raw: bytes, transcribe: Callable[[int, str], dict]) -> tuple[int, dict]:
    """(HTTP status, body)。不正 req は 400 + {"error":{"code","message"}}。"""
    try:
        req = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return 400, err_body("invalid json")
    errs = sorted(_REQ.iter_errors(req), key=lambda e: list(e.path))
    if errs:
        return 400, err_body(errs[0].message)
    try:
        return 200, transcribe(req["utterance_id"], req["pcm_ref"])
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as e:  # ASR モデル/共有メモリ (FileNotFoundError は OSError) 由来。500 を返して worker を落とさない
        return 500, err_body(str(e))


def make_server(transcribe: Callable[[int, str], dict], host: str = "127.0.0.1", port: int = 0) -> HTTPServer:
    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            status, body = handle_request(self.rfile.read(n), transcribe)
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    return HTTPServer((host, port), H)


def load_whisper_model():
    """faster-whisper モデルを読み込む (未導入なら ImportError)。"""
    from faster_whisper import WhisperModel

    return WhisperModel("small", device="cuda", compute_type="float16")


def make_transcribe(model, cfg: Config | None = None) -> Callable[[int, str], dict]:
    """make_server 用 transcribe(utterance_id, pcm_ref)。pcm_ref は共有メモリ名 (設計書 12.1)。

    ハルシネーション除外後に残った全セグメントを連結して 1 件の res にする。
    全セグメントが除外された場合は no_speech_prob=1.0 の空 res を返し、呼出側 AsrClient が破棄する。
    """
    def transcribe(utterance_id: int, pcm_ref: str) -> dict:
        shm = shared_memory.SharedMemory(name=pcm_ref)
        try:
            pcm = bytes(shm.buf)
        finally:
            shm.close()
        segs = transcribe_pcm(pcm, utterance_id, model, cfg)
        if not segs:
            return {"utterance_id": utterance_id, "text": "", "no_speech_prob": 1.0, "compression_ratio": 0.0}
        return {"utterance_id": utterance_id, "text": "".join(x["text"] for x in segs),
                "no_speech_prob": max(x["no_speech_prob"] for x in segs),
                "compression_ratio": max(x["compression_ratio"] for x in segs)}
    return transcribe


def main(argv: list[str] | None = None, load_model: Callable[[], object] = load_whisper_model,
         run: Callable[[HTTPServer], None] | None = None) -> int:
    """python -m idle_talk.asr_worker。モデルは起動時に 1 回だけ読み込む。"""
    p = argparse.ArgumentParser(prog="idle-talk-asr")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args(argv)
    try:
        model = load_model()
    except ImportError as e:
        print(f"asr_worker: faster_whisper is not installed ({e}); pip install faster-whisper",
              file=sys.stderr)
        return 2
    server = make_server(make_transcribe(model), "127.0.0.1", args.port)
    try:
        (run or (lambda s: s.serve_forever()))(server)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def transcribe_pcm(pcm_s16le: bytes, utterance_id: int = 0, model=None, cfg: Config | None = None) -> list[dict]:
    """faster-whisper で 16kHz mono s16le PCM を認識し、ハルシネーション除外後のセグメントを返す。"""
    import numpy as np

    if model is None:
        model = load_whisper_model()
    audio = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32) / 32768.0
    segs, _ = model.transcribe(audio, language="ja")
    out = []
    for s in segs:
        if is_hallucination(s.no_speech_prob, s.compression_ratio, cfg):
            continue
        out.append({"utterance_id": utterance_id, "text": s.text,
                    "no_speech_prob": float(s.no_speech_prob), "compression_ratio": float(s.compression_ratio)})
    return out


if __name__ == "__main__":
    raise SystemExit(main())
