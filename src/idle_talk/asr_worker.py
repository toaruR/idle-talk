"""ASR ワーカー (設計書 12.1)。req 検証・ハルシネーション除外・HTTP ハンドラ・認識エンジン呼び出し。

エンジンは [asr].engine で選ぶ。reazon (既定) は ReazonSpeech を sherpa-onnx で CPU 実行し、
faster-whisper と同じ形のセグメント (text / no_speech_prob / compression_ratio) を返す。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from multiprocessing import shared_memory
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft7Validator

from .config import SETTINGS_FILE, Asr, Config, load_asr, load_ports
from .errors import ERR_ASR_FAILED
from .portlock import acquire_or_exit

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


REAZON_MAX_S = 25  # ReazonSpeech は約 30 秒までの音声しか扱えないので、長い区間はこの秒数で切る
# ReazonSpeech は物音 (無音・打鍵・机を叩く・咳・ハム) を必ず 1 音の声 (「え」「あ」「何」「あっ」) に聞き違える。
# no_speech_prob が無く幻聴除外で捨てられないので、1 音 (+促音・長音・小書き・句読点) だけの結果は語なしとする。
# 「うん」「へえ」など 2 音以上の相槌は残る。
_REAZON_NOISE = re.compile(r"[あいうえおんアイウエオン何][っッーぁぃぅぇぉ]?[、。!?！？…]*")


def compression_ratio(text: str) -> float:
    """faster-whisper と同じ定義 (UTF-8 の長さ / zlib 圧縮後の長さ)。同じ語の繰り返しで大きくなる。"""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


class ReazonModel:
    """ReazonSpeech (sherpa-onnx の transducer) を faster-whisper の transcribe と同じ形で呼べるようにする。

    no_speech_prob は出ないので 0 とし、語が出ない・物音の聞き違い (_REAZON_NOISE) ならセグメントを返さない (呼出側で語なしとして破棄)。"""

    RATE = 16000

    def __init__(self, recognizer):
        self.rec = recognizer

    def transcribe(self, audio, language="ja"):
        segs = []
        step = REAZON_MAX_S * self.RATE
        for i in range(0, max(len(audio), 1), step):
            s = self.rec.create_stream()
            s.accept_waveform(self.RATE, audio[i:i + step])
            self.rec.decode_stream(s)
            text = s.result.text.strip()
            if text and not _REAZON_NOISE.fullmatch(text):
                segs.append(SimpleNamespace(text=text, no_speech_prob=0.0, compression_ratio=compression_ratio(text)))
        return iter(segs), None


def _pick_onnx(d: Path, stem: str) -> str:
    """<stem>*.onnx を探す。int8 量子化版があればそちらを使う。"""
    found = sorted(d.glob(f"{stem}*.onnx"))
    if not found:
        raise FileNotFoundError(f"{d} に {stem}*.onnx がありません")
    int8 = [f for f in found if "int8" in f.name]
    return str((int8 or found)[0])


def load_reazon_model(asr: Asr) -> ReazonModel:
    """ReazonSpeech を読み込む (sherpa-onnx 未導入なら ImportError、モデルが無ければ FileNotFoundError)。"""
    import sherpa_onnx

    d = Path(asr.model_dir)
    files = {k: _pick_onnx(d, k) for k in ("encoder", "decoder", "joiner")}
    return ReazonModel(sherpa_onnx.OfflineRecognizer.from_transducer(
        **files, tokens=str(d / "tokens.txt"), num_threads=asr.threads, sample_rate=ReazonModel.RATE,
        feature_dim=80, decoding_method="greedy_search"))


def load_asr_model(asr: Asr):
    return load_reazon_model(asr) if asr.engine == "reazon" else load_whisper_model()


_INSTALL_HINT = {"reazon": "sherpa_onnx is not installed ({e}); pip install sherpa-onnx",
                 "whisper": "faster_whisper is not installed ({e}); pip install faster-whisper"}


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


def main(argv: list[str] | None = None, load_model: Callable[[], object] | None = None,
         run: Callable[[HTTPServer], None] | None = None) -> int:
    """python -m idle_talk.asr_worker。モデルは起動時に 1 回だけ読み込む ([asr].engine で選ぶ)。"""
    p = argparse.ArgumentParser(prog="idle-talk-asr")
    p.add_argument("--config", default=SETTINGS_FILE)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--pid-file", default=None)  # 指定時は二重起動を拒否 (exit 4)。ctl が使う
    args = p.parse_args(argv)
    port = load_ports(args.config).asr if args.port is None else args.port
    asr = load_asr(args.config)
    lock = acquire_or_exit(args.pid_file) if args.pid_file else None
    try:
        try:
            model = load_model() if load_model else load_asr_model(asr)
        except ImportError as e:
            print("asr_worker: " + _INSTALL_HINT[asr.engine].format(e=e), file=sys.stderr)
            return 2
        except FileNotFoundError as e:
            print(f"asr_worker: {e} ([asr].model_dir を確認)", file=sys.stderr)
            return 2
        print(f"asr_worker: engine {asr.engine}", file=sys.stderr)
        server = make_server(make_transcribe(model), "127.0.0.1", port)
        try:
            (run or (lambda s: s.serve_forever()))(server)
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return 0
    finally:
        if lock is not None:
            lock.release()


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
