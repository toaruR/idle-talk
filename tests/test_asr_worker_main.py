"""asr_worker.main の試験。faster_whisper / sherpa_onnx は import せず、fake モデル/注入で検証する。"""
import json
import sys
import threading
import urllib.request
from multiprocessing import shared_memory
from types import SimpleNamespace

import numpy as np
import pytest

from idle_talk import asr_worker
from idle_talk.config import Asr, load_asr


class FakeModel:
    def __init__(self, segs):
        self.segs = segs
        self.calls = 0

    def transcribe(self, audio, language):
        self.calls += 1
        return iter(self.segs), None


def seg(text, nsp=0.1, cr=1.0):
    return SimpleNamespace(text=text, no_speech_prob=nsp, compression_ratio=cr)


def test_main_serves_once_with_model_loaded_once():
    loads = []
    model = FakeModel([seg("やあ"), seg("幻聴", nsp=0.9), seg("です")])
    out = {}

    def run(server):
        assert server.server_address[0] == "127.0.0.1"
        shm = shared_memory.SharedMemory(create=True, size=640)
        port = server.server_address[1]

        def client():
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/", data=json.dumps(
                    {"utterance_id": 7, "pcm_ref": shm.name, "lang": "ja"}).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as r:
                out["body"] = json.loads(r.read())

        t = threading.Thread(target=client)
        t.start()
        server.handle_request()
        t.join()
        shm.close()
        shm.unlink()

    rc = asr_worker.main(["--port", "0"], load_model=lambda: loads.append(1) or model, run=run)
    assert rc == 0 and loads == [1] and model.calls == 1
    assert out["body"] == {"utterance_id": 7, "text": "やあです", "no_speech_prob": 0.1,
                           "compression_ratio": 1.0}


def test_all_hallucination_returns_discardable_res():
    shm = shared_memory.SharedMemory(create=True, size=640)
    try:
        tr = asr_worker.make_transcribe(FakeModel([seg("x", nsp=0.9)]))
        res = tr(3, shm.name)
    finally:
        shm.close()
        shm.unlink()
    assert res["text"] == "" and asr_worker.is_hallucination(res["no_speech_prob"], res["compression_ratio"])


def test_default_port_is_8090(monkeypatch):
    seen = {}
    monkeypatch.setattr(asr_worker, "make_server",
                        lambda tr, host, port: seen.update(host=host, port=port) or SimpleNamespace(
                            server_close=lambda: None))
    assert asr_worker.main([], load_model=lambda: object(), run=lambda s: None) == 0
    assert seen == {"host": "127.0.0.1", "port": 8090}


@pytest.mark.parametrize("engine,module,msg", [("whisper", "faster_whisper", "faster_whisper is not installed"),
                                                ("reazon", "sherpa_onnx", "sherpa_onnx is not installed")])
def test_missing_engine_package_exits_nonzero_with_message(monkeypatch, capsys, tmp_path, engine, module, msg):
    monkeypatch.setitem(sys.modules, module, None)  # import を ImportError にする
    f = tmp_path / "s.toml"
    f.write_text(f'[asr]\nengine = "{engine}"\n', encoding="utf-8")
    rc = asr_worker.main(["--config", str(f), "--port", "0"],
                         run=lambda s: (_ for _ in ()).throw(AssertionError("served")))
    assert rc != 0
    assert msg in capsys.readouterr().err


def test_reazon_missing_model_dir_exits_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.setitem(sys.modules, "sherpa_onnx", SimpleNamespace(OfflineRecognizer=None))
    f = tmp_path / "s.toml"
    f.write_text(f'[asr]\nmodel_dir = "{(tmp_path / "none").as_posix()}"\n', encoding="utf-8")
    rc = asr_worker.main(["--config", str(f), "--port", "0"],
                         run=lambda s: (_ for _ in ()).throw(AssertionError("served")))
    assert rc != 0 and "model_dir" in capsys.readouterr().err


class FakeRecognizer:
    """sherpa-onnx OfflineRecognizer の代わり。受け取った音声の長さを記録し、texts を順に返す。"""

    def __init__(self, texts):
        self.texts = list(texts)
        self.lengths = []

    def create_stream(self):
        stream = SimpleNamespace(result=SimpleNamespace(text=""))
        stream.accept_waveform = lambda rate, audio: self.lengths.append(len(audio))
        return stream

    def decode_stream(self, stream):
        stream.result.text = self.texts.pop(0)


def test_reazon_model_returns_whisper_shaped_segments():
    m = asr_worker.ReazonModel(FakeRecognizer(["こんにちは"]))
    segs, _ = m.transcribe(np.zeros(16000, np.float32), language="ja")
    segs = list(segs)
    assert [s.text for s in segs] == ["こんにちは"]
    assert segs[0].no_speech_prob == 0.0 and segs[0].compression_ratio == asr_worker.compression_ratio("こんにちは")
    assert not asr_worker.is_hallucination(segs[0].no_speech_prob, segs[0].compression_ratio)


def test_reazon_model_empty_text_gives_no_segment_and_discardable_res():
    shm = shared_memory.SharedMemory(create=True, size=640)
    try:
        res = asr_worker.make_transcribe(asr_worker.ReazonModel(FakeRecognizer([" "])))(4, shm.name)
    finally:
        shm.close()
        shm.unlink()
    assert res["text"] == "" and asr_worker.is_hallucination(res["no_speech_prob"], res["compression_ratio"])


@pytest.mark.parametrize("text,kept", [("え", False), ("あっ", False), ("何", False), ("えー？", False),
                                       ("うん", True), ("へえ", True), ("え、本当？", True), ("待って", True)])
def test_reazon_model_drops_single_mora_noise(text, kept):
    segs, _ = asr_worker.ReazonModel(FakeRecognizer([text])).transcribe(np.zeros(16000, np.float32))
    assert [s.text for s in segs] == ([text] if kept else [])


def test_reazon_model_splits_long_audio():
    rec = FakeRecognizer(["前半", "後半"])
    n = asr_worker.REAZON_MAX_S * 16000
    segs, _ = asr_worker.ReazonModel(rec).transcribe(np.zeros(n + 100, np.float32))
    assert [s.text for s in segs] == ["前半", "後半"] and rec.lengths == [n, 100]


def test_compression_ratio_flags_repetition():
    assert asr_worker.compression_ratio("") == 0.0
    assert asr_worker.compression_ratio("今日は雨が降っていて少し寒いね") < 2.4
    assert asr_worker.compression_ratio("ありがとう" * 40) > 2.4


def test_load_asr_defaults_and_validation(tmp_path):
    f = tmp_path / "s.toml"
    assert load_asr(f) == Asr("reazon", "data/models/reazonspeech", 2)
    f.write_text('[asr]\nengine = "whisper"\nthreads = 4\n', encoding="utf-8")
    assert load_asr(f) == Asr("whisper", "data/models/reazonspeech", 4)
    for bad in ('engine = "vosk"', "threads = 0", 'threads = "2"', 'model_dir = ""', "foo = 1"):
        f.write_text(f"[asr]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_asr(f)


def test_port_from_settings_file_and_cli_override(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(asr_worker, "make_server",
                        lambda tr, host, port: seen.append(port) or SimpleNamespace(server_close=lambda: None))
    f = tmp_path / "s.toml"
    f.write_text("[ports]\nasr = 9191\n", encoding="utf-8")
    assert asr_worker.main(["--config", str(f)], load_model=lambda: object(), run=lambda s: None) == 0
    assert asr_worker.main(["--config", str(f), "--port", "9292"], load_model=lambda: object(),
                           run=lambda s: None) == 0
    assert seen == [9191, 9292]
