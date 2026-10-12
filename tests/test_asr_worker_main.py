"""asr_worker.main の試験。faster_whisper は import せず、fake モデル/注入で検証する。"""
import json
import sys
import threading
import urllib.request
from multiprocessing import shared_memory
from types import SimpleNamespace

from idle_talk import asr_worker


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


def test_missing_faster_whisper_exits_nonzero_with_message(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # import を ImportError にする
    rc = asr_worker.main(["--port", "0"], run=lambda s: (_ for _ in ()).throw(AssertionError("served")))
    assert rc != 0
    assert "faster_whisper is not installed" in capsys.readouterr().err


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
