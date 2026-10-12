"""実サービス統合試験 (-m integration)。実 llama-server + 実 AivisSpeech (ASR は模擬)。
接続不能・プロセス未検出は失敗として扱う。psutil が必要 (pip install psutil)。
T-E7 は実 llama-server を kill するため最後に実行する (定義順)。"""
import asyncio
import http.client
import os
import statistics
import threading
import time
import urllib.parse
import urllib.request

import pytest

from idle_talk.clause import ClauseSplitter
from idle_talk.clients import LlmClient, TtsClient
from idle_talk.config import Config
from idle_talk.epoch import Epoch
from idle_talk.llm_supervisor import FALLBACK_PHRASE
from idle_talk.pipeline import ResponsePipeline

pytestmark = pytest.mark.integration

CFG = Config()
LLAMA = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
AIVIS = os.environ.get("AIVIS_URL", "http://127.0.0.1:10101")
SPEAKER = int(os.environ.get("AIVIS_SPEAKER", "888753760"))
MSGS = [{"role": "user", "content": "こんにちは。今日の天気について一言で答えて。"}]
LLM_BUDGET_MS = 300  # §8.1 VAR_SUM の内訳
TTS_BUDGET_MS = 250
FIRST_AUDIO_P50_MS = LLM_BUDGET_MS + TTS_BUDGET_MS  # 550


def _port(url: str) -> int:
    return urllib.parse.urlparse(url).port


def _listener_process(port: int):
    import psutil

    for c in psutil.net_connections(kind="tcp"):
        if c.status == psutil.CONN_LISTEN and c.laddr.port == port and c.pid:
            return psutil.Process(c.pid)
    raise AssertionError(f"no process listening on port {port}")


def _tree_cpu(proc, interval: float) -> float:
    procs = [proc] + proc.children(recursive=True)
    for p in procs:
        p.cpu_percent(None)
    time.sleep(interval)
    return sum(p.cpu_percent(None) for p in procs)


def test_first_clause_to_first_tts_audio_p50():
    tts = TtsClient(AIVIS, SPEAKER, CFG)
    llm = LlmClient(LLAMA, CFG)
    samples = []
    for _ in range(CFG.TEST_N):
        sp = ClauseSplitter()
        clause = None
        for tok in llm.stream_chat(MSGS):
            got = sp.feed(tok)
            if got:
                clause = got[0]
                break
        if clause is None:
            rest = sp.flush()
            assert rest, "no synthesizable clause from llm"
            clause = rest[0]
        t0 = time.monotonic()  # 最初の合成可能な節の確定
        wav = tts.synthesize(clause)
        samples.append((time.monotonic() - t0) * 1000)  # TTS 最初の音声
        assert wav[:4] == b"RIFF"
    assert statistics.median(samples) <= FIRST_AUDIO_P50_MS, samples


def test_e8_tts_cpu_returns_to_idle_after_cancel():
    proc = _listener_process(_port(AIVIS))
    idle = _tree_cpu(proc, 1.0)
    text = "これは計算停止を確認するための長い文章です。" * 20
    q = urllib.request.urlopen(urllib.request.Request(
        f"{AIVIS}/audio_query?text={urllib.parse.quote(text)}&speaker={SPEAKER}", method="POST"),
        timeout=CFG.HTTP_TIMEOUT.tts_sentence * 4).read()
    host = urllib.parse.urlparse(AIVIS)
    conn = http.client.HTTPConnection(host.hostname, host.port, timeout=30)
    conn.request("POST", f"/cancellable_synthesis?speaker={SPEAKER}", body=q,
                 headers={"Content-Type": "application/json"})
    th = threading.Thread(target=lambda: conn.getresponse(), daemon=True)
    th.start()
    time.sleep(1.0)
    busy = _tree_cpu(proc, 0.5)
    assert busy > idle, "synthesis did not load the engine"
    conn.close()  # cancel: 接続を閉じて合成を打ち切る
    settle = CFG.E_PARAMS.tts_cpu_settle_s
    time.sleep(settle - 0.5)
    after = _tree_cpu(proc, 0.5)  # cancel 後 settle 秒の時点
    assert after <= idle + CFG.E_PARAMS.tts_cpu_delta_pt, (idle, busy, after)


def test_e7_real_llama_kill_gives_fallback_within_llm_first():
    proc = _listener_process(_port(LLAMA))
    emitted, audio = [], []
    pipe = ResponsePipeline(LlmClient(LLAMA, CFG).stream_chat, lambda t: b"RIFF....WAVE", lambda: None,
                            lambda ep, utt, ch, text, wav: audio.append((time.monotonic(), text)),
                            emitted.append, Epoch())
    proc.kill()
    t0 = time.monotonic()
    asyncio.run(pipe.respond(MSGS))
    assert audio and audio[0][1] == FALLBACK_PHRASE
    assert audio[0][0] - t0 <= CFG.HTTP_TIMEOUT.llm_first
    assert any(m.get("t") == "error" and m.get("code") == 5001 for m in emitted)
