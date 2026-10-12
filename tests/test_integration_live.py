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
from types import SimpleNamespace

import pytest

from idle_talk.clause import ClauseSplitter
from idle_talk.clients import LlmClient, TtsClient
from idle_talk.config import Config, load_llm_launch, load_ports, load_speaker
from idle_talk.epoch import Epoch
from idle_talk.llm_supervisor import (
    FALLBACK_PHRASE,
    WATCH_INTERVAL_S,
    LlmSupervisor,
    build_llama_argv,
    popen_starter,
    wait_ready,
    watch,
)
from idle_talk.main import LLM_READY_TIMEOUT_S
from idle_talk.pipeline import ResponsePipeline

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
    """/synthesis は切断後も計算が続くため、残る計算は合成中の 1 文ぶん (tts_sentence 以内) に限られることを見る。"""
    proc = _listener_process(_port(AIVIS))
    idle = _tree_cpu(proc, 1.0)
    text = "これは計算停止を確認するための、句読点まで切れ目のない少し長めの一文です。" * 3
    q = urllib.request.urlopen(urllib.request.Request(
        f"{AIVIS}/audio_query?text={urllib.parse.quote(text)}&speaker={SPEAKER}", method="POST"),
        timeout=CFG.HTTP_TIMEOUT.tts_sentence * 4).read()
    host = urllib.parse.urlparse(AIVIS)
    conn = http.client.HTTPConnection(host.hostname, host.port, timeout=30)
    conn.request("POST", f"/synthesis?speaker={SPEAKER}", body=q,
                 headers={"Content-Type": "application/json"})
    th = threading.Thread(target=lambda: conn.getresponse(), daemon=True)
    th.start()
    busy = _tree_cpu(proc, 0.2)
    assert busy > idle, "synthesis did not load the engine"
    conn.close()  # cancel: 接続を閉じる (計算は 1 文ぶん続く)
    wait = CFG.HTTP_TIMEOUT.tts_sentence + CFG.E_PARAMS.tts_cpu_settle_s
    time.sleep(wait - 0.5)
    after = _tree_cpu(proc, 0.5)  # cancel 後 tts_sentence + settle 秒の時点
    assert after <= idle + CFG.E_PARAMS.tts_cpu_delta_pt, (idle, busy, after)


def test_e7_real_llama_kill_gives_fallback_then_restart_answers_next_turn(tmp_path):
    """kill → llm_first 内に定型句 → main と同じ監督 (watch + LlmSupervisor) が初回バックオフで再起動 → 次ターンで応答。
    [llm].model の設定が必要。再起動した llama-server は試験後も動かしたままにする。"""
    launch = load_llm_launch()
    assert launch.model, "[llm].model in idle_talk.toml is required for T-E7"
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

    killed = SimpleNamespace(poll=lambda: -9)  # kill 済みプロセスを監督中として渡す
    sup = LlmSupervisor(popen_starter(build_llama_argv(launch, _port(LLAMA)), tmp_path / "llama.log"), CFG)
    sup.proc = killed

    async def restart() -> float:
        task = asyncio.get_running_loop().create_task(watch(sup))
        t1 = time.monotonic()
        while sup.proc is killed:
            await asyncio.sleep(0.05)
        task.cancel()
        return time.monotonic() - t1
    restart_s = asyncio.run(restart())
    assert restart_s <= CFG.LLM_RETRY.backoff_s[0] + 2 * WATCH_INTERVAL_S, restart_s  # 初回バックオフ内
    assert wait_ready(LLAMA, LLM_READY_TIMEOUT_S), "restarted llama-server not ready"

    emitted.clear()
    audio.clear()
    asyncio.run(pipe.respond(MSGS))  # 次ターン
    assert audio and audio[0][1] != FALLBACK_PHRASE, audio
    assert not any(m.get("t") == "error" for m in emitted), emitted
