"""M0 計測ハーネス (設計書 §9/§10/§1.4/§11.1)。

実 llama-server / AivisSpeech / faster-whisper に対し run_m0 で測定し、
evaluate_m0 で合否 (違反キーのリスト) を判定する。
python -m idle_talk.m0_bench: 違反 0 件なら終了コード 0、1 件以上なら 1 (CLI フラグなし)。
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import wave
from collections.abc import Callable

from .clients import LlmClient, TtsClient
from .config import Config, load_ports, load_speaker

GIB = 1024 ** 3
LLM_VRAM_MAX_GIB = 9.0  # §1.4 LLM は GPU0 で 9GiB 以内
ASR_VRAM_MAX_GIB = 3.0  # §1.4 ASR 追加 3GiB 以内
APP_RAM_MAX_GIB = 8.0  # §1.4 アプリ全体で追加 RAM 8GiB 以内

REQUIRED_KEYS = (
    "llm_first_s", "llm_total_s", "llm_think_count",
    "tts_first_ms", "tts_ratio_p95", "asr_silence_outputs",
    "llm_vram_gib", "asr_vram_gib", "app_ram_gib",
)

PROMPT = [{"role": "user", "content": "こんにちは。一言で挨拶して。"}]
TTS_TEXTS = ("こんにちは。", "今日はいい天気ですね。", "ゆっくりお話ししましょう。",
             "最近どんなことがありましたか。", "それはとても楽しそうですね。")


def evaluate_m0(m: dict, cfg: Config | None = None) -> list[str]:
    """測定値 dict を判定し、違反の説明文リストを返す (空なら合格)。キー欠落は違反。"""
    cfg = cfg or Config()
    t = cfg.HTTP_TIMEOUT
    v: list[str] = []
    for k in REQUIRED_KEYS:
        if k not in m:
            v.append(f"missing:{k}")
    if v:
        return v
    if m["llm_think_count"] != 0:
        v.append(f"llm_think_count={m['llm_think_count']} (must be 0)")
    if m["llm_first_s"] > t.llm_first:
        v.append(f"llm_first_s={m['llm_first_s']:.2f} > {t.llm_first}")
    if m["llm_total_s"] > t.llm_total:
        v.append(f"llm_total_s={m['llm_total_s']:.2f} > {t.llm_total}")
    if not m["tts_ratio_p95"] < cfg.TTS_RATIO_MAX:  # 判定は「<」(§11.1)
        v.append(f"tts_ratio_p95={m['tts_ratio_p95']:.2f} !< {cfg.TTS_RATIO_MAX}")
    if m["asr_silence_outputs"] != 0:
        v.append(f"asr_silence_outputs={m['asr_silence_outputs']} (must be 0)")
    if m["llm_vram_gib"] > LLM_VRAM_MAX_GIB:
        v.append(f"llm_vram_gib={m['llm_vram_gib']:.2f} > {LLM_VRAM_MAX_GIB}")
    if m["asr_vram_gib"] > ASR_VRAM_MAX_GIB:
        v.append(f"asr_vram_gib={m['asr_vram_gib']:.2f} > {ASR_VRAM_MAX_GIB}")
    if m["app_ram_gib"] > APP_RAM_MAX_GIB:
        v.append(f"app_ram_gib={m['app_ram_gib']:.2f} > {APP_RAM_MAX_GIB}")
    return v


def wav_duration_s(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / float(w.getframerate())


def p95(values: list[float]) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(0.95 * len(s)))]


def gpu0_used_gib() -> float:
    """nvidia-smi で GPU0 の使用 VRAM(GiB)。失敗は例外 (実機必須)。"""
    out = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10).stdout
    return float(out.strip().splitlines()[0]) / 1024.0


def process_rss_gib() -> float:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            wintypes.HANDLE(k32.GetCurrentProcess()), ctypes.byref(pmc), pmc.cb)
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return pmc.WorkingSetSize / GIB
    with open("/proc/self/status", encoding="utf-8") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024 / GIB
    raise OSError("VmRSS not found")


def run_m0(llm_url: str | None = None, aivis_url: str | None = None, speaker: int | None = None,
           cfg: Config | None = None,
           llm: LlmClient | None = None, tts: TtsClient | None = None,
           transcribe: Callable[[bytes], list] | None = None,
           vram_probe: Callable[[], float] = gpu0_used_gib,
           ram_probe: Callable[[], float] = process_rss_gib,
           clock: Callable[[], float] = time.monotonic) -> dict:
    """実サービスに対し M0 測定を実行し測定値 dict を返す。接続不能は例外 (失敗)。"""
    cfg = cfg or Config()
    ports = load_ports()
    llm_url = llm_url or os.environ.get("LLAMA_URL", f"http://127.0.0.1:{ports.llm}")
    aivis_url = aivis_url or os.environ.get("AIVIS_URL", f"http://127.0.0.1:{ports.tts}")
    speaker = speaker if speaker is not None else int(os.environ.get("AIVIS_SPEAKER", load_speaker()))
    llm = llm or LlmClient(llm_url, cfg)
    tts = tts or TtsClient(aivis_url, speaker, cfg)
    if transcribe is None:
        from .asr_worker import transcribe_pcm

        def transcribe(pcm: bytes) -> list:
            return transcribe_pcm(pcm, cfg=cfg)

    ram0 = ram_probe()

    # LLM: 最初のトークン時間・全体時間・思考タグ
    t0 = clock()
    first = None
    parts: list[str] = []
    for tok in llm.stream_chat(PROMPT):
        if first is None:
            first = clock() - t0
        parts.append(tok)
    total = clock() - t0
    text = "".join(parts)
    if first is None:
        raise RuntimeError("llm returned no tokens")
    llm_vram = vram_probe()

    # TTS: 最初の1文の合成時間と 処理時間÷音声長 の p95
    ratios: list[float] = []
    tts_first_ms = None
    for s in TTS_TEXTS:
        ts = clock()
        wav = tts.synthesize(s)
        el = clock() - ts
        if tts_first_ms is None:
            tts_first_ms = el * 1000.0
        ratios.append(el / wav_duration_s(wav))

    # ASR: 無音入力で出力 0 件 (幻聴率) と追加 VRAM
    pcm = b"\x00\x00" * (16000 * cfg.E_PARAMS.silence_input_s)
    before = vram_probe()
    outs = transcribe(pcm)
    after = vram_probe()

    return {
        "llm_first_s": first,
        "llm_total_s": total,
        "llm_think_count": text.count("<think>"),
        "llm_text": text,
        "tts_first_ms": tts_first_ms,
        "tts_ratio_p95": p95(ratios),
        "asr_silence_outputs": len(outs),
        "llm_vram_gib": llm_vram,
        "asr_vram_gib": max(0.0, after - before),
        "app_ram_gib": max(0.0, ram_probe() - ram0),
    }


def main() -> int:
    m = run_m0()
    for k, val in m.items():
        print(f"{k}: {val}")
    violations = evaluate_m0(m)
    for x in violations:
        print("VIOLATION:", x)
    return 0 if not violations else 1


if __name__ == "__main__":
    sys.exit(main())
