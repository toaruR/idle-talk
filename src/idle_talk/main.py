"""起動エントリ (設計書 12.2 / 12.4)。二重起動は終了コード 4。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import subprocess
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path

from .clients import AsrClient, LlmClient, TtsClient
from .config import Config
from .director import IdleTalkDirector
from .metrics import MetricsRecorder
from .portlock import acquire_or_exit
from .resource_sampler import SAMPLE_INTERVAL_MS, ResourceSampler
from .server import serve
from .store import Store
from .vad import EnergyVad

log = logging.getLogger("idle_talk.main")


def _asr_send(url: str):
    def send(req: dict, timeout: float) -> tuple[int, dict]:
        r = urllib.request.Request(url, data=json.dumps(req).encode("utf-8"),
                                   headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=timeout) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
    return send


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True, timeout=5)
        out = r.stdout.strip()
        return out if r.returncode == 0 and out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def build_director(data_dir: Path, args: argparse.Namespace, cfg: Config) -> IdleTalkDirector:
    store = Store(data_dir / "idle_talk.db", bool(cfg.DB_BACKUP))
    llm = LlmClient(args.llm_url, cfg)
    tts = TtsClient(args.tts_url, config=cfg)
    asr = AsrClient(_asr_send(args.asr_url), cfg)
    return IdleTalkDirector(store, llm.stream_chat, tts.synthesize, lambda: None, EnergyVad(), asr, cfg,
                            metrics=MetricsRecorder(store), app_commit=git_commit())


async def soak_loop(sampler: ResourceSampler, interval_ms: int = SAMPLE_INTERVAL_MS,
                    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    """interval_ms ごとに RSS/VRAM を events(kind='soak_sample') へ記録する (§11.1 常駐 G5)。

    計測失敗 (nvidia-smi 不在等) はログのみで継続する。キャンセルで終了。
    """
    while True:
        await sleep(interval_ms / 1000)
        try:
            await asyncio.to_thread(sampler.sample)
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, sqlite3.Error):  # 計測器 (nvidia-smi/psutil/DB) の失敗で常駐ループを止めない
            log.exception("soak sample failed")


async def _run(director: IdleTalkDirector, port: int) -> None:
    server, pinger = await serve(director.core, port)
    log.info("listening on 127.0.0.1:%d", port)
    sampler = ResourceSampler(director.store, clock=director.now_ms)
    soak = asyncio.get_running_loop().create_task(soak_loop(sampler))
    try:
        await server.serve_forever()
    finally:
        pinger.cancel()
        soak.cancel()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="idle-talk")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--pid-file", default=None)
    p.add_argument("--llm-url", default="http://127.0.0.1:8080")
    p.add_argument("--tts-url", default="http://127.0.0.1:10101")
    p.add_argument("--asr-url", default="http://127.0.0.1:8090")
    args = p.parse_args(argv)
    data_dir = Path(args.data_dir)
    lock = acquire_or_exit(args.pid_file or data_dir / "idle_talk.pid")  # 二重起動は SystemExit(4)
    director = None
    try:
        director = build_director(data_dir, args, Config())
        asyncio.run(_run(director, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        if director is not None:
            director.close()
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
