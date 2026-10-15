"""llama-server の起動コマンドと再起動監督 (設計書 3.3 / 12.4, E7)。時計・sleep・起動は注入可能。"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path

from .config import Config, LlmLaunch

log = logging.getLogger("idle_talk.llm_supervisor")

FALLBACK_PHRASE = "ちょっと考え込んじゃいました。もう一度お願いします。"
NO_THINK_KWARGS = '{"enable_thinking":false}'


def build_llama_command(model_path: str, port: int = 8080, ngl: int = 99) -> str:
    """思考無効のテンプレート設定 (--jinja + enable_thinking=false) を含む起動コマンド文字列。"""
    return (f"llama-server -m {model_path} --port {port} -ngl {ngl} "
            f"--jinja --chat-template-kwargs '{NO_THINK_KWARGS}'")


def build_llama_argv(launch: LlmLaunch, port: int) -> list[str]:
    """build_llama_command と同じ起動内容を、シェルを介さない Popen 用の引数リストで返す。

    main_gpu >= 0 なら分割せずその GPU だけに載せる (他の GPU は既存アプリ優先、設計書 1.4)。
    mmproj があれば画像入力を有効にする (ドラマモードの画面、設計書 6.5)。"""
    gpu = ["-sm", "none", "-mg", str(launch.main_gpu)] if launch.main_gpu >= 0 else []
    vision = ["--mmproj", launch.mmproj] if launch.mmproj else []
    if launch.mmproj and launch.image_max_tokens > 0:
        vision += ["--image-max-tokens", str(launch.image_max_tokens)]
    return [launch.server, "-m", launch.model, "--host", "127.0.0.1", "--port", str(port),
            "-ngl", str(launch.ngl), "-c", str(launch.ctx), *gpu, *vision,
            "--jinja", "--chat-template-kwargs", NO_THINK_KWARGS]


def llama_env() -> dict[str, str]:
    """GPU 番号を nvidia-smi と同じ PCI 順にそろえた環境変数 (CUDA の既定は速い順で食い違いうる)。"""
    return {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def popen_starter(argv: list[str], log_path: Path) -> Callable[[], subprocess.Popen]:
    """起動ごとに log_path へ追記する Popen を返す関数。"""
    def start() -> subprocess.Popen:
        log.info("starting llama-server: %s", argv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as out:
            return subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    env=llama_env())
    return start


def is_ready(base_url: str, timeout: float = 1.0) -> bool:
    """llama-server の /health が 200 (モデル読込完了) なら True。読込中 (503) や接続不能は False。"""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def has_vision(base_url: str, timeout: float = 2.0) -> bool:
    """llama-server が画像入力に対応しているか (/props の modalities.vision。mmproj 付きで起動したとき True)。"""
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/props", timeout=timeout) as r:
            props = json.load(r)
    except (OSError, ValueError):
        return False
    return isinstance(props, dict) and bool((props.get("modalities") or {}).get("vision"))


def wait_ready(base_url: str, timeout_s: float, interval_s: float = 0.5,
               clock: Callable[[], float] = time.monotonic,
               sleep: Callable[[float], None] = time.sleep,
               ready: Callable[[str], bool] = is_ready) -> bool:
    deadline = clock() + timeout_s
    while clock() < deadline:
        if ready(base_url):
            return True
        sleep(interval_s)
    return False


class LlmSupervisor:
    """異常終了ごとに on_crash() を呼ぶ。LLM_RETRY: バックオフ 1s→2s→4s、window 内 max 回まで再起動。
    max+1 回目 (window 内 4 回目) は再起動せず rest 要求 + warn。window 超過後は回数リセット。"""

    def __init__(self, start: Callable[[], object], cfg: Config | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 request_rest: Callable[[], None] = lambda: None,
                 warn: Callable[[str], None] = lambda m: None) -> None:
        self._start = start
        self.cfg = cfg or Config()
        self.clock = clock
        self._sleep = sleep
        self._request_rest = request_rest
        self._warn = warn
        self._crashes: list[float] = []
        self.proc: object | None = None

    def start(self) -> None:
        self.proc = self._start()

    def exited(self) -> bool:
        """監督中のプロセスが終了していれば True。未起動 (proc なし) は False。"""
        return self.proc is not None and self.proc.poll() is not None

    def stop(self) -> None:
        """Director 終了時に監督中のプロセスを止める。"""
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def on_crash(self) -> str:
        """'restarted' | 'rest'。"""
        r = self.cfg.LLM_RETRY
        now = self.clock()
        self._crashes = [t for t in self._crashes if now - t <= r.window_s]
        self._crashes.append(now)
        n = len(self._crashes)
        if n > r.max:
            self._warn(f"llama-server crashed {n} times within {r.window_s}s: requesting rest")
            self._request_rest()
            return "rest"
        self._sleep(r.backoff_s[min(n - 1, len(r.backoff_s) - 1)])
        self.start()
        return "restarted"


WATCH_INTERVAL_S = 0.5  # プロセス終了の検知間隔 (初回バックオフ 1s より短くする)


async def watch(sup: LlmSupervisor, interval_s: float = WATCH_INTERVAL_S,
                sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> str:
    """プロセス終了を検知するたびに on_crash() で再起動する。'rest' になったら監督を終えて 'rest' を返す。"""
    while True:
        await sleep(interval_s)
        if not sup.exited():
            continue
        log.warning("llama-server exited (code %s)", sup.proc.poll())
        result = await asyncio.to_thread(sup.on_crash)  # バックオフの sleep でループを止めない
        if result == "rest":
            return result
