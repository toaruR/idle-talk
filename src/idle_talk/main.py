"""起動エントリ (設計書 12.2 / 12.4)。二重起動は終了コード 4。"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import subprocess
import threading
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .clients import AsrClient, LlmClient, TtsClient
from .config import (SETTINGS_FILE, Config, load_llm_launch, load_ports, load_speaker, load_spontaneous,
                     load_voice)
from .director import IdleTalkDirector
from .llm_supervisor import LlmSupervisor, build_llama_argv, is_ready, popen_starter, wait_ready, watch
from .metrics import MetricsRecorder
from .portlock import acquire_or_exit
from .resource_sampler import SAMPLE_INTERVAL_MS, ResourceSampler
from .server import serve
from .store import Store
from .vad import EnergyVad

log = logging.getLogger("idle_talk.main")

LLM_READY_TIMEOUT_S = 120  # 起動時のモデル読込待ち上限 (4B Q4 の実測は約 5s)
SPONT_TICK_S = 1.0  # 自発発話の判定周期 (沈黙秒数の分解能)
STOP_POLL_S = 0.5  # 停止ファイル (ctl stop) の確認周期
PID_FILE = "idle_talk.pid"  # data_dir 直下。二重起動検査と ctl の停止対象
STOP_FILE = "idle_talk.stop"  # data_dir 直下。置かれたら正常終了する (ctl stop)
WEB_DIR = Path(__file__).resolve().parents[2] / "web"  # src レイアウト前提 (リポジトリ直下の web/)


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


class TtsSettingsReloader:
    """設定ファイルの更新時刻が変わったら [tts] (話者・声パラメータ) を読み直して tts に反映する。

    合成のたびに呼ぶので、ファイルを保存すれば再起動なしで次の発話から効く。
    不正な値は警告して前の値を使い続ける。--speaker 指定時は話者を上書きしない。"""

    def __init__(self, tts: TtsClient, path: str | Path, fixed_speaker: bool = False) -> None:
        self.tts = tts
        self.path = Path(path)
        self.fixed_speaker = fixed_speaker
        self._mtime = self._stat()

    def _stat(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def reload_if_changed(self) -> bool:
        mtime = self._stat()
        if mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            speaker = self.tts.speaker if self.fixed_speaker else load_speaker(self.path)
            voice = load_voice(self.path)
        except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
            log.warning("tts settings not reloaded (keeping previous): %s", e)
            return False
        self.tts.speaker, self.tts.voice = speaker, voice
        log.info("tts settings reloaded: speaker=%d voice=%s", speaker, voice)
        return True

    def synthesize(self, text: str) -> bytes:
        self.reload_if_changed()
        return self.tts.synthesize(text)


def build_director(data_dir: Path, args: argparse.Namespace, cfg: Config) -> IdleTalkDirector:
    store = Store(data_dir / "idle_talk.db", bool(cfg.DB_BACKUP))
    llm = LlmClient(args.llm_url, cfg)
    tts = TtsClient(args.tts_url, args.speaker, cfg, voice=load_voice(args.config))
    if not tts.warmup():
        log.warning("tts warmup failed: %s", args.tts_url)
    asr = AsrClient(_asr_send(args.asr_url), cfg)
    reloader = TtsSettingsReloader(tts, args.config, fixed_speaker=args.speaker_from_cli)
    return IdleTalkDirector(store, llm.stream_chat, reloader.synthesize, lambda: None, EnergyVad(), asr, cfg,
                            metrics=MetricsRecorder(store), app_commit=git_commit())


def ui_defaults(cfg: Config) -> dict:
    """画面の設定欄の初期値 (ui メッセージと同じキー)。idle_talk.toml の [spontaneous] を反映した cfg から作る。"""
    return {"on": cfg.spontaneous_enabled, "interval_s": cfg.spont_interval_s,
            "quiet_start": cfg.quiet_start, "quiet_end": cfg.quiet_end}


def start_web(port: int, ws_port: int, ui: dict | None = None, root: Path = WEB_DIR) -> ThreadingHTTPServer:
    """web/ を 127.0.0.1:port で配信する。/config.json で WS ポートと設定欄の初期値をブラウザへ渡す。

    file:// だと Origin が null で WS に拒否されるため、ブラウザはこの URL から開く。"""
    config = json.dumps({"ws_port": ws_port, "ui": ui or {}}).encode("utf-8")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_GET(self):
            if self.path == "/config.json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(config)))
                self.end_headers()
                self.wfile.write(config)
                return
            super().do_GET()

        def end_headers(self):  # 毎回再検証させる (JS 更新後に古いキャッシュで動かないように)
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, format, *args):  # アクセスログは出さない
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=httpd.serve_forever, name="web", daemon=True).start()
    log.info("web ui: http://127.0.0.1:%d/", httpd.server_address[1])
    return httpd


def start_llm(args: argparse.Namespace, data_dir: Path, cfg: Config) -> LlmSupervisor | None:
    """[llm].model があれば llama-server を起動して監督する。無ければ手動起動扱いで None。

    既に同じポートで応答している llama-server がある場合は二重起動せず、監督もしない。"""
    launch = load_llm_launch(args.config)
    if not launch.model:
        return None
    if is_ready(args.llm_url):
        log.warning("llama-server already running at %s: not supervised", args.llm_url)
        return None
    argv = build_llama_argv(launch, urllib.parse.urlparse(args.llm_url).port)
    sup = LlmSupervisor(popen_starter(argv, data_dir / "llama-server.log"), cfg, warn=log.warning)
    sup.start()
    if not wait_ready(args.llm_url, LLM_READY_TIMEOUT_S):
        log.warning("llama-server not ready within %ds: %s", LLM_READY_TIMEOUT_S, args.llm_url)
    return sup


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


async def llm_watch(director: IdleTalkDirector, sup: LlmSupervisor) -> None:
    """llama-server の異常終了を LLM_RETRY に従って再起動し、上限超過で休止へ移す (E7)。"""
    if await watch(sup) == "rest":
        director.request_rest()


async def spontaneous_loop(director: IdleTalkDirector, interval_s: float = SPONT_TICK_S,
                           sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    """interval_s ごとに自発発話と無操作休止 (IDLE_TIMEOUT) を判定する。キャンセルで終了。"""
    while True:
        await sleep(interval_s)
        director.spontaneous_tick()


async def stop_watch(path: Path, stop: Callable[[], None], interval_s: float = STOP_POLL_S,
                     sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
    """path が現れたら削除して stop を呼ぶ (ctl stop からの正常終了要求)。キャンセルで終了。"""
    while True:
        await sleep(interval_s)
        if path.exists():
            path.unlink(missing_ok=True)
            log.info("stop requested: %s", path)
            stop()
            return


async def _run(director: IdleTalkDirector, port: int, sup: LlmSupervisor | None = None,
               stop_file: Path | None = None) -> None:
    server, pinger = await serve(director.core, port)
    log.info("listening on 127.0.0.1:%d", port)
    sampler = ResourceSampler(director.store, clock=director.now_ms)
    loop = asyncio.get_running_loop()
    stopped = asyncio.Event()
    tasks = [pinger, loop.create_task(soak_loop(sampler)), loop.create_task(spontaneous_loop(director))]
    if sup is not None:
        tasks.append(loop.create_task(llm_watch(director, sup)))
    if stop_file is not None:
        tasks.append(loop.create_task(stop_watch(stop_file, stopped.set)))
    try:
        await stopped.wait()  # Ctrl+C はキャンセルで抜ける
    finally:
        for t in tasks:
            t.cancel()
        server.close()
        await server.wait_closed()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="idle-talk")
    p.add_argument("--config", default=SETTINGS_FILE)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--web-port", type=int, default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--pid-file", default=None)
    p.add_argument("--llm-url", default=None)
    p.add_argument("--tts-url", default=None)
    p.add_argument("--asr-url", default=None)
    p.add_argument("--speaker", type=int, default=None)
    args = p.parse_args(argv)
    ports = load_ports(args.config)  # CLI 指定が設定ファイルより優先
    args.port = ports.ws if args.port is None else args.port
    args.web_port = ports.web if args.web_port is None else args.web_port
    args.llm_url = args.llm_url or f"http://127.0.0.1:{ports.llm}"
    args.tts_url = args.tts_url or f"http://127.0.0.1:{ports.tts}"
    args.asr_url = args.asr_url or f"http://127.0.0.1:{ports.asr}"
    args.speaker_from_cli = args.speaker is not None
    args.speaker = load_speaker(args.config) if args.speaker is None else args.speaker
    data_dir = Path(args.data_dir)
    lock = acquire_or_exit(args.pid_file or data_dir / PID_FILE)  # 二重起動は SystemExit(4)
    stop_file = data_dir / STOP_FILE
    stop_file.unlink(missing_ok=True)  # 前回の停止要求の残りで即終了しない
    director = sup = web = None
    try:
        cfg = Config(**load_spontaneous(args.config))
        web = start_web(args.web_port, args.port, ui_defaults(cfg))
        sup = start_llm(args, data_dir, cfg)
        director = build_director(data_dir, args, cfg)
        asyncio.run(_run(director, args.port, sup, stop_file))
    except KeyboardInterrupt:
        pass
    finally:
        if director is not None:
            director.close()
        if sup is not None:
            sup.stop()
        if web is not None:
            web.shutdown()
            web.server_close()
        lock.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
