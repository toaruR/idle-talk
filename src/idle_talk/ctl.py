"""起動・停止ラッパー: python -m idle_talk.ctl start|stop|restart|status|memories。

asr_worker と main をバックグラウンドで起動し、出力は data_dir の asr_worker.log / idle_talk.log に追記する。
二重起動は各プロセスが書く PID ファイル (idle_talk.pid / asr_worker.pid) の生存確認で拒否する。
停止は main に停止ファイルを置いて正常終了 (run_meta 記録・llama-server 停止) を待ち、
STOP_TIMEOUT_S 内に終わらなければ強制終了する。asr_worker は保存する状態が無いので即終了する。
memories は保存された記憶の一覧表示と削除 (--delete ID / --clear)。起動中でも使える (WAL)。
"""
from __future__ import annotations

import argparse
import os
import signal
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from .config import SETTINGS_FILE, load_ports
from .errors import EXIT_DUPLICATE_LAUNCH
from .main import DB_FILE, PID_FILE, STOP_FILE
from .portlock import pid_alive

ASR_PID_FILE = "asr_worker.pid"
START_CHECK_S = 5.0  # 起動直後の失敗 (import エラー・ポート使用中など) を待つ時間
STOP_TIMEOUT_S = 20.0  # 正常終了の待ち上限。超えたら強制終了
POLL_S = 0.2


def read_pid(path: Path) -> int | None:
    """PID ファイルの PID が生存していれば返す。無い・壊れている・死んでいれば None。"""
    try:
        pid = int(path.read_text().strip())
    except (ValueError, OSError):
        return None
    return pid if pid_alive(pid) else None


def wait_until(cond: Callable[[], bool], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(POLL_S)
    return cond()


def spawn(module: str, args: list[str], log_path: Path) -> subprocess.Popen:
    """端末から切り離して python -m module を起動する (端末を閉じても・Ctrl+C でも止まらない)。"""
    kw: dict = {}
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kw["start_new_session"] = True
    with log_path.open("ab") as out:
        return subprocess.Popen([sys.executable, "-m", module, *args], stdout=out, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, **kw)


def start_one(name: str, module: str, args: list[str], pid_file: Path, log_path: Path) -> int | None:
    """起動して PID ファイルが書かれるのを待つ。起動直後に終了したら None。"""
    proc = spawn(module, args, log_path)
    if wait_until(lambda: read_pid(pid_file) is not None or proc.poll() is not None, START_CHECK_S) \
            and proc.poll() is not None:
        print(f"{name}: failed to start (exit {proc.returncode}). see {log_path}")
        return None
    pid = read_pid(pid_file)
    print(f"{name}: started (pid {pid or proc.pid})")
    return pid or proc.pid


def kill(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)  # Windows では TerminateProcess
    except OSError:
        pass


def cmd_start(args: argparse.Namespace, data_dir: Path) -> int:
    pid = read_pid(data_dir / PID_FILE)
    if pid is not None:
        print(f"idle-talk: already running (pid {pid})")
        return EXIT_DUPLICATE_LAUNCH
    data_dir.mkdir(parents=True, exist_ok=True)
    common = ["--config", str(args.config)]
    asr_pid = read_pid(data_dir / ASR_PID_FILE)
    if asr_pid is not None:
        print(f"asr_worker: already running (pid {asr_pid})")
    elif start_one("asr_worker", "idle_talk.asr_worker", [*common, "--pid-file", str(data_dir / ASR_PID_FILE)],
                   data_dir / ASR_PID_FILE, data_dir / "asr_worker.log") is None:
        return 1
    if start_one("idle-talk", "idle_talk.main", [*common, "--data-dir", str(data_dir)],
                 data_dir / PID_FILE, data_dir / "idle_talk.log") is None:
        return 1
    print(f"open http://127.0.0.1:{load_ports(args.config).web}/")
    return 0


def cmd_stop(args: argparse.Namespace, data_dir: Path) -> int:
    pid = read_pid(data_dir / PID_FILE)
    stop_file = data_dir / STOP_FILE
    if pid is None:
        print("idle-talk: not running")
    else:
        stop_file.write_text("")
        if wait_until(lambda: not pid_alive(pid), STOP_TIMEOUT_S):
            print(f"idle-talk: stopped (pid {pid})")
        else:
            kill(pid)
            print(f"idle-talk: killed after {STOP_TIMEOUT_S:.0f}s (pid {pid}); llama-server may be left running")
    stop_file.unlink(missing_ok=True)
    asr_pid = read_pid(data_dir / ASR_PID_FILE)
    if asr_pid is None:
        print("asr_worker: not running")
    else:
        kill(asr_pid)
        wait_until(lambda: not pid_alive(asr_pid), STOP_TIMEOUT_S)
        (data_dir / ASR_PID_FILE).unlink(missing_ok=True)
        print(f"asr_worker: stopped (pid {asr_pid})")
    return 0


def cmd_status(args: argparse.Namespace, data_dir: Path) -> int:
    pid = read_pid(data_dir / PID_FILE)
    asr_pid = read_pid(data_dir / ASR_PID_FILE)
    print(f"idle-talk: {f'running (pid {pid})' if pid else 'stopped'}")
    print(f"asr_worker: {f'running (pid {asr_pid})' if asr_pid else 'stopped'}")
    if pid:
        print(f"open http://127.0.0.1:{load_ports(args.config).web}/")
    return 0 if pid else 3


def cmd_memories(args: argparse.Namespace, data_dir: Path) -> int:
    """記憶の確認・削除 (設計書 3.4: 利用者が確認・削除できる)。削除できるのは deletable=1 の行だけ。"""
    db = data_dir / DB_FILE
    if not db.exists():
        print(f"no database: {db}")
        return 1
    con = sqlite3.connect(str(db), isolation_level=None)
    try:
        if args.clear:
            n = con.execute("DELETE FROM memories WHERE deletable=1").rowcount
            print(f"deleted {n} memories")
        elif args.delete is not None:
            n = con.execute("DELETE FROM memories WHERE id=? AND deletable=1", (args.delete,)).rowcount
            print(f"deleted memory {args.delete}" if n else f"memory {args.delete}: not found or not deletable")
            return 0 if n else 1
        else:
            rows = con.execute("SELECT id, content, deletable FROM memories ORDER BY id").fetchall()
            for mid, content, deletable in rows:
                print(f"{mid:4d}{'' if deletable else ' (固定)'}  {content}")
            if not rows:
                print("no memories")
    finally:
        con.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="idle-talk-ctl")
    p.add_argument("command", choices=("start", "stop", "restart", "status", "memories"))
    p.add_argument("--config", default=SETTINGS_FILE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--delete", type=int, metavar="ID", help="memories: 指定 id の記憶を削除")
    p.add_argument("--clear", action="store_true", help="memories: 削除可能な記憶をすべて削除")
    args = p.parse_args(argv)
    args.config = Path(args.config).resolve()
    data_dir = Path(args.data_dir).resolve()
    if args.command == "restart":
        cmd_stop(args, data_dir)
        return cmd_start(args, data_dir)
    return {"start": cmd_start, "stop": cmd_stop, "status": cmd_status,
            "memories": cmd_memories}[args.command](args, data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
