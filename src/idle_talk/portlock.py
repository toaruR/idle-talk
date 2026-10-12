"""二重起動防止 PID ファイル (設計書 12.4: 起動時に生存確認、reject)。"""
from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from .errors import EXIT_DUPLICATE_LAUNCH

log = logging.getLogger("idle_talk.portlock")


class DuplicateLaunch(Exception):
    """生存中の Director が既にいる。"""


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        h = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(h))
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PidLock:
    def __init__(self, path: str | Path, pid: int | None = None,
                 is_alive: Callable[[int], bool] = pid_alive) -> None:
        self.path = Path(path)
        self.pid = os.getpid() if pid is None else pid
        self._is_alive = is_alive

    def acquire(self) -> None:
        """生存 PID が残っていれば DuplicateLaunch。死んだ PID は回収して取得する。"""
        if self.path.exists():
            try:
                old = int(self.path.read_text().strip())
            except (ValueError, OSError):
                old = 0
            if old and self._is_alive(old):
                log.error("duplicate launch rejected: pid %d alive", old)
                raise DuplicateLaunch(old)
            log.info("stale pid file reclaimed (pid %s)", old)
            self.path.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(self.pid))

    def release(self) -> None:
        try:
            if self.path.exists() and int(self.path.read_text().strip()) == self.pid:
                self.path.unlink()
        except (ValueError, OSError):
            pass


def acquire_or_exit(path: str | Path, **kw) -> PidLock:
    """取得できなければ終了コード 4 で終了する。"""
    lock = PidLock(path, **kw)
    try:
        lock.acquire()
    except DuplicateLaunch:
        sys.exit(EXIT_DUPLICATE_LAUNCH)
    return lock
