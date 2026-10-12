"""llama-server の起動コマンドと再起動監督 (設計書 3.3 / 12.4, E7)。時計・sleep・起動は注入可能。"""
from __future__ import annotations

import time
from collections.abc import Callable

from .config import Config

FALLBACK_PHRASE = "ちょっと考え込んじゃいました。もう一度お願いします。"
NO_THINK_KWARGS = '{"enable_thinking":false}'


def build_llama_command(model_path: str, port: int = 8080, ngl: int = 99) -> str:
    """思考無効のテンプレート設定 (--jinja + enable_thinking=false) を含む起動コマンド文字列。"""
    return (f"llama-server -m {model_path} --port {port} -ngl {ngl} "
            f"--jinja --chat-template-kwargs '{NO_THINK_KWARGS}'")


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
