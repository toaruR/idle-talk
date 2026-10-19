"""WebSocket サーバ (設計書 3.1 / 7 / 12.4)。

接続制御ロジック (WsCore) は転送層から独立しており、クロック注入で試験できる。
serve() が websockets ライブラリへの薄い接続層。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable
from typing import Protocol
from urllib.parse import urlparse

from websockets.exceptions import ConnectionClosed

from .config import Config
from .epoch import Epoch
from .errors import (
    CLOSE_4001,
    CLOSE_ORIGIN_DENIED,
    CLOSE_VERSION_MISMATCH,
    ERR_OUT_OF_RANGE,
)
from .protocol import MAX_CHUNK_PCM, unpack_up_frame, validate_message

log = logging.getLogger("idle_talk.server")

PROTOCOL_VERSION = 1
ALLOWED_HOSTS = ("localhost", "127.0.0.1", "::1", "[::1]")
BIND_HOST = "127.0.0.1"
MISSED_PINGS_LIMIT = 2
UI_FIELDS = ("mute", "on", "radio", "interval_s", "quiet_start", "quiet_end", "speaker", "drama", "drama_mic", "screen", "pc", "persona")


class Conn(Protocol):
    def send_text(self, s: str) -> None: ...
    def send_bytes(self, b: bytes) -> None: ...
    def close(self, code: int) -> None: ...


def check_bind_host(host: str) -> str:
    if host != BIND_HOST:
        raise ValueError(f"bind address must be {BIND_HOST} only: {host}")
    return host


def origin_allowed(origin: str | None) -> bool:
    if not origin:
        return False
    try:
        return urlparse(origin).hostname in ("localhost", "127.0.0.1", "::1")
    except ValueError:
        return False


class WsCore:
    def __init__(self, cfg: Config | None = None, epoch: Epoch | None = None,
                 clock: Callable[[], float] = time.monotonic, host: str = BIND_HOST,
                 on_audio: Callable[[int, bytes], None] | None = None,
                 on_playback: Callable[[dict], None] | None = None,
                 on_ui: Callable[[dict], None] | None = None,
                 on_hello: Callable[[], None] | None = None,
                 on_snap: Callable[[dict], None] | None = None,
                 on_disconnect: Callable[[], None] | None = None) -> None:
        self.host = check_bind_host(host)
        self.cfg = cfg or Config()
        self.epoch = epoch or Epoch()
        self.clock = clock
        self.on_audio = on_audio
        self.on_playback = on_playback
        self.on_ui = on_ui
        self.on_hello = on_hello
        self.on_snap = on_snap
        self.on_disconnect = on_disconnect
        self.settings: dict = {"mute": False, "on": self.cfg.spontaneous_enabled, "radio": self.cfg.radio,
                               "interval_s": self.cfg.spont_interval_s,
                               "quiet_start": self.cfg.quiet_start,
                               "quiet_end": self.cfg.quiet_end}
        self.conn: Conn | None = None
        self.hello_done = False
        self._ever_connected = False
        self._last_ping = 0.0
        self._unanswered = False
        self._missed = 0

    # --- 接続制御 ---
    def connect(self, conn: Conn, origin: str | None) -> bool:
        if not origin_allowed(origin):
            log.warning("origin rejected: %r", origin)
            conn.close(CLOSE_ORIGIN_DENIED)
            return False
        if self.conn is not None:
            log.info("replacing existing connection (latest wins)")
            old = self.conn
            self.disconnect(old)
            old.close(CLOSE_4001)
        self.conn = conn
        self.hello_done = False
        if self._ever_connected:
            new = self.epoch.bump()
            self._send(conn, {"t": "cancel", "new_epoch": new})
        self._ever_connected = True
        self._missed = 0
        self._send_ping()
        return True

    def disconnect(self, conn: Conn) -> None:
        """転送層から接続終了を通知。置き換え済みの旧接続は無視する。"""
        if self.conn is conn:
            self.conn = None
            self.hello_done = False
            if self.on_disconnect:
                self.on_disconnect()

    def _send(self, conn: Conn, msg: dict) -> None:
        conn.send_text(json.dumps(msg, ensure_ascii=False))

    def _error(self, conn: Conn, code: int, message: str) -> None:
        self._send(conn, {"t": "error", "code": code, "message": message[:200]})

    # --- ping / pong ---
    def _send_ping(self) -> None:
        self._last_ping = self.clock()
        self._unanswered = True
        self._send(self.conn, {"t": "ping", "ts": int(self._last_ping * 1000)})

    def tick(self) -> bool:
        """定期呼び出し。切断扱いにしたら True。"""
        if self.conn is None:
            return False
        if self.clock() - self._last_ping < self.cfg.PING_INTERVAL:
            return False
        if self._unanswered:
            self._missed += 1
            if self._missed >= MISSED_PINGS_LIMIT:
                log.warning("pong missed %d times: connection dropped", self._missed)
                dead = self.conn
                self.disconnect(dead)
                dead.close(1001)
                return True
        self._send_ping()
        return False

    # --- 受信 ---
    def on_text(self, conn: Conn, raw: str) -> None:
        if conn is not self.conn:
            return
        try:
            msg = json.loads(raw)
        except ValueError:
            log.warning("invalid json dropped")
            self._error(conn, ERR_OUT_OF_RANGE, "invalid json")
            return
        if not validate_message(msg):
            if isinstance(msg, dict) and msg.get("t") == "hello":
                log.warning("invalid hello rejected")
                conn.close(CLOSE_VERSION_MISMATCH)
                return
            log.warning("out-of-range message rejected: %.200r", msg)  # 原因の値を残す (200 字で打ち切り)
            self._error(conn, ERR_OUT_OF_RANGE, "invalid or out-of-range value")
            return
        t = msg["t"]
        if t == "hello":
            if msg["version"] != PROTOCOL_VERSION:
                log.warning("version mismatch: %s", msg["version"])
                conn.close(CLOSE_VERSION_MISMATCH)
                return
            self.hello_done = True
            if self.on_hello:  # 接続直後に現在の状態を知らせる (状態遷移が起きるまで表示が「未接続」のままになるため)
                self.on_hello()
        elif t == "pong":
            self._unanswered = False
            self._missed = 0
        elif t == "ping":
            self._send(conn, {"t": "pong", "ts": msg["ts"]})
        elif t == "ui":
            for k in UI_FIELDS:
                if k in msg:
                    self.settings[k] = msg[k]
            if self.on_ui:
                self.on_ui(dict(self.settings))
        elif t == "playback":
            if not self.epoch.is_current(msg["epoch"]):
                log.info("stale playback dropped: epoch %d != %d", msg["epoch"], self.epoch.current)
                return
            if self.on_playback:
                self.on_playback(msg)
        elif t == "snap":  # ドラマモードの画面 1 枚 (設計書 6.5)
            if self.on_snap:
                self.on_snap(msg)
        else:
            log.warning("unexpected message type from browser: %s", t)

    def on_binary(self, conn: Conn, data: bytes) -> None:
        if conn is not self.conn:
            return
        if len(data) > MAX_CHUNK_PCM:
            log.warning("oversize binary dropped: %d bytes", len(data))
            return
        try:
            seq, pcm = unpack_up_frame(data)
        except ValueError:
            log.warning("short binary dropped: %d bytes", len(data))
            return
        if self.on_audio:
            self.on_audio(seq, pcm)


class _WsAdapter:
    def __init__(self, ws) -> None:
        self.ws = ws

    def send_text(self, s: str) -> None:
        asyncio.ensure_future(self._safe(self.ws.send(s)))

    def send_bytes(self, b: bytes) -> None:
        asyncio.ensure_future(self._safe(self.ws.send(b)))

    def close(self, code: int) -> None:
        asyncio.ensure_future(self._safe(self.ws.close(code)))

    @staticmethod
    async def _safe(coro) -> None:
        try:
            await coro
        except (ConnectionClosed, ConnectionError, OSError, RuntimeError) as e:  # 切断済みへの送信は無視
            log.debug("send to closed ws ignored: %s", e)


async def serve(core: WsCore, port: int = 0):
    """websockets サーバを起動して (server, ping_task) を返す。"""
    from websockets.asyncio.server import serve as ws_serve

    async def handler(ws):
        conn = _WsAdapter(ws)
        if not core.connect(conn, ws.request.headers.get("Origin")):
            await ws.wait_closed()
            return
        try:
            async for m in ws:
                if isinstance(m, bytes):
                    core.on_binary(conn, m)
                else:
                    core.on_text(conn, m)
        finally:
            core.disconnect(conn)

    server = await ws_serve(handler, core.host, port, ping_interval=None,
                            max_size=4 * MAX_CHUNK_PCM)

    async def pinger():
        while True:
            await asyncio.sleep(1)
            core.tick()

    return server, asyncio.ensure_future(pinger())
