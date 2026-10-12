"""模擬ブラウザ (§11.2 結合層)。WsCore の Conn として振る舞い、音声を送り、下りチャンクに playback で応答する。"""
from __future__ import annotations

import json

from idle_talk.protocol import pack_up_frame, unpack_down_header

LOUD = (b"\xd0\x07" * 320)  # 20ms, 振幅 2000
QUIET = b"\x00\x00" * 320


class MockBrowser:
    def __init__(self, auto_playback: bool = True):
        self.texts: list[dict] = []
        self.chunks: list[tuple[int, int, int, bytes]] = []
        self.closed: list[int] = []
        self.auto_playback = auto_playback
        self.core = None
        self.seq = 0

    # Conn
    def send_text(self, s: str) -> None:
        self.texts.append(json.loads(s))

    def send_bytes(self, b: bytes) -> None:
        ep, utt, cid, _n = unpack_down_header(b)
        self.chunks.append((ep, utt, cid, b[16:]))
        if self.auto_playback and self.core is not None:
            self.playback(ep, utt, cid, "started")
            self.playback(ep, utt, cid, "finished")

    def close(self, code: int) -> None:
        self.closed.append(code)

    # ブラウザ側動作
    def attach(self, core, origin: str = "http://localhost:8765") -> None:
        self.core = core
        assert core.connect(self, origin)
        core.on_text(self, json.dumps({"t": "hello", "version": 1, "sample_rate": 16000, "client_id": "mock"}))

    def playback(self, ep: int, utt: int, cid: int, state: str, pos: int = 100) -> None:
        self.core.on_text(self, json.dumps({"t": "playback", "epoch": ep, "utterance_id": utt,
                                            "chunk_id": cid, "position_ms": pos, "state": state}))

    def send_frame(self, pcm: bytes) -> None:
        self.core.on_binary(self, pack_up_frame(self.seq, pcm))
        self.seq += 1

    def speak(self, loud_frames: int = 15, quiet_frames: int = 12) -> None:
        for _ in range(loud_frames):
            self.send_frame(LOUD)
        for _ in range(quiet_frames):
            self.send_frame(QUIET)

    def of(self, t: str) -> list[dict]:
        return [m for m in self.texts if m["t"] == t]
