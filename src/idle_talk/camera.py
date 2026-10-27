"""カメラモード (設計書 6.9) の映像ソース。cameras.toml のソースを順に 1 つずつ見る。

ソースはブラウザの Web カメラ (url = "browser")、ネットワークストリーム (rtsp:// 等、ffmpeg が読める URL)、
この PC のカメラ (device = DirectShow の機器名) のいずれか。ブラウザ以外は ffmpeg で 1 コマ取る。
FSM には触れない。画像は保存しない (Director の keep_snaps を除く)。URL のパスワードはログに出さない。
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import NamedTuple

from .screen import THUMB_H, THUMB_W, thumb_diff

CAMERAS_FILE = "cameras.toml"  # 設定ファイル (idle_talk.toml) と同じフォルダ。パスワードを含むので git に入れない
BROWSER = "browser"
SNAP_W, SNAP_H = 640, 360  # web/client.js の SNAP_W / SNAP_H と同じ (約 106 トークン)
GRAB_TIMEOUT_S = 15.0  # RTSP は接続に 3〜4 秒かかった (実測 2026-10-27)
JPEG_Q = 5  # ffmpeg の mjpeg の品質 (2〜31、小さいほど高画質)


class Source(NamedTuple):
    label: str  # LLM に伝える名前 (「部屋のカメラ」「駐車場」など)
    url: str = ""  # "browser" か ffmpeg の入力 URL
    device: str = ""  # DirectShow の機器名 (ffmpeg -list_devices true -f dshow -i dummy で一覧)

    @property
    def browser(self) -> bool:
        return self.url == BROWSER


class Cameras(NamedTuple):
    sources: tuple[Source, ...] = (Source("", BROWSER),)  # ファイルが無ければブラウザの Web カメラだけ
    ffmpeg: str = "ffmpeg"


def load_cameras(path: str | Path) -> Cameras:
    """cameras.toml を読む。無ければ既定 (ブラウザの Web カメラだけ)。不正は ValueError。"""
    p = Path(path)
    if not p.is_file():
        return Cameras()
    with p.open("rb") as f:
        raw = tomllib.load(f)
    unknown = set(raw) - {"ffmpeg", "source"}
    if unknown:
        raise ValueError(f"unknown keys in {p}: {sorted(unknown)}")
    ffmpeg = raw.get("ffmpeg", "ffmpeg")
    if type(ffmpeg) is not str or not ffmpeg:
        raise ValueError(f"{p}: ffmpeg must be a non-empty string")
    items = raw.get("source", [])
    if type(items) is not list or not items:
        raise ValueError(f"{p}: [[source]] is required")
    sources = []
    for i, it in enumerate(items):
        if type(it) is not dict or set(it) - set(Source._fields):
            raise ValueError(f"{p}: source[{i}] has unknown keys: {sorted(set(it) - set(Source._fields))}")
        label, url, device = it.get("label", ""), it.get("url", ""), it.get("device", "")
        if not all(type(v) is str for v in (label, url, device)) or len(label) > 40:
            raise ValueError(f"{p}: source[{i}] label (up to 40 chars) / url / device must be strings")
        if bool(url) == bool(device):
            raise ValueError(f"{p}: source[{i}] needs exactly one of url / device")
        sources.append(Source(label, url, device))
    return Cameras(tuple(sources), ffmpeg)


def mask_url(url: str) -> str:
    """ログ用に URL のパスワードを伏せる (rtsp://user:pass@host -> rtsp://user:***@host)。"""
    return re.sub(r"(://[^:/@]*):[^@/]*@", r"\1:***@", url)


def describe(src: Source) -> str:
    return src.label or ("browser" if src.browser else mask_url(src.url) if src.url else src.device)


def _run(cmd: list[str], data: bytes | None = None) -> bytes:
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    r = subprocess.run(cmd, input=data, capture_output=True, timeout=GRAB_TIMEOUT_S, creationflags=flags)
    if r.returncode != 0 or not r.stdout:
        # stderr に URL (パスワード) が出ることがあるので伏せる
        raise OSError(f"ffmpeg exit {r.returncode}: {mask_url(r.stderr.decode(errors='replace').strip()[-200:])}")
    return r.stdout


def grab(src: Source, ffmpeg: str = "ffmpeg") -> tuple[str, bytes]:
    """ブラウザ以外のソースから 1 コマ取り、(JPEG の base64, 縮小画像) を返す。失敗は OSError / TimeoutExpired。"""
    if src.device:
        inp = ["-f", "dshow", "-i", f"video={src.device}"]
    else:
        inp = (["-rtsp_transport", "tcp"] if src.url.startswith("rtsp") else []) + ["-i", src.url]
    jpeg = _run([ffmpeg, "-hide_banner", "-loglevel", "error", *inp, "-frames:v", "1",
                 "-vf", f"scale={SNAP_W}:{SNAP_H}:force_original_aspect_ratio=decrease",
                 "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", str(JPEG_Q), "pipe:1"])
    thumb = _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-i", "pipe:0",
                  "-vf", f"scale={THUMB_W}:{THUMB_H},format=gray", "-f", "rawvideo", "pipe:1"], jpeg)
    if len(thumb) != THUMB_W * THUMB_H:
        raise OSError(f"ffmpeg thumb size {len(thumb)}")
    return base64.b64encode(jpeg).decode(), thumb


class CameraCycle:
    """ソースを順に 1 つずつ見る。変化の判定はソースごとに、前に LLM に見せたそのソースの画像と比べる。"""

    def __init__(self, cameras: Cameras) -> None:
        self.c = cameras
        self._i = -1
        self._base: dict[int, bytes | None] = {}

    @property
    def needs_browser(self) -> bool:
        return any(s.browser for s in self.c.sources)

    def next(self) -> Source:
        self._i = (self._i + 1) % len(self.c.sources)
        return self.c.sources[self._i]

    def diff(self, thumb: bytes | None) -> float:
        """今のソースで前に LLM に見せた画像との差 (%)。初回・縮小画像なしは 100。"""
        base = self._base.get(self._i)
        return 100.0 if thumb is None or base is None else thumb_diff(thumb, base)

    def mark_shown(self, thumb: bytes | None) -> None:
        self._base[self._i] = thumb

    def reset(self) -> None:
        self._i = -1
        self._base.clear()
