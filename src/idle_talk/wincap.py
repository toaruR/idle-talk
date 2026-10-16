"""前面ウィンドウの撮影 (設計書 6.7、Windows 専用)。標準ライブラリ (ctypes / zlib) だけで撮って PNG にする。

画面モードの source = "window" で使う。ブラウザの共有と違い、前面のウィンドウが替われば自動でそれを撮る。
撮るのは画面に見えているまま (画面の DC から切り抜く) なので、DRM 付きの動画などは黒く写る。
"""
from __future__ import annotations

import base64
import logging
import struct
import sys
import zlib
from typing import NamedTuple

from .screen import THUMB_H, THUMB_W

log = logging.getLogger("idle_talk.wincap")

SHOT_W, SHOT_H = 640, 360  # fit の既定 (web/client.js の SNAP_W / SNAP_H と同じ)
TITLE_MAX = 100  # LLM に渡すウィンドウタイトルの上限字数
# idle-talk 自身の画面 (web/index.html の <title>)。ブラウザのタイトルは「idle-talk - Google Chrome」のように先頭に来る。
# 部分一致にしないのは、VS Code 等のタイトルにもフォルダ名の idle-talk が入るため
SELF_TITLE = "idle-talk"


class Shot(NamedTuple):
    png: str      # base64
    thumb: bytes  # THUMB_W x THUMB_H の明るさ
    title: str


def fit(w: int, h: int, max_w: int = SHOT_W, max_h: int = SHOT_H) -> tuple[int, int]:
    """縦横比を保って max_w x max_h に収める (拡大はしない)。web/client.js の snapSize と同じ。"""
    s = min(max_w / w, max_h / h, 1)
    return max(1, round(w * s)), max(1, round(h * s))


def excluded(title: str, patterns: tuple[str, ...]) -> bool:
    """タイトルが空 (デスクトップ等)、idle-talk 自身の画面、除外の語を含む (大文字小文字は区別しない) なら撮らない。"""
    t = title.casefold()
    return not title.strip() or t.startswith(SELF_TITLE) or any(p.casefold() in t for p in patterns)


def bgra_to_png(bgra: bytes, w: int, h: int) -> bytes:
    """上から下の順の BGRA 画素を RGB の PNG にする。"""
    rgb = bytearray(w * h * 3)
    rgb[0::3], rgb[1::3], rgb[2::3] = bgra[2::4], bgra[1::4], bgra[0::4]
    stride = w * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(h))  # 各行フィルタなし

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def posterize(bgra: bytes, bits: int) -> bytes:
    """各色を上位 bits ビットに減らす (段の中央の値にする)。文字は読めたまま PNG が小さくなる (1280x720 で 5 ビットなら約半分)。"""
    if bits >= 8:
        return bgra
    step = 1 << (8 - bits)
    return bgra.translate(bytes(v - v % step + step // 2 for v in range(256)))


def bgra_to_gray(bgra: bytes) -> bytes:
    return bytes(round(0.299 * bgra[i + 2] + 0.587 * bgra[i + 1] + 0.114 * bgra[i]) for i in range(0, len(bgra), 4))


class WindowCapture:
    """前面ウィンドウを撮る。grab() は撮れなければ (除外・最小化・失敗) None。"""

    def __init__(self, exclude: tuple[str, ...] = (), width: int = SHOT_W, color_bits: int = 8) -> None:
        if sys.platform != "win32":
            raise OSError("window capture is only available on Windows")
        import ctypes
        from ctypes import wintypes
        self.exclude = exclude
        self.size = (width, width * 9 // 16)  # この 16:9 の枠に収める
        self.color_bits = color_bits
        self._c, self._w = ctypes, wintypes
        u, g, d = ctypes.WinDLL("user32"), ctypes.WinDLL("gdi32"), ctypes.WinDLL("dwmapi")
        H = wintypes.HANDLE
        for f, res, args in (
                (u.GetForegroundWindow, wintypes.HWND, []),
                (u.GetWindowTextLengthW, ctypes.c_int, [wintypes.HWND]),
                (u.GetWindowTextW, ctypes.c_int, [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]),
                (u.IsIconic, wintypes.BOOL, [wintypes.HWND]),
                (u.GetWindowRect, wintypes.BOOL, [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]),
                (u.GetSystemMetrics, ctypes.c_int, [ctypes.c_int]),
                (u.GetDC, wintypes.HDC, [wintypes.HWND]),
                (u.ReleaseDC, ctypes.c_int, [wintypes.HWND, wintypes.HDC]),
                (d.DwmGetWindowAttribute, ctypes.c_long, [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]),
                (g.CreateCompatibleDC, wintypes.HDC, [wintypes.HDC]),
                (g.CreateCompatibleBitmap, wintypes.HBITMAP, [wintypes.HDC, ctypes.c_int, ctypes.c_int]),
                (g.SelectObject, H, [wintypes.HDC, H]),
                (g.SetStretchBltMode, ctypes.c_int, [wintypes.HDC, ctypes.c_int]),
                (g.SetBrushOrgEx, wintypes.BOOL, [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]),
                (g.StretchBlt, wintypes.BOOL, [wintypes.HDC] + [ctypes.c_int] * 4 + [wintypes.HDC] + [ctypes.c_int] * 4
                 + [wintypes.DWORD]),
                (g.GetDIBits, ctypes.c_int, [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT, ctypes.c_void_p,
                                             ctypes.c_void_p, wintypes.UINT]),
                (g.DeleteObject, wintypes.BOOL, [H]),
                (g.DeleteDC, wintypes.BOOL, [wintypes.HDC])):
            f.restype, f.argtypes = res, args  # 64bit のハンドルを int に切り詰めないよう型を付ける
        self._u, self._g, self._d = u, g, d
        try:  # 拡大表示 (DPI 125% 等) でも実際の画素の座標で撮る
            u.SetProcessDpiAwarenessContext.restype = wintypes.BOOL
            u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2
        except (AttributeError, OSError):
            pass

    def _title(self, hwnd) -> str:
        n = self._u.GetWindowTextLengthW(hwnd)
        buf = self._c.create_unicode_buffer(n + 1)
        self._u.GetWindowTextW(hwnd, buf, n + 1)
        return buf.value

    def _rect(self, hwnd) -> tuple[int, int, int, int] | None:
        """ウィンドウの見えている範囲 (影を除く) を仮想画面に収めて (left, top, right, bottom)。"""
        r = self._w.RECT()
        if (self._d.DwmGetWindowAttribute(hwnd, 9, self._c.byref(r), self._c.sizeof(r)) != 0  # EXTENDED_FRAME_BOUNDS
                and not self._u.GetWindowRect(hwnd, self._c.byref(r))):
            return None
        m = self._u.GetSystemMetrics
        vx, vy = m(76), m(77)  # SM_XVIRTUALSCREEN / SM_YVIRTUALSCREEN
        box = (max(r.left, vx), max(r.top, vy), min(r.right, vx + m(78)), min(r.bottom, vy + m(79)))
        return box if box[2] - box[0] > 0 and box[3] - box[1] > 0 else None

    def _stretch(self, screen, box: tuple[int, int, int, int], w: int, h: int) -> bytes:
        """画面の box を w x h に縮めて上から下の順の BGRA にする。"""
        c, g = self._c, self._g
        mem = g.CreateCompatibleDC(screen)
        bmp = g.CreateCompatibleBitmap(screen, w, h)
        try:
            old = g.SelectObject(mem, bmp)
            g.SetStretchBltMode(mem, 4)  # HALFTONE: 縮小で文字がつぶれにくい
            g.SetBrushOrgEx(mem, 0, 0, None)
            if not g.StretchBlt(mem, 0, 0, w, h, screen, box[0], box[1], box[2] - box[0], box[3] - box[1],
                                0x00CC0020 | 0x40000000):  # SRCCOPY | CAPTUREBLT
                raise OSError("StretchBlt failed")
            g.SelectObject(mem, old)
            header = struct.pack("<IiiHHIIiiII", 40, w, -h, 1, 32, 0, 0, 0, 0, 0, 0)  # 32bit、上から下
            info = c.create_string_buffer(header + bytes(16))
            buf = c.create_string_buffer(w * h * 4)
            if g.GetDIBits(mem, bmp, 0, h, buf, info, 0) != h:
                raise OSError("GetDIBits failed")
            return buf.raw
        finally:
            g.DeleteObject(bmp)
            g.DeleteDC(mem)

    def grab(self) -> Shot | None:
        hwnd = self._u.GetForegroundWindow()
        if not hwnd or self._u.IsIconic(hwnd):
            return None
        title = self._title(hwnd)
        if excluded(title, self.exclude):
            log.info("screen capture: excluded window")  # タイトルは残さない (見ているものが分かるため)
            return None
        box = self._rect(hwnd)
        if box is None:
            return None
        w, h = fit(box[2] - box[0], box[3] - box[1], *self.size)
        screen = self._u.GetDC(None)
        try:
            bgra = self._stretch(screen, box, w, h)
            thumb = bgra_to_gray(self._stretch(screen, box, THUMB_W, THUMB_H))
        finally:
            self._u.ReleaseDC(None, screen)
        png = bgra_to_png(posterize(bgra, self.color_bits), w, h)
        return Shot(base64.b64encode(png).decode(), thumb, title[:TITLE_MAX])
