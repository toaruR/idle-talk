"""呼びかけ (ウェイクワード) で ChatGPT の音声会話を始める (設計書 6.11)。

専用プロファイル (data/chatgpt-chrome) の Chrome を remote debugging つきで起動し、CDP で chatgpt.com の
音声ボタンを押す。クリックは Input.dispatchMouseEvent (信頼された入力) なので、マイクの許可とユーザー操作の
条件を満たす。会話の間 idle-talk は聞かず話さず、声も画面の更新も無くなったら終了ボタンを押して戻る。

ページの判定 (PROBE_JS) は Chappy (D:\\vagrant\\chappy、WebView2 版) の voice-probe.js を移したもの。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable
from pathlib import Path

from .config import Chatgpt

log = logging.getLogger(__name__)

WAKE_EXTRA = 2  # 呼びかけの前後に許す字数 (「ねえチャッピー」「チャッピーさん」)。文中の「はチャッピーで大丈夫だよ」は起こさない
VOICED_FRAMES = 10  # 会話中のマイクで声とみなす連続フレーム数 (20ms x 10 = 200ms)。物音 1 つで延長しない
START_TIMEOUT_S = 12.0  # ページの読み込み + 音声ボタン
STOP_TIMEOUT_S = 6.0
OFF_STABLE_S = 0.5  # 終了ボタンの後、開始ボタンに戻ったまま保つ時間
CLOSED_S = 2.0  # 会話中に音声の画面がこれだけ続けて無ければ、利用者が ChatGPT 側で終えたとみなす
POLL_S = 0.5
LAUNCH_TIMEOUT_S = 15.0
PROFILE_DIR = "chatgpt-chrome"

# 呼びかけた結果 (start の戻り値)
STARTED, ALREADY = "started", "already"
LOGIN, NEEDS_ACTION, UNSUPPORTED, TIMEOUT, NO_BROWSER = "login", "needs_action", "unsupported", "timeout", "no_browser"
OK_RESULTS = (STARTED, ALREADY)
# 会話が終わった理由 (session の戻り値)
END_IDLE, END_CLOSED, END_MAX, END_LOST = "idle", "closed", "max", "lost"

_STRIP = set("ー〜~・")


def _norm(text: str) -> str:
    """NFKC、カタカナ→ひらがな、英字は小文字、空白・句読点・長音を除く (Chappy の WakePhraseMatcher と同じ方針)。"""
    out = []
    for c in unicodedata.normalize("NFKC", text):
        if c in _STRIP or not (c.isalnum() or "\u3040" <= c <= "\u30ff"):
            continue
        if "\u30a1" <= c <= "\u30f6":
            c = chr(ord(c) - 0x60)
        out.append(c.lower())
    return "".join(out)


def wake_match(text: str, phrases: tuple[str, ...]) -> bool:
    """発話がほぼ呼びかけだけ (前後 WAKE_EXTRA 字まで) なら True。あいまい一致はしない: 長音を除くと
    「ちゃっぴ」の 4 字で、1 字違いを許すと「〜ちゃった」が当たった (talk.log 12.5 万発話で 24 件、2026-11-19)。"""
    n = _norm(text)
    for p in phrases:
        w = _norm(p)
        if w and w in n and len(n) - len(w) <= WAKE_EXTRA:
            return True
    return False


class IdleWatch:
    """会話の無操作の締め切り。最初は first_s、誰かの声を聞いた後は idle_s。活動があるたびに延ばす。"""

    def __init__(self, now: float, first_s: float, idle_s: float) -> None:
        self.first_s, self.idle_s = first_s, idle_s
        self.spoke = False
        self.deadline = now + first_s

    def activity(self, now: float, voice: bool = False) -> None:
        self.spoke = self.spoke or voice
        self.deadline = max(self.deadline, now + (self.idle_s if self.spoke else self.first_s))

    def expired(self, now: float) -> bool:
        return now >= self.deadline


# ---- CDP ----
class CdpError(RuntimeError):
    pass


class Cdp:
    """1 つのターゲットへの CDP 接続 (websockets)。応答は id で突き合わせ、イベントは捨てる。"""

    def __init__(self, ws) -> None:
        self.ws = ws
        self._id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader = asyncio.get_running_loop().create_task(self._read())

    @classmethod
    async def connect(cls, url: str) -> Cdp:
        from websockets.asyncio.client import connect
        try:  # Origin を付けない (Chrome は --remote-allow-origins 外の Origin を拒否する)。システムのプロキシを通さない
            ws = await connect(url, max_size=None, open_timeout=5, proxy=None)
        except (OSError, TimeoutError, ValueError) as e:
            raise CdpError(f"connect failed: {e}") from e
        return cls(ws)

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                fut = self._pending.pop(msg.get("id"), None) if "id" in msg else None
                if fut is None or fut.done():
                    continue
                if "error" in msg:
                    fut.set_exception(CdpError(str(msg["error"])[:200]))
                else:
                    fut.set_result(msg.get("result", {}))
        except Exception as e:  # 切断: 待っている呼び出しを全部失敗にする
            log.debug("cdp reader ended: %s", e)
        finally:
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(CdpError("connection closed"))
            self._pending.clear()

    async def send(self, method: str, params: dict | None = None, timeout: float = 5.0) -> dict:
        if self._reader.done():
            raise CdpError("connection closed")
        self._id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[self._id] = fut
        try:
            await self.ws.send(json.dumps({"id": self._id, "method": method, "params": params or {}}))
            return await asyncio.wait_for(fut, timeout)
        except TimeoutError as e:
            raise CdpError(f"{method} timed out") from e
        except Exception as e:
            if isinstance(e, CdpError):
                raise
            raise CdpError(f"{method} failed: {e}") from e
        finally:
            self._pending.pop(self._id, None)

    @property
    def closed(self) -> bool:
        return self._reader.done()

    async def close(self) -> None:
        try:
            await self.ws.close()
        except Exception:
            pass
        self._reader.cancel()


# ---- ページ ----
PROBE_JS = r"""(() => {
  const shown = e => {
    const r = e.getBoundingClientRect(), s = getComputedStyle(e);
    return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth && s.visibility === 'visible' && s.display !== 'none' && s.opacity !== '0';
  };
  const visible = e => shown(e) && !e.disabled && e.getAttribute('aria-disabled') !== 'true';
  const name = e => {
    const labelled = e.getAttribute('aria-labelledby');
    return (e.getAttribute('aria-label') || (labelled ? labelled.split(/\s+/).map(id => document.getElementById(id)?.textContent || '').join(' ') : '') || e.getAttribute('title') || e.innerText || '').trim();
  };
  const allControls = [...document.querySelectorAll('button,[role="button"],a')], controls = allControls.filter(visible);
  const starts = controls.filter(e => /^(start voice(?: (?:mode|chat|conversation))?|use voice(?: mode)?|voice(?: mode)?|音声モード(?:を開始(?:する)?)?|音声(?:会話|チャット)(?:を開始(?:する)?)?|音声を開始(?:する)?|音声を使用(?:する)?)$/i.test(name(e)));
  const mutes = controls.filter(e => /^(?:unmute|mute)(?: (?:the |your )?microphone)?$|^microphone$|^ミュート(?:(?:を)?解除)?$|^マイク(?:を)?(?:ミュート|オフ|オン)?(?:する)?$|^マイクを(?:オン|オフ)にする$/i.test(name(e)));
  const dialogs = [...document.querySelectorAll('[role="dialog"],[role="alertdialog"]')].filter(shown);
  const explicitEnd = e => /^(?:(?:end|exit|close|stop) voice(?: (?:mode|chat|conversation|call))?|end (?:voice )?call|音声(?:会話|チャット|モード)?を終了(?:する)?|音声(?:会話|チャット|モード)から退出(?:する)?|通話を終了(?:する)?)$/i.test(name(e))
    || /^(?:(?:close|exit|end)-voice|voice-(?:close|exit|end))(?:-(?:button|mode|chat|call))*$/i.test(e.getAttribute('data-testid') || '');
  const voiceContext = e => {
    const dialog = e.closest('[role="dialog"],[role="alertdialog"]');
    return mutes.length > 0 && (dialog ? mutes.some(m => dialog.contains(m)) : dialogs.length === 0);
  };
  const ends = controls.filter(e => explicitEnd(e) || voiceContext(e) && /^(end (?:voice )?(?:chat|conversation)|close|会話を終了(?:する)?|閉じる|終了)$/i.test(name(e)));
  const logins = controls.filter(e => /^(log in|sign in|ログイン|サインイン)$/i.test(name(e)));
  const messages = [...document.querySelectorAll('[data-message-author-role="assistant"]')], last = messages.length ? messages[messages.length - 1] : null;
  const busyControl = allControls.some(e => shown(e) && /^(stop generating|stop streaming|stop response|stop responding|生成を停止|応答を停止|回答を停止|生成を中止)$/i.test(name(e)));
  const busyMessage = !!last && (last.getAttribute('aria-busy') === 'true' || [...last.querySelectorAll('[aria-busy="true"]')].some(shown));
  const meta = { busy: busyControl || busyMessage, revision: messages.length + ':' + (last ? last.textContent.length : 0) };
  const result = v => Object.assign(v, meta);
  const point = e => {
    const r = e.getBoundingClientRect(), x = Math.max(0, r.left) + (Math.min(innerWidth, r.right) - Math.max(0, r.left)) / 2, y = Math.max(0, r.top) + (Math.min(innerHeight, r.bottom) - Math.max(0, r.top)) / 2;
    const hit = document.elementFromPoint(x, y);
    return hit && (hit === e || e.contains(hit)) ? { x, y } : null;
  };
  if (ends.length > 1) return result({ kind: 'Unsupported' });
  if (ends.length === 1) return result({ kind: 'Active', end: point(ends[0]) });
  if (mutes.length > 0) return result({ kind: 'Unsupported' });
  if (logins.length > 0) return result({ kind: 'Login' });
  if (dialogs.length > 0) return result({ kind: 'NeedsAction' });
  if (starts.length > 1) return result({ kind: 'Unsupported' });
  if (starts.length === 1) {
    const p = point(starts[0]);
    return result(p ? { kind: 'Ready', start: p } : { kind: 'NeedsAction' });
  }
  return result({ kind: 'Loading' });
})()"""

CONTROLS_JS = r"""[...document.querySelectorAll('button,[role="button"],a')].map(e => (e.getAttribute('aria-label') || e.getAttribute('title') || e.innerText || '').trim()).filter(n => /音声|声|マイク|voice|mic/i.test(n)).slice(0, 20)"""
SETTLE_S = 2.0
_STUCK = {"Login": LOGIN, "NeedsAction": NEEDS_ACTION, "Unsupported": UNSUPPORTED}

# 音声の画面が出ている (終了ボタンが特定できない Unsupported も、マイクのボタンがあれば音声中のことがある)
VOICE_KINDS = ("Active", "Unsupported")


class ChatgptPage:
    """chatgpt.com のタブ 1 つ。probe でページの状態 (kind・ボタンの位置・生成中・応答の版) を返す。"""

    def __init__(self, cdp: Cdp, origin: str) -> None:
        self.cdp, self.origin = cdp, origin

    async def probe(self) -> dict:
        res = await self.cdp.send("Runtime.evaluate", {"expression": PROBE_JS, "returnByValue": True})
        if "exceptionDetails" in res:
            return {"kind": "Loading"}
        url = (await self.cdp.send("Runtime.evaluate", {"expression": "location.origin", "returnByValue": True}))
        if url.get("result", {}).get("value") != self.origin:  # 認証のページ等に移っている
            return {"kind": "NeedsAction"}
        value = res.get("result", {}).get("value")
        return value if isinstance(value, dict) and "kind" in value else {"kind": "Unsupported"}

    async def click(self, x: float, y: float) -> None:
        await self.cdp.send("Input.dispatchMouseEvent", {"type": "mouseMoved", "x": x, "y": y})
        await self.cdp.send("Input.dispatchMouseEvent",
                            {"type": "mousePressed", "x": x, "y": y, "button": "left", "clickCount": 1})
        await self.cdp.send("Input.dispatchMouseEvent",
                            {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1})

    async def navigate(self, url: str) -> None:
        await self.cdp.send("Page.navigate", {"url": url})

    async def front(self) -> None:
        await self.cdp.send("Page.bringToFront")


# ---- Chrome ----
def find_chrome(path: str = "") -> str | None:
    if path:
        return path if Path(path).is_file() else None
    roots = [os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    for root in filter(None, roots):
        p = Path(root) / "Google" / "Chrome" / "Application" / "chrome.exe"
        if p.is_file():
            return str(p)
    return None


def chrome_args(chrome: str, profile: Path, port: int, url: str) -> list[str]:
    return [chrome, f"--user-data-dir={profile}", f"--remote-debugging-port={port}",
            "--no-first-run", "--no-default-browser-check",
            # 最小化・裏のウィンドウでも音声会話のタイマーと音を止めない
            "--disable-background-timer-throttling", "--disable-renderer-backgrounding",
            "--disable-backgrounding-occluded-windows", f"--app={url}"]


def _http_json(url: str, method: str = "GET", timeout: float = 2.0):
    req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (127.0.0.1 の CDP だけ)
        return json.loads(r.read().decode("utf-8"))


class ChromeHost:
    """専用プロファイルの Chrome。起動していなければ起動し、chatgpt.com のタブを返す。"""

    def __init__(self, cfg: Chatgpt, profile: Path,
                 http: Callable[..., object] = _http_json,
                 popen: Callable[[list[str]], object] = lambda a: subprocess.Popen(a, close_fds=True),
                 connect: Callable[[str], Awaitable[Cdp]] = Cdp.connect,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg, self.profile = cfg, profile
        self.http, self.popen, self.connect, self.clock = http, popen, connect, clock
        self.base = f"http://127.0.0.1:{cfg.port}"
        parts = urllib.parse.urlsplit(cfg.url)
        self.origin = f"{parts.scheme}://{parts.netloc}"
        self._page: ChatgptPage | None = None

    async def _version(self) -> dict | None:
        try:
            v = await asyncio.to_thread(self.http, self.base + "/json/version")
        except (OSError, ValueError):
            return None
        ws = v.get("webSocketDebuggerUrl", "") if isinstance(v, dict) else ""
        return v if ws.startswith(f"ws://127.0.0.1:{self.cfg.port}/") else None

    async def launch(self) -> None:
        if await self._version() is not None:
            return
        chrome = find_chrome(self.cfg.chrome)
        if chrome is None:
            raise CdpError("chrome.exe not found ([chatgpt].chrome)")
        self.profile.mkdir(parents=True, exist_ok=True)
        log.info("chatgpt: launching chrome (profile %s, port %d)", self.profile, self.cfg.port)
        self.popen(chrome_args(chrome, self.profile, self.cfg.port, self.cfg.url))
        deadline = self.clock() + LAUNCH_TIMEOUT_S
        while self.clock() < deadline:
            if await self._version() is not None:
                return
            await asyncio.sleep(0.3)
        raise CdpError("chrome did not open the debugging port")

    async def page(self) -> ChatgptPage:
        if self._page is not None and not self._page.cdp.closed:
            return self._page
        await self.launch()
        targets = await asyncio.to_thread(self.http, self.base + "/json/list")
        pages = [t for t in targets if t.get("type") == "page" and str(t.get("url", "")).startswith(self.origin)]
        if not pages:
            pages = [await asyncio.to_thread(self.http, self.base + "/json/new?" + self.cfg.url, "PUT")]
        cdp = await self.connect(pages[0]["webSocketDebuggerUrl"])
        self._page = ChatgptPage(cdp, self.origin)
        await self._grant_mic()
        return self._page

    async def _grant_mic(self) -> None:
        """chatgpt.com のマイクを許可しておく (専用プロファイルの初回に許可の確認を出さない)。失敗はログだけ。"""
        v = await self._version()
        if v is None:
            return
        try:
            b = await self.connect(v["webSocketDebuggerUrl"])
            try:
                await b.send("Browser.grantPermissions", {"origin": self.origin, "permissions": ["audioCapture"]})
            finally:
                await b.close()
        except CdpError as e:
            log.warning("chatgpt: grant mic failed: %s", e)

    def forget(self) -> None:
        self._page = None


class ChatgptVoice:
    """呼びかけの判定と、ChatGPT の音声会話 1 回分 (開始 → 無操作の監視 → 終了)。"""

    def __init__(self, cfg: Chatgpt, host: ChromeHost,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> None:
        self.cfg, self.host, self.clock, self.sleep = cfg, host, clock, sleep
        self.watch: IdleWatch | None = None
        self._voiced = 0

    def matches(self, text: str) -> bool:
        return wake_match(text, self.cfg.phrases)

    def hear(self, level: float, threshold: float) -> None:
        """会話中のマイクの 1 フレーム (Director から)。声が VOICED_FRAMES 続いたら締め切りを延ばす。
        スピーカーなら ChatGPT の声も入るが、どちらが話していても会話は続いている。"""
        self._voiced = self._voiced + 1 if level >= threshold else 0
        if self._voiced >= VOICED_FRAMES and self.watch is not None:
            self.watch.activity(self.clock(), voice=True)

    async def warm(self) -> None:
        """起動時に Chrome とタブを用意しておく (呼びかけからの開始を速くする)。失敗はログだけ。"""
        try:
            await self.host.page()
        except (CdpError, OSError, ValueError, KeyError) as e:
            log.warning("chatgpt: warm up failed: %s", e)

    async def start(self) -> str:
        """音声ボタンを押す。STARTED / ALREADY なら会話中。"""
        try:
            page = await self.host.page()
            try:
                await page.front()
            except CdpError:
                pass
            clicked = False
            stuck: tuple[str, float] | None = None  # (止まって見える状態, その始まり)
            deadline = self.clock() + START_TIMEOUT_S
            while self.clock() < deadline:
                snap = await page.probe()
                kind = snap.get("kind")
                if kind == "Active":
                    return STARTED if clicked else ALREADY
                if kind in _STUCK:  # 読み込みの途中は一瞬 NeedsAction になる (実機 2026-11-19)。続いたときだけ確定
                    stuck = stuck if stuck and stuck[0] == kind else (kind, self.clock())
                    if self.clock() - stuck[1] >= SETTLE_S:
                        await self._log_controls(page)
                        return _STUCK[kind]
                else:
                    stuck = None
                if kind == "Ready" and not clicked and snap.get("start"):
                    await page.click(snap["start"]["x"], snap["start"]["y"])
                    clicked = True
                await self.sleep(0.2)
            await self._log_controls(page)
            return TIMEOUT
        except (CdpError, OSError, ValueError, KeyError) as e:
            log.warning("chatgpt: start failed: %s", e)
            self.host.forget()
            return NO_BROWSER

    async def _log_controls(self, page: ChatgptPage) -> None:
        """始められなかったとき、音声らしいボタンの名前をログに出す (ChatGPT の画面が変わったときの PROBE_JS の直し用)。"""
        try:
            res = await page.cdp.send("Runtime.evaluate", {"expression": CONTROLS_JS, "returnByValue": True})
            log.warning("chatgpt: voice-like controls: %s", res.get("result", {}).get("value"))
        except (CdpError, AttributeError) as e:
            log.debug("chatgpt: controls not listed: %s", e)

    async def session(self) -> str:
        """会話の終わりを待つ。無操作・上限なら終了ボタンを押してから理由を返す。"""
        started = self.clock()
        self.watch = IdleWatch(started, self.cfg.first_timeout_s, self.cfg.idle_timeout_s)
        self._voiced = 0
        off_since: float | None = None
        last_rev = None
        try:
            while True:
                await self.sleep(POLL_S)
                now = self.clock()
                try:
                    page = await self.host.page()
                    snap = await page.probe()
                except (CdpError, OSError, ValueError, KeyError) as e:
                    log.warning("chatgpt: lost the page: %s", e)
                    self.host.forget()
                    return END_LOST
                if snap.get("kind") in VOICE_KINDS:
                    off_since = None
                else:
                    off_since = now if off_since is None else off_since
                    if now - off_since >= CLOSED_S:
                        return END_CLOSED
                rev = snap.get("revision")
                if snap.get("busy") or (last_rev is not None and rev != last_rev):
                    self.watch.activity(now)
                last_rev = rev
                if now - started >= self.cfg.max_s:
                    await self.stop()
                    return END_MAX
                if self.watch.expired(now):
                    await self.stop()
                    return END_IDLE
        finally:
            self.watch = None

    async def stop(self) -> bool:
        """終了ボタンを押し、開始ボタンに戻ったのを確かめる。押せない・戻らなければページを開き直してマイクを切る。"""
        try:
            page = await self.host.page()
            if await self._end_by_button(page):
                return True
            log.warning("chatgpt: end button did not work, reloading the page")
            await page.navigate(self.cfg.url)
            return await self._wait_off(page, 8.0)
        except (CdpError, OSError, ValueError, KeyError) as e:
            log.warning("chatgpt: stop failed: %s", e)
            self.host.forget()
            return False

    async def _end_by_button(self, page: ChatgptPage) -> bool:
        clicked = False
        off_since: float | None = None
        deadline = self.clock() + STOP_TIMEOUT_S
        while self.clock() < deadline:
            snap = await page.probe()
            kind = snap.get("kind")
            if kind in ("Ready", "Login"):
                off_since = self.clock() if off_since is None else off_since
                if self.clock() - off_since >= OFF_STABLE_S:
                    return True
            else:
                off_since = None
                if kind == "Active" and not clicked and snap.get("end"):
                    await page.click(snap["end"]["x"], snap["end"]["y"])
                    clicked = True
                elif kind in ("Unsupported", "NeedsAction") or (kind == "Active" and not snap.get("end")):
                    return False
            await self.sleep(0.1)
        return False

    async def _wait_off(self, page: ChatgptPage, timeout: float) -> bool:
        off_since: float | None = None
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            await self.sleep(0.3)
            try:
                kind = (await page.probe()).get("kind")
            except CdpError:
                off_since = None
                continue
            if kind in VOICE_KINDS:
                off_since = None
                continue
            off_since = self.clock() if off_since is None else off_since
            if self.clock() - off_since >= 1.0:
                return True
        return False
