"""見ているページを読む (利用者が「このページどう思う？」と声で頼んだとき)。

前面に近いブラウザ (Chrome / Edge) のウィンドウを Z 順で探し、アドレスバーの URL を UI Automation で読む
(PowerShell の System.Windows.Automation。Python の追加の依存は要らない)。本文は r.jina.ai でテキストにする。
外部に出るのは読むページの URL だけ。idle-talk 自身・[screen].exclude の語を含むウィンドウは読まない。
"""
from __future__ import annotations

import ctypes
import logging
import re
import subprocess
import sys
import webbrowser
from collections.abc import Callable
from urllib.parse import quote

from .wincap import excluded

log = logging.getLogger("idle_talk.browser_page")

BROWSER_TITLE = re.compile(r" - (?:Google Chrome|Microsoft\u200b? Edge)$")  # Edge のタイトルにはゼロ幅空白が入る
CHROME_CLASS = "Chrome_WidgetWin_1"  # VS Code・Obsidian 等の Electron も同じクラスなのでタイトルの末尾で見分ける
UIA_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 15.0  # r.jina.ai は記事ページで 7 秒前後かかった (2026-10-21)。検索の FETCH_TIMEOUT_S より長く待つ
PAGE_ASK_MAX_CHARS = 1500  # LLM に渡す本文の上限 (検索の抜粋より長く。履歴は ANSWER_TURNS に絞る)
MIN_LINE_CHARS = 20  # これより短い行 (メニュー・ボタン等) は本文に含めない
PAGE_NOTICE = "ちょっと読んでみるね。"
PAGE_FAIL = "ごめん、見ているページが分からなかった。ブラウザでページを開いてから、もう一度聞いてね。"
SUMMARY_NOTICE = "ChatGPT で要約を開いたよ。"
SUMMARY_URL = "https://chatgpt.com/?q="
SUMMARY_PROMPT = "{url} を日本語、小学生にも分かるように箇条書きで要約して"
_SUMMARY_WORD = re.compile(r"要約|まとめ")

# 「このページ」「この記事」等を指して、意見・感想・要約を求める発話
_PAGE_WORD = re.compile(r"(この|その|今の|(今|いま)見て(い|る)|開いて(い|る))\s*(ページ|記事|サイト|ブログ|タブ|画面)"
                        r"|(ページ|記事|サイト|ブログ)(を|、)?\s*(読んで|見て)")
_ASK_WORD = re.compile(r"どう(思う|おもう|かな|です|でしょう|\?|？|$)|意見|感想|要約|まとめて|読んで|見て|教えて|説明|解説|何て|なんて|どんな")

_PS = ("Add-Type -AssemblyName UIAutomationClient,UIAutomationTypes;"
       "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
       "$e=[Windows.Automation.AutomationElement]::FromHandle([IntPtr]{hwnd});"
       "$c=New-Object Windows.Automation.PropertyCondition("
       "[Windows.Automation.AutomationElement]::ControlTypeProperty,[Windows.Automation.ControlType]::Edit);"
       "$x=$e.FindFirst([Windows.Automation.TreeScope]::Descendants,$c);"
       "if($x){{$x.GetCurrentPattern([Windows.Automation.ValuePattern]::Pattern).Current.Value}}")


def wants_page(text: str) -> bool:
    """利用者が見ているページについて意見・感想・要約を求めているか。"""
    t = re.sub(r"\s+", "", text)
    m = _PAGE_WORD.search(t)
    return m is not None and _ASK_WORD.search(t[m.start():]) is not None


def wants_summary(text: str) -> bool:
    """見ているページの要約を求めているか (ChatGPT のタブで要約させる)。"""
    return wants_page(text) and _SUMMARY_WORD.search(text) is not None


def summary_url(url: str) -> str:
    """ページの URL を要約させる ChatGPT の URL。"""
    return SUMMARY_URL + quote(SUMMARY_PROMPT.format(url=url), safe="")


def page_title(title: str) -> str:
    """ウィンドウのタイトルからブラウザ名を除く。ブラウザでなければ空。"""
    m = BROWSER_TITLE.search(title)
    return title[:m.start()] if m else ""


def normalize_url(value: str) -> str:
    """アドレスバーの値 (https:// が省かれる) を URL にする。http(s) のページでなければ空。"""
    v = value.strip()
    if not v or " " in v or v.startswith(("chrome://", "edge://", "about:", "file:", "chrome-extension:")):
        return ""
    if not v.startswith(("http://", "https://")):
        v = "https://" + v
    host = v.split("://", 1)[1].split("/", 1)[0]
    return v if "." in host or host.startswith(("localhost", "127.")) else ""


def body_excerpt(text: str, title: str = "", max_chars: int = PAGE_ASK_MAX_CHARS) -> str:
    """r.jina.ai のテキストから、メニュー等の短い行を除いた本文を max_chars 字。ページのタイトル (「 - 」より前) が
    出ればそこから (前の宣伝・お知らせを飛ばす。Qiita は本文の前に求人の案内が入った)。"""
    body = text.split("Markdown Content:", 1)[-1]
    flat = " / ".join(ln.strip() for ln in body.splitlines() if len(ln.strip()) >= MIN_LINE_CHARS)
    head = title.split(" - ", 1)[0].strip()[:MIN_LINE_CHARS]
    i = flat.find(head) if len(head) >= 4 else -1
    return flat[max(i, 0):][:max_chars]


def page_messages(messages: list[dict], page: dict, turns: int, image: str | None = None) -> list[dict]:
    """読んだページ (本文。無ければ image = 画面の data URL) を最後の利用者発話に添えて答えさせるメッセージ。
    履歴は turns に絞る。"""
    system = messages[:1] if messages and messages[0]["role"] == "system" else []
    rest = messages[len(system):][-turns:]
    last = rest.pop() if rest and rest[-1]["role"] == "user" else {"role": "user", "content": ""}
    rule = ("この内容を踏まえて、利用者の求めに話し言葉で 2〜4 文で答えてください。意見を求められたら自分の考えも言ってください。"
            "{src}に無いことは推測せず、URL は書かないでください)")
    if image is None:
        ref = (f"(利用者がブラウザで見ているページ「{page['title']}」の本文の抜粋です。" + rule.format(src="抜粋")
               + "\n" + page["text"])
        return system + rest + [{"role": "user", "content": (last["content"] + "\n\n" + ref).strip()}]
    title = f"「{page['title']}」" if page.get("title") else ""
    ref = f"(利用者が見ているウィンドウ{title}の画面です。" + rule.format(src="映っている範囲")
    return system + rest + [{"role": "user", "content": [
        {"type": "text", "text": (last["content"] + "\n\n" + ref).strip()},
        {"type": "image_url", "image_url": {"url": image}}]}]


def browser_windows() -> list[tuple[int, str]]:
    """表示中のブラウザのウィンドウを前面から順に [(hwnd, ページのタイトル)]。Windows 以外は空。"""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    found: list[tuple[int, str]] = []
    buf = ctypes.create_unicode_buffer(512)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def each(hwnd, _):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd):
            user32.GetClassNameW(hwnd, buf, 512)
            if buf.value == CHROME_CLASS:
                user32.GetWindowTextW(hwnd, buf, 512)
                title = page_title(buf.value)
                if title:
                    found.append((int(hwnd), title))
        return True

    user32.EnumWindows(each, 0)  # Z 順 (前面から)
    return found


def address_bar(hwnd: int) -> str:
    """ブラウザのウィンドウのアドレスバーの値。読めなければ空。"""
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS.format(hwnd=int(hwnd))],
                           capture_output=True, timeout=UIA_TIMEOUT_S,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("address bar read failed: %s", e)
        return ""
    return r.stdout.decode("utf-8", "replace").strip().splitlines()[0] if r.stdout.strip() else ""


class PageReader:
    """reader() -> {title, url, text} | None (ページが見つからない)。本文が読めなければ text は空。
    windows() / address(hwnd) / fetch(url, headers) / open_url(url) は注入可能 (テストで差し替える)。"""

    def __init__(self, fetch: Callable[..., bytes], exclude: tuple[str, ...] = (),
                 windows: Callable[[], list[tuple[int, str]]] = browser_windows,
                 address: Callable[[int], str] = address_bar,
                 open_url: Callable[[str], object] = lambda u: webbrowser.open(u, new=2)) -> None:
        self._fetch = fetch
        self.exclude = tuple(exclude)
        self._windows = windows
        self._address = address
        self._open = open_url

    def __call__(self) -> dict | None:
        found = self.locate()
        if found is not None:
            found["text"] = self._read(found["url"], found["title"])
        return found

    def locate(self) -> dict | None:
        """見ているページの {title, url} (本文は読まない)。見つからなければ None。"""
        for hwnd, title in self._windows():
            if excluded(title, self.exclude):  # idle-talk 自身・パスワード管理など。ログにもタイトルを残さない
                continue
            url = normalize_url(self._address(hwnd))
            if not url:
                continue
            log.info("page ask: %s", url)
            return {"title": title[:200], "url": url}
        log.info("page ask: no browser page")
        return None

    def summarize(self) -> dict | None:
        """見ているページを ChatGPT に要約させるタブを開き、そのページの {title, url} を返す。"""
        found = self.locate()
        if found is not None:
            self._open(summary_url(found["url"]))
        return found

    def _read(self, url: str, title: str) -> str:
        from .web_search import READER_URL
        try:
            text = self._fetch(READER_URL + url, {"X-Return-Format": "text"}, timeout=READ_TIMEOUT_S)
        except (OSError, ValueError) as e:
            log.warning("page read failed: %s %s", url, e)
            return ""
        return body_excerpt(text.decode("utf-8", "replace"), title)
