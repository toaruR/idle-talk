"""Web 検索 (利用者の質問に答えるための調べもの)。

LLM が返答の先頭に「検索: 検索語」とだけ書いたら、検索エンジン (DuckDuckGo lite → 失敗・0 件なら Bing) で検索し、
上位 1 件のページ本文を r.jina.ai (ページをテキストにする中継) で読んで抜粋を添え、もう一度 LLM に答えさせる。
[search].realtime なら Yahoo! リアルタイム検索 (ログイン不要で X の投稿を新しい順に検索できる) の投稿も添える。
外部に出るのは検索語と読むページの URL だけ (音声・書き起こし全文は送らない)。
"""
from __future__ import annotations

import base64
import binascii
import datetime
import html
import json
import logging
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from functools import lru_cache
from importlib.resources import files

log = logging.getLogger("idle_talk.web_search")

DDG_URL = "https://lite.duckduckgo.com/lite/?kl=jp-jp&q="
BING_URL = "https://www.bing.com/search?setlang=ja&cc=JP&q="
BING_NEWS_URL = "https://www.bing.com/news/search?setlang=ja&cc=JP&qft=sortbydate%3d%221%22&q="  # 新しい順
READER_URL = "https://r.jina.ai/"
REALTIME_URL = "https://search.yahoo.co.jp/realtime/search?p="  # X の投稿を新しい順に (ログイン不要)
REALTIME_TOP_URL = "https://search.yahoo.co.jp/realtime"  # X のトレンド (buzzTrend)
FETCH_TIMEOUT_S = 8.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"  # 既定の Python-urllib だと結果が返らないことがある
MAX_RESULTS = 5
RESULTS_MAX_CHARS = 800  # LLM に渡す検索結果 (見出しと抜粋) の上限 (ctx 4k を圧迫しない)
PAGE_MAX_CHARS = 800  # LLM に渡すページ本文の抜粋の上限
ANSWER_TURNS = 4  # 結果を渡して答えさせるときの履歴ターン数 (結果のぶん履歴を削る)
SEARCH_MAX_ROUNDS = 3  # 初回を含む。追加検索の無限ループを防ぐ
RESEARCH_ROUND_MAX_CHARS = 1000  # 検索を重ねても資料が際限なく増えないよう各回を制限
SEARCH_INCOMPLETE_NOTICE = "ここまでの検索では、答えを確かめる情報が十分に揃いませんでした。"
SEARCH_NOTICE = "ちょっと調べてみるね。"  # 検索を始めたことを声で伝える
SEARCH_WAIT_NOTICE = "もう少し待ってね。"  # 検索が SEARCH_WAIT_S を超えたら 1 回だけ言う
SEARCH_WAIT_S = 5.0
POSTS_MAX = 3  # 検索結果に添える X の投稿の件数
POSTS_MAX_CHARS = 400  # LLM に渡す X の投稿の上限 (検索結果とは別枠)
POST_MAX_CHARS = 140  # 投稿 1 件の上限
TRENDS_MAX = 3  # 1 回に話題カードにする X のトレンドの数 (トレンドごとに検索ページを 1 回読む)
TREND_POSTS = 3  # トレンドのカードの概要に使う投稿の数
TREND_POST_CHARS = 60  # 同じく投稿 1 件の上限 (カードの概要は director.CARD_FACTS_MAX で切られる)
X_TREND = "x_trend"  # トレンドから作った話題カードの angle

_MARK = re.compile(r"^\s*(?:検索|けんさく)\s*[:：]\s*")
_SEARCH_ANY = re.compile(r"(?:検索|けんさく)\s*[:：]")
_ANY_MARK = re.compile(r"(?:検索|けんさく|リンク)\s*[:：]")
LINK_NOTICE = "読んでみるね。"  # 調べたページの URL を画面の URL 欄に出したことを声で伝える


def search_rule(today: datetime.date) -> str:
    return (f"今日は{today.year}年{today.month}月{today.day}日です。"
            "株価や為替などの今の数値、最新のニュース、今話題になっていること、天気、人物や物事の詳しい事実など、調べないと正確に答えられないことを聞かれたとき、"
            "または利用者が「もっと詳しく」などと直前の話の続きを求めたときは、返答の代わりに 1 行で「検索: 検索語」とだけ出力してください。"
            "検索語は会話の文脈を補った具体的な語にしてください (例: 検索: 日経平均株価 今日)。それ以外は普通に返答してください。")


def with_search_rule(messages: list[dict], today: datetime.date | None = None) -> list[dict]:
    """system メッセージに検索の指示を足した複製を返す。"""
    rule = search_rule(today or datetime.date.today())
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + rule}] + messages[1:]
    return [{"role": "system", "content": rule}] + messages


def link_rule() -> str:
    return ("利用者が、さっき話したニュースや調べたページの URL・リンク・出典・記事を見たい、教えてほしいと言ったときは、"
            "URL を読み上げず、返答の代わりに 1 行で「リンク:」とだけ出力してください。")


def with_link_rule(messages: list[dict]) -> list[dict]:
    """system メッセージにリンク表示の指示を足した複製を返す (直近の検索結果があるときだけ使う)。"""
    if messages and messages[0]["role"] == "system":
        return [{"role": "system", "content": messages[0]["content"] + link_rule()}] + messages[1:]
    return [{"role": "system", "content": link_rule()}] + messages


def find_mark(clause: str, links: bool = False) -> tuple[str, str | None, str]:
    """節の中の指示を探す (先頭とは限らない)。(指示の前の文, 'search' / 'links' / None, 指示より後)。
    指示が無ければ (clause, None, "")。links=False ならリンク表示の指示は探さない。"""
    m = (_ANY_MARK if links else _SEARCH_ANY).search(clause)
    if m is None:
        return clause, None, ""
    kind = "links" if m.group(0).startswith("リンク") else "search"
    return clause[:m.start()].strip(), kind, clause[m.end():]


def link_items(results: list[dict]) -> list[dict]:
    """画面の URL 欄に出す [{title, url}]。http(s) 以外は出さない。"""
    return [{"title": r["title"][:200], "url": r["url"]} for r in results
            if r.get("url", "").startswith(("http://", "https://"))]


def parse_query(text: str) -> str:
    """「検索: 日経平均株価」の行から検索語を取り出す (最初の行のみ)。"""
    line = _MARK.sub("", text.lstrip()).split("\n", 1)[0]
    return line.strip().strip("「」\"'。").strip()


def _clean(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def parse_duckduckgo(page: str) -> list[dict]:
    """DuckDuckGo lite 版の結果ページから [{title, snippet, url}] を取り出す。"""
    out: list[dict] = []
    links = list(re.finditer(r"<a[^>]*href=[\"']([^\"']+)[\"'][^>]*class=[\"']result-link[\"'][^>]*>(.*?)</a>",
                             page, re.S))
    for i, m in enumerate(links):
        href, title = m.groups()
        rest = page[m.end():links[i + 1].start() if i + 1 < len(links) else len(page)]  # 次の結果までに抜粋がある
        sn = re.search(r"class=[\"']result-snippet[\"'][^>]*>(.*?)</td>", rest, re.S)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(html.unescape(href)).query)
        out.append({"title": _clean(title), "snippet": _clean(sn.group(1)) if sn else "",
                    "url": q.get("uddg", [html.unescape(href)])[0]})
    return out


def _bing_url(href: str) -> str:
    """Bing の転送リンク (/ck/a?...&u=a1<base64url>) から元の URL を取り出す。"""
    u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
    if u.startswith("a1"):
        try:
            return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            pass
    return href


def parse_bing(page: str) -> list[dict]:
    """Bing の結果ページから [{title, snippet, url}] を取り出す。"""
    out: list[dict] = []
    for block in page.split('class="b_algo"')[1:]:
        block = re.sub(r"<(script|style)\b.*?</\1>", "", block, flags=re.S)
        h = re.search(r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.S)
        if not h:
            continue
        sn = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        out.append({"title": _clean(h.group(2)), "snippet": _clean(sn.group(1)) if sn else "",
                    "url": _bing_url(html.unescape(h.group(1)))})
    return out


_LISTING_SEGS = {"tag", "tags", "theme", "themes", "category", "categories", "search", "ranking", "list"}
# 話題カードの補充で順に検索するジャンル (毎回同じ記事にならないよう回す)
NEWS_QUERIES = ("ニュース", "経済 ニュース", "IT 技術 ニュース", "科学 研究 ニュース", "スポーツ ニュース",
                "エンタメ ニュース", "国際 ニュース", "暮らし ニュース")


def parse_bing_news(page: str) -> list[dict]:
    """Bing ニュース検索の結果から記事の [{title, snippet, url}] を取り出す。
    概要が無いもの・URL がサイトのトップや 1 階層だけのもの (広告・特集ページ) は除く。"""
    out: list[dict] = []
    for block in page.split('<div class="news-card newsitem')[1:]:
        attrs = dict(re.findall(r'data-(url|title)="([^"]*)"', block.split(">", 1)[0]))
        sn = re.search(r'class="snippet"[^>]*>(.*?)</div>', block, re.S)
        url, title = html.unescape(attrs.get("url", "")), _clean(attrs.get("title", ""))
        snippet = _clean(sn.group(1)) if sn else ""
        segs = urllib.parse.urlparse(url).path.strip("/").split("/")
        if not (title and snippet and url.startswith(("http://", "https://")) and len(segs) >= 2):
            continue
        if _LISTING_SEGS & {s.lower() for s in segs}:  # タグ・特集などの一覧ページは記事ではない
            continue
        out.append({"title": title.removesuffix("...").strip(), "snippet": snippet, "url": url})
    return out


def news_search(query: str, fetch: Callable[..., bytes] | None = None) -> list[dict]:
    """新しい順のニュース記事を検索する (話題カードの補充用)。失敗は []。"""
    try:
        page = (fetch or http_fetch)(BING_NEWS_URL + urllib.parse.quote(query)).decode("utf-8", "replace")
    except (OSError, ValueError) as e:
        log.warning("news search failed: %r %s", query, e)
        return []
    return parse_bing_news(page)


def _page_data(page: str) -> dict:
    """Yahoo! リアルタイム検索のページに埋め込まれた __NEXT_DATA__ の pageData。無ければ ValueError。"""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', page, re.S)
    if not m:
        raise ValueError("no __NEXT_DATA__")
    try:
        return json.loads(m.group(1))["props"]["pageProps"]["pageData"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"unexpected __NEXT_DATA__: {e!r}") from None


def _post_text(text: str) -> str:
    """投稿の本文。検索語の強調 (タブで囲んだ START / END) と先頭の返信先 @ を除く。"""
    text = re.sub(r"\tSTART\t|\tEND\t", "", text)
    text = re.sub(r"^(?:@\w+\s+)+", "", text.strip())
    return re.sub(r"\s+", " ", text).strip()


def parse_realtime(page: str) -> list[dict]:
    """Yahoo! リアルタイム検索の結果から X の投稿 [{title, snippet, url, post}] を新しい順に取り出す。
    センシティブ指定・本文なし・同じ本文 (リポストの重複) は除く。title は投稿日時。"""
    out: list[dict] = []
    seen: set[str] = set()
    for e in (_page_data(page).get("timeline") or {}).get("entry") or []:
        text = _post_text(e.get("displayText") or "")
        if e.get("possiblySensitive") or not text or text in seen:
            continue
        seen.add(text)
        when = datetime.datetime.fromtimestamp(e["createdAt"]).strftime("%m/%d %H:%M") if e.get("createdAt") else ""
        out.append({"title": f"X の投稿 {when}".strip(), "snippet": text[:POST_MAX_CHARS],
                    "url": (e.get("url") or "").split("?", 1)[0], "post": True})
    return out


def parse_trends(page: str) -> list[dict]:
    """Yahoo! リアルタイム検索のトップから X のトレンド [{query, genre, url}] を順位順に取り出す。"""
    out = []
    for it in (_page_data(page).get("buzzTrend") or {}).get("items") or []:
        q = (it.get("query") or "").strip()
        if q:
            out.append({"query": q, "genre": it.get("genre") or "", "url": REALTIME_URL + urllib.parse.quote(q)})
    return out


def realtime_search(query: str, fetch: Callable[..., bytes] | None = None) -> list[dict]:
    """X の投稿を新しい順に検索する。失敗は []。"""
    try:
        return parse_realtime((fetch or http_fetch)(REALTIME_URL + urllib.parse.quote(query)).decode("utf-8", "replace"))
    except (OSError, ValueError) as e:
        log.warning("realtime search failed: %r %s", query, e)
        return []


def x_trends(fetch: Callable[..., bytes] | None = None, limit: int = TRENDS_MAX) -> list[dict]:
    """話題カード用の X のトレンド [{title, snippet, url}]。概要は「X で話題 (ジャンル)」+ 新しい投稿の本文。
    投稿の取れないトレンドは除く。失敗は []。"""
    fetch = fetch or http_fetch
    try:
        trends = parse_trends(fetch(REALTIME_TOP_URL).decode("utf-8", "replace"))
    except (OSError, ValueError) as e:
        log.warning("x trends failed: %s", e)
        return []
    out = []
    for t in trends[:limit]:
        posts = realtime_search(t["query"], fetch)[:TREND_POSTS]
        if not posts:
            continue
        head = f"X で話題 ({t['genre']})。" if t["genre"] else "X で話題。"
        out.append({"title": t["query"], "url": t["url"], "angle": X_TREND,
                    "snippet": head + "投稿: " + " / ".join(f"「{p['snippet'][:TREND_POST_CHARS]}」" for p in posts)})
    return out


ENGINES: dict[str, tuple[str, Callable[[str], list[dict]]]] = {
    "duckduckgo": (DDG_URL, parse_duckduckgo),
    "bing": (BING_URL, parse_bing),
}


def page_excerpt(text: str, query: str, max_chars: int = PAGE_MAX_CHARS) -> str:
    """r.jina.ai のテキストから、検索語の最も長い語が最初に出る位置の少し前から max_chars 字を抜き出す。"""
    body = text.split("Markdown Content:", 1)[-1]  # 先頭の Title / URL Source 等の見出しを除く
    flat = " / ".join(ln.strip() for ln in body.splitlines() if ln.strip())
    words = sorted(query.split(), key=len, reverse=True)
    i = flat.find(words[0]) if words else -1
    start = max(0, i - max_chars // 8) if i >= 0 else 0
    return flat[start:start + max_chars]


def format_results(results: list[dict], max_chars: int = RESULTS_MAX_CHARS) -> str:
    posts = [r for r in results if r.get("post")]
    results = [r for r in results if not r.get("post")]
    lines, n = [], 0
    for i, r in enumerate(results, 1):
        line = f"{i}. {r['title']}: {r['snippet']}"
        if n + len(line) > max_chars:
            break
        lines.append(line)
        n += len(line) + 1
    pages = [f"[{r['title']} の本文の抜粋] {r['page']}" for r in results if r.get("page")]
    said, n = [], 0
    for r in posts:
        line = f"- ({r['title'].removeprefix('X の投稿').strip()}) {r['snippet']}"
        if n + len(line) > POSTS_MAX_CHARS:
            break
        said.append(line)
        n += len(line) + 1
    return "\n".join(lines + pages + (["[X の投稿 (新しい順)]"] + said if said else []))


def answer_messages(messages: list[dict], query: str, results: list[dict],
                    today: datetime.date | None = None) -> list[dict]:
    """検索結果を最後の利用者発話に添えて答えさせるメッセージ。履歴は ANSWER_TURNS に絞る。"""
    today = today or datetime.date.today()
    system = messages[:1] if messages and messages[0]["role"] == "system" else []
    turns = messages[len(system):][-ANSWER_TURNS:]
    last = turns.pop() if turns and turns[-1]["role"] == "user" else {"role": "user", "content": ""}
    body = format_results(results)
    if body:
        ref = (f"(Web で「{query}」を検索した結果です。今日は{today.year}年{today.month}月{today.day}日。"
               "この結果に書かれている内容だけを根拠に、話し言葉で 2〜4 文で答えてください。"
               "数値は結果のとおりに言い、日時が分かれば添えてください。結果に無いことは推測せず、分からないと言ってください。"
               + ("X の投稿は個人の発言なので事実と決めつけず、「〜という声がある」のように伝えてください。"
                  if any(r.get("post") for r in results) else "")
               + "URL や「検索:」は書かないでください)\n" + body)
    else:
        ref = (f"(Web で「{query}」を検索しましたが、結果が得られませんでした。"
               "推測で答えず、調べられなかったことを話し言葉で 1〜2 文で伝えてください)")
    return system + turns + [{"role": "user", "content": (last["content"] + "\n\n" + ref).strip()}]


@lru_cache(maxsize=1)
def search_deeper_skill() -> str:
    """検索時だけ読む、同梱の深掘り検索手順。frontmatter はプロンプトに含めない。"""
    text = files("idle_talk").joinpath("skills/search-deeper/SKILL.md").read_text(encoding="utf-8")
    return text.split("---", 2)[-1].strip()


def research_messages(messages: list[dict], searches: list[tuple[str, list[dict]]], remaining: int,
                      today: datetime.date | None = None) -> list[dict]:
    """元の質問・各回の資料・残り回数を渡し、追加検索か最終回答を選ばせる。"""
    today = today or datetime.date.today()
    has_system = bool(messages and messages[0]["role"] == "system")
    rule = (search_deeper_skill() if remaining > 0 else
            "調査は終了しました。今回は最終回答だけを返してください。追加検索の指示は出さないでください。"
            "資料で確認できた内容と未確認の点を、人物設定の口調で2〜4文で伝えてください。"
            "未確認の事実を作らず、資料からの推論は推論として区別し、Xの投稿は個人の発言として扱ってください。"
            "日時が重要なら添え、URLは読み上げないでください。")
    rule += f"\n追加検索の残り回数: {remaining}回。"
    system = [{"role": "system", "content": (messages[0]["content"] + "\n" if has_system else "") + rule}]
    turns = messages[int(has_system):][-ANSWER_TURNS:]
    last = turns.pop() if turns and turns[-1]["role"] == "user" else {"role": "user", "content": ""}
    sources = []
    for i, (query, results) in enumerate(searches, 1):
        body = format_results(results)[:RESEARCH_ROUND_MAX_CHARS] or "結果が得られませんでした。"
        sources.append(f"[検索{i}: {query}]\n{body}")
    ref = (f"今日は{today.year}年{today.month}月{today.day}日。以下は検索履歴と外部資料です。\n"
           + "\n\n".join(sources))
    return system + turns + [{"role": "user", "content": (last["content"] + "\n\n" + ref).strip()}]


def http_fetch(url: str, headers: dict[str, str] | None = None, timeout: float = FETCH_TIMEOUT_S) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class WebSearch:
    """search(query) -> 結果。engines を順に試し、結果が出たら read_page なら上位 1 件の本文を読む。
    取得・解析の失敗は次のエンジンへ、全滅なら [] (会話は止めない)。fetch(url, headers=None) は注入可能。"""

    def __init__(self, fetch: Callable[..., bytes] = http_fetch, engines: tuple[str, ...] = ("duckduckgo", "bing"),
                 read_page: bool = True, max_results: int = MAX_RESULTS, realtime: bool = False) -> None:
        unknown = set(engines) - set(ENGINES)
        if unknown:
            raise ValueError(f"unknown search engines: {sorted(unknown)}")
        self._fetch = fetch
        self.engines = tuple(engines)
        self.read_page = read_page
        self.max_results = max_results
        self.realtime = realtime  # X の投稿 (POSTS_MAX 件) も添える

    def __call__(self, query: str) -> list[dict]:
        results: list[dict] = []
        for name in self.engines:
            url, parse = ENGINES[name]
            try:
                results = parse(self._fetch(url + urllib.parse.quote(query)).decode("utf-8", "replace"))
            except (OSError, ValueError) as e:
                log.warning("web search failed (%s): %r %s", name, query, e)
                continue
            if results:
                log.info("web search (%s): %r -> %d results", name, query, len(results))
                break
            log.warning("web search returned no results (%s): %r", name, query)
        results = results[:self.max_results]
        if results and self.read_page:
            results[0] = {**results[0], "page": self._read(results[0]["url"], query)}
        if self.realtime:
            posts = realtime_search(query, self._fetch)[:POSTS_MAX]
            log.info("realtime search: %r -> %d posts", query, len(posts))
            results += posts
        return results

    def _read(self, url: str, query: str) -> str:
        """ページ本文の抜粋。読めなければ空 (検索結果の抜粋だけで答える)。"""
        if not url.startswith(("http://", "https://")):
            return ""
        try:
            text = self._fetch(READER_URL + url, {"X-Return-Format": "text"}).decode("utf-8", "replace")
        except (OSError, ValueError) as e:
            log.warning("page read failed: %s %s", url, e)
            return ""
        return page_excerpt(text, query)
