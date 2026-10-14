import asyncio
import datetime
import time

import pytest

from idle_talk import pipeline as pipeline_mod
from idle_talk.pipeline import ResponsePipeline
from idle_talk.web_search import (ANSWER_TURNS, BING_NEWS_URL, BING_URL, DDG_URL, LINK_NOTICE, READER_URL, SEARCH_NOTICE, SEARCH_WAIT_NOTICE,
                                  WebSearch, answer_messages, find_mark, format_results, link_items, page_excerpt, parse_bing,
                                  news_search, parse_bing_news, parse_duckduckgo, parse_query, with_search_rule)

LITE_PAGE = """
<table>
<tr><td>1.&nbsp;</td><td>
<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nikkei.com%2Fmarketdata%2Fquote%2FNK225%2F&amp;rut=abc" class='result-link'>日経平均株価 (日経225) &amp; チャート</a>
</td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>
 <b>日経平均株価</b> 67,777.52 (10/13 15:45 大引) -1,253.40
</td></tr>
<tr><td>2.&nbsp;</td><td>
<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Ffinance.yahoo.co.jp%2Fquote%2F998407.O&amp;rut=def" class='result-link'>日経平均株価：指数情報 - Yahoo!ファイナンス</a>
</td></tr>
<tr><td>&nbsp;</td><td class='result-snippet'>指数、チャート、最新の関連ニュース。</td></tr>
</table>
"""
TODAY = datetime.date(2026, 10, 14)


# ---- 解析 ----

def test_parse_duckduckgo_title_snippet_url():
    rs = parse_duckduckgo(LITE_PAGE)
    assert rs == [
        {"title": "日経平均株価 (日経225) & チャート", "snippet": "日経平均株価 67,777.52 (10/13 15:45 大引) -1,253.40",
         "url": "https://www.nikkei.com/marketdata/quote/NK225/"},
        {"title": "日経平均株価：指数情報 - Yahoo!ファイナンス", "snippet": "指数、チャート、最新の関連ニュース。",
         "url": "https://finance.yahoo.co.jp/quote/998407.O"},
    ]


def test_parse_duckduckgo_bot_page_is_empty():
    assert parse_duckduckgo("<html>anomaly detected</html>") == []


@pytest.mark.parametrize("clause,want", [
    ("検索:", ("", "search", "")), ("検索： 日経平均", ("", "search", " 日経平均")),
    ("けんさく: 天気", ("", "search", " 天気")),
    ("知りたいなら検索: ベトナム 資金洗浄", ("知りたいなら", "search", " ベトナム 資金洗浄")),
    ("検査の結果、", ("検査の結果、", None, "")), ("検索した", ("検索した", None, "")),
])
def test_find_mark(clause, want):
    assert find_mark(clause) == want


def test_parse_query_first_line_only():
    assert parse_query("検索：「日経平均株価 今日」\n日経平均は") == "日経平均株価 今日"
    assert parse_query(" 検索: 東京の天気") == "東京の天気"


def test_format_results_respects_max_chars():
    rs = [{"title": "t" * 10, "snippet": "s" * 10}] * 5
    assert format_results(rs, max_chars=52).count("\n") == 1  # 1 行 25 字 + 改行で 2 行まで


def test_with_search_rule_appends_to_system_only():
    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "株価は?"}]
    out = with_search_rule(msgs, TODAY)
    assert out[0]["content"].startswith("SYS") and "検索: 検索語" in out[0]["content"] and "2026年10月14日" in out[0]["content"]
    assert out[1:] == msgs[1:] and msgs[0]["content"] == "SYS"  # 元は変えない


def test_answer_messages_trims_history_and_attaches_results():
    hist = [{"role": "user" if i % 2 else "assistant", "content": f"h{i}"} for i in range(10)]
    msgs = [{"role": "system", "content": "SYS"}] + hist + [{"role": "user", "content": "日経平均は?"}]
    out = answer_messages(msgs, "日経平均株価", parse_duckduckgo(LITE_PAGE), TODAY)
    assert out[0] == {"role": "system", "content": "SYS"}  # 検索の指示は付けない (再び「検索:」と答えさせない)
    assert len(out) == 1 + ANSWER_TURNS
    assert out[-1]["content"].startswith("日経平均は?") and "67,777.52" in out[-1]["content"]


def test_answer_messages_without_results_says_unknown():
    out = answer_messages([{"role": "user", "content": "q"}], "x", [], TODAY)
    assert "結果が得られませんでした" in out[-1]["content"]


BING_PAGE = """
<ol id="b_results"><li class="b_algo" data-id><style>.x{}</style>
<h2><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=1&amp;u=a1aHR0cHM6Ly93d3cubmlra2VpLmNvbS9tYXJrZXRkYXRhL3F1b3RlL05LMjI1Lw&amp;ntb=1">日経平均株価 <strong>リアルタイム</strong></a></h2>
<div class="b_caption"><p class="b_lineclamp2"><span class="news_dt">2026年6月24日</span>&nbsp;&#0183;&#32;日経平均の指数情報。</p></div></li>
<li class="b_algo"><h2><a href="https://example.test/b">B</a></h2></li></ol>
"""
JINA_TEXT = ("Title: 日経平均株価\n\nURL Source: https://www.nikkei.com/\n\nMarkdown Content:\nメニュー\n速報\n"
             "国内株式\n日経平均株価\n\n67,777.52\n(10/13 15:45 大引)\n始値\n68,564.85\n")


BING_NEWS_PAGE = """
<div class="news-card newsitem cardcommon" data-url="https://www.msn.com/ja-jp/sports/baseball/abc" data-title="日本シリーズ進出は? プロ野球 ..." data-author="BASEBALL KING">
<div class="snippet" title="x">12日に放送された番組で、日本シリーズ進出チームを予想した。</div></div>
<div class="news-card newsitem cardcommon" data-url="https://mainichi.jp/kyushu/" data-title="秋得キャンペーン">
<div class="snippet">50%OFF で読み放題</div></div>
<div class="news-card newsitem cardcommon" data-url="https://www.sanspo.com/tag/player-mlb/otani/" data-title="大谷翔平ニュース">
<div class="snippet">最新ニュース一覧</div></div>
<div class="news-card newsitem cardcommon" data-url="https://example.test/a/1" data-title="概要なし"></div>
"""


def test_parse_bing_news_keeps_articles_only():
    assert parse_bing_news(BING_NEWS_PAGE) == [
        {"title": "日本シリーズ進出は? プロ野球", "snippet": "12日に放送された番組で、日本シリーズ進出チームを予想した。",
         "url": "https://www.msn.com/ja-jp/sports/baseball/abc"}]


def test_news_search_sorted_by_date_and_failure_is_empty():
    urls = []
    assert len(news_search("経済 ニュース", lambda u: urls.append(u) or BING_NEWS_PAGE.encode())) == 1
    assert urls[0].startswith(BING_NEWS_URL) and "sortbydate" in urls[0]

    def boom(u):
        raise OSError("down")
    assert news_search("x", boom) == []


def test_parse_bing_decodes_redirect_url():
    assert parse_bing(BING_PAGE) == [
        {"title": "日経平均株価 リアルタイム", "snippet": "2026年6月24日 · 日経平均の指数情報。",
         "url": "https://www.nikkei.com/marketdata/quote/NK225/"},
        {"title": "B", "snippet": "", "url": "https://example.test/b"},
    ]


def test_page_excerpt_starts_near_query_word():
    ex = page_excerpt(JINA_TEXT, "日経平均株価 今日", max_chars=40)
    assert "67,777.52" in ex and "Title" not in ex and len(ex) <= 40


def fake_fetch(pages, calls):
    def fetch(url, headers=None):
        calls.append((url, headers))
        for prefix, body in pages.items():
            if url.startswith(prefix):
                if isinstance(body, Exception):
                    raise body
                return body.encode()
        raise AssertionError(url)
    return fetch


def test_web_search_failure_returns_empty():
    calls = []
    ws = WebSearch(fake_fetch({DDG_URL: OSError("down"), BING_URL: OSError("down")}, calls))
    assert ws("x") == [] and len(calls) == 2


def test_web_search_falls_back_to_bing_and_reads_top_page():
    calls = []
    ws = WebSearch(fake_fetch({DDG_URL: "<html>anomaly</html>", BING_URL: BING_PAGE, READER_URL: JINA_TEXT}, calls))
    rs = ws("日経平均株価")
    assert rs[0]["url"] == "https://www.nikkei.com/marketdata/quote/NK225/" and "67,777.52" in rs[0]["page"]
    assert "page" not in rs[1]
    assert calls[-1] == (READER_URL + "https://www.nikkei.com/marketdata/quote/NK225/", {"X-Return-Format": "text"})
    assert "本文の抜粋" in format_results(rs)


def test_web_search_page_read_failure_keeps_results():
    calls = []
    ws = WebSearch(fake_fetch({DDG_URL: LITE_PAGE, READER_URL: OSError("x")}, calls))
    rs = ws("日経平均株価")
    assert len(rs) == 2 and rs[0]["page"] == ""
    assert not any(u.startswith(BING_URL) for u, _ in calls)  # 1 つ目で結果が出たら次のエンジンは使わない


def test_web_search_quotes_query_and_limits_without_page():
    calls = []
    ws = WebSearch(fake_fetch({DDG_URL: LITE_PAGE}, calls), engines=("duckduckgo",), read_page=False, max_results=1)
    assert len(ws("日経 平均")) == 1 and len(calls) == 1
    assert calls[0][0].endswith("q=%E6%97%A5%E7%B5%8C%20%E5%B9%B3%E5%9D%87")


def test_web_search_rejects_unknown_engine():
    with pytest.raises(ValueError):
        WebSearch(engines=("google",))


# ---- 応答パイプライン ----

def make(llm_outputs, search, order=None):
    calls, audio, emits = [], [], []

    def llm(messages):
        calls.append(messages)
        return iter(llm_outputs[len(calls) - 1])

    p = ResponsePipeline(llm, lambda t: b"wav", lambda: None,
                         lambda ep, u, c, t, w: audio.append(t), emits.append)
    return p, calls, audio, emits


MSGS = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "日経平均株価をおしえて"}]


def test_search_mark_announces_then_answers_from_results():
    queries = []
    p, calls, audio, emits = make([["検", "索: 日経平均", "株価\n", "余計な続き"], ["今日は", "6万円台です。"]],
                                  lambda q: queries.append(q) or parse_duckduckgo(LITE_PAGE))

    async def run():
        await p.respond(MSGS, search=lambda q: queries.append(q) or parse_duckduckgo(LITE_PAGE))
        await asyncio.sleep(0)  # 別スレッドからの emit を受け取る
    asyncio.run(run())
    assert queries == ["日経平均株価"]
    assert audio == [SEARCH_NOTICE, "今日は6万円台です。"]
    assert "検索: 検索語" in calls[0][0]["content"] and "検索: 検索語" not in calls[1][0]["content"]
    assert "67,777.52" in calls[1][-1]["content"]
    assert [(m["t"], m["active"]) for m in emits if m["t"] == "search"] == [("search", True), ("search", False)]


def test_plain_answer_is_not_searched():
    p, calls, audio, _ = make([["検査の", "結果、", "大丈夫です。"]], None)
    searched = []
    asyncio.run(p.respond(MSGS, search=searched.append))
    assert searched == [] and len(calls) == 1
    assert audio == ["検査の結果、", "大丈夫です。"]


def test_without_search_no_rule_and_mark_is_spoken():
    p, calls, audio, _ = make([["検索: x"]], None)
    asyncio.run(p.respond(MSGS))
    assert calls[0] == MSGS and audio == ["検索: x"]


def test_empty_query_falls_back_to_user_text():
    queries = []
    p, _, _, _ = make([["検索:\n"], ["分かりません。"]], None)
    asyncio.run(p.respond(MSGS, search=lambda q: queries.append(q) or []))
    assert queries == ["日経平均株価をおしえて"]


def test_slow_search_says_wait_notice_once(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "SEARCH_WAIT_S", 0.05)
    p, _, audio, _ = make([["検索: x\n"], ["はい。"]], None)
    asyncio.run(p.respond(MSGS, search=lambda q: time.sleep(0.3) or []))
    assert audio == [SEARCH_NOTICE, SEARCH_WAIT_NOTICE, "はい。"]


@pytest.mark.parametrize("clause,links,want", [
    ("リンク:", True, ("", "links", "")), ("リンク：1", True, ("", "links", "1")),
    ("はい、リンク:", True, ("はい、", "links", "")),
    ("リンク:", False, ("リンク:", None, "")), ("リンゴ", True, ("リンゴ", None, "")),
    ("検索: x", True, ("", "search", " x")),
])
def test_find_mark_links(clause, links, want):
    assert find_mark(clause, links=links) == want


def test_search_mark_at_end_speaks_preamble_then_searches():
    """実機ログ: 返答の末尾に「検索: 語」を書いた。前の文は読み上げ、指示は読まずに検索する。"""
    queries = []
    p, calls, audio, _ = make([["ニュースの URL を", "探しているんですね。", "知りたいなら、",
                                "検索: ベトナム 資金洗浄 最新ニュース"], ["分かりました。"]], None)
    asyncio.run(p.respond(MSGS, search=lambda q: queries.append(q) or []))
    assert queries == ["ベトナム 資金洗浄 最新ニュース"]
    assert audio == ["ニュースの URL を探しているんですね。", "知りたいなら、", SEARCH_NOTICE, "分かりました。"]
    assert not any("検索" in a for a in audio)


def test_link_items_only_http():
    rs = [{"title": "A", "url": "https://a.test/"}, {"title": "B", "url": "javascript:alert(1)"}, {"title": "C"}]
    assert link_items(rs) == [{"title": "A", "url": "https://a.test/"}]


def test_search_keeps_links_then_link_request_shows_them():
    p, calls, audio, emits = make([["検索: 日経平均株価\n"], ["6万円台です。"], ["リンク", ":"]], None)

    async def run():
        await p.respond(MSGS, search=lambda q: parse_duckduckgo(LITE_PAGE))
        await p.respond(MSGS + [{"role": "user", "content": "そのページを教えて"}], search=lambda q: [])
        await asyncio.sleep(0)
    asyncio.run(run())
    assert "リンク:" not in calls[0][0]["content"]  # 検索結果が無いうちはリンクの指示を足さない
    assert "「リンク:」" in calls[2][0]["content"]
    assert audio[-1] == LINK_NOTICE and len(calls) == 3
    shown = [m for m in emits if m["t"] == "links"]
    assert shown == [{"t": "links", "items": [
        {"title": "日経平均株価 (日経225) & チャート", "url": "https://www.nikkei.com/marketdata/quote/NK225/"},
        {"title": "日経平均株価：指数情報 - Yahoo!ファイナンス", "url": "https://finance.yahoo.co.jp/quote/998407.O"}]}]


def test_link_mark_without_search_results_is_spoken():
    p, _, audio, emits = make([["リンク:", "はこちら。"]], None)
    asyncio.run(p.respond(MSGS, search=lambda q: []))
    assert audio == ["リンク:はこちら。"] and not [m for m in emits if m["t"] == "links"]


def test_cancel_during_search_skips_second_llm():
    p, calls, audio, _ = make([["検索: x\n"], ["古い答え。"]], None)

    def search(q):
        p.epoch.bump()  # 検索中に割り込みで取消された
        return []
    asyncio.run(p.respond(MSGS, search=search))
    assert len(calls) == 1 and audio == [SEARCH_NOTICE]


def test_load_search(tmp_path):
    from idle_talk.config import Search, load_search
    f = tmp_path / "s.toml"
    assert load_search(f) == Search(False)
    f.write_text("[search]\non = true\n", encoding="utf-8")
    assert load_search(f) == Search(True, ("duckduckgo", "bing"), True)
    f.write_text('[search]\non = true\nengines = ["bing"]\nread_page = false\n', encoding="utf-8")
    assert load_search(f) == Search(True, ("bing",), False)


@pytest.mark.parametrize("body", ['[search]\non = "yes"\n', '[search]\nengines = ["google"]\n',
                                  "[search]\nengines = []\n", '[search]\nread_page = 1\n', "[search]\nurl = 1\n"])
def test_load_search_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_search
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_search(f)


def test_repo_settings_search_loads():
    from pathlib import Path

    from idle_talk.config import load_search
    load_search(Path(__file__).resolve().parents[1] / "idle_talk.toml")
