import asyncio
from urllib.parse import quote

from idle_talk.browser_page import (PAGE_FAIL, PAGE_NOTICE, SUMMARY_NOTICE, PageReader, body_excerpt, normalize_url,
                                    page_messages, page_title, summary_url, wants_page, wants_summary)
from idle_talk.pipeline import ResponsePipeline
from idle_talk.web_search import READER_URL


def test_wants_page():
    for t in ("このページどう思う？", "この記事の感想を聞かせて", "今見てるページを要約して",
              "このサイトについて教えて", "記事を読んでどう思う", "この ページ どう"):
        assert wants_page(t), t
    for t in ("今日はどう思う？", "ページって何", "この前の話どう思う", "記事を書いた", ""):
        assert not wants_page(t), t


def test_page_title():
    assert page_title("Jev 試した - Qiita - Google Chrome") == "Jev 試した - Qiita"
    assert page_title("ニュース - Microsoft\u200b Edge") == "ニュース"
    assert page_title("talk.log - idle-talk - Visual Studio Code") == ""
    assert page_title("Loop - obsidian - Obsidian 1.14.4") == ""


def test_normalize_url():
    assert normalize_url("qiita.com/a/items/1") == "https://qiita.com/a/items/1"
    assert normalize_url("http://example.com/x") == "http://example.com/x"
    assert normalize_url("localhost:8000/") == "https://localhost:8000/"
    for v in ("", "chrome://newtab/", "edge://settings", "file:///C:/a.html", "検索 する語", "Work モードで作成", "newtab"):
        assert normalize_url(v) == "", v


def test_body_excerpt_drops_short_lines():
    text = "search\nLogin\nTrend\n" + "これは記事の本文の一行目で、十分に長い文章です。\nflag\n" + "二行目も記事の本文で、同じく長めの文章になっています。\n"
    assert body_excerpt(text) == "これは記事の本文の一行目で、十分に長い文章です。 / 二行目も記事の本文で、同じく長めの文章になっています。"
    assert len(body_excerpt("あ" * 50 + "\n", max_chars=10)) == 10


def test_body_excerpt_starts_at_title():
    text = ("求人サイトではエンジニアのあなたにマッチした求人が見つかります。\nJevにウォーリーを探させたら0.4秒で見つけやがった!!\n"
            "本文は画像の検証の話で、二十字を超えています。\n")
    out = body_excerpt(text, "Jevにウォーリーを探させたら0.4秒で見つけやがった!! #AI - Qiita")
    assert out.startswith("Jevにウォーリー") and "求人" not in out
    assert body_excerpt(text, "無関係なタイトル - 例").startswith("求人サイト")  # 見つからなければ先頭から


def reader(windows, addresses, pages, calls=None):
    calls = calls if calls is not None else []

    def fetch(url, headers=None, timeout=None):
        calls.append((url, headers, timeout))
        body = pages[url]
        if isinstance(body, Exception):
            raise body
        return body.encode()

    return PageReader(fetch, ("1Password",), lambda: windows, lambda h: addresses[h]), calls


def test_page_reader_skips_self_and_excluded():
    body = "記事の本文です。ここは二十字以上ある長い行になっています。"
    r, calls = reader([(1, "idle-talk"), (2, "金庫 - 1Password"), (3, "新しいタブ"), (4, "記事")],
                      {1: "127.0.0.1:8000/", 2: "my.1password.com", 3: "chrome://newtab/", 4: "example.com/a"},
                      {READER_URL + "https://example.com/a": body})
    assert r() == {"title": "記事", "url": "https://example.com/a", "text": body}
    assert calls[0][1] == {"X-Return-Format": "text"} and calls[0][2] > 8  # 検索より長く待つ


def test_page_reader_none_and_read_failure():
    assert reader([], {}, {})[0]() is None
    r, _ = reader([(4, "記事")], {4: "example.com/a"}, {READER_URL + "https://example.com/a": OSError("timeout")})
    assert r() == {"title": "記事", "url": "https://example.com/a", "text": ""}


def test_page_messages_text_and_image():
    msgs = [{"role": "system", "content": "S"}] + [{"role": "user", "content": f"u{i}"} for i in range(8)]
    out = page_messages(msgs, {"title": "T", "text": "本文"}, 4)
    assert out[0] == msgs[0] and len(out) == 5
    assert out[-1]["content"].startswith("u7\n\n") and "「T」" in out[-1]["content"] and out[-1]["content"].endswith("本文")
    img = page_messages(msgs, {"title": "T"}, 4, "data:image/png;base64,AA")
    parts = img[-1]["content"]
    assert parts[0]["type"] == "text" and parts[0]["text"].startswith("u7") and "画面" in parts[0]["text"]
    assert parts[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}


def run(page, shot=None, user="このページどう思う？", llm_out=("面白い記事だね。",)):
    seen, spoken = [], []
    emits = run.emits = []

    def llm(m):
        seen.append(m)
        return iter(llm_out)

    p = ResponsePipeline(llm, lambda t: b"w", lambda: None, lambda ep, u, c, t, w: spoken.append(t), emits.append)
    asyncio.run(p.respond([{"role": "system", "content": "S"}, {"role": "user", "content": user}],
                          search=lambda q: [], page=page, shot=shot))
    return p, seen, spoken


def test_pipeline_reads_page():
    page = {"title": "記事", "url": "https://example.com/a", "text": "本文"}
    p, seen, spoken = run(lambda: page)
    assert spoken == [PAGE_NOTICE, "面白い記事だね。"]
    assert len(seen) == 1 and seen[0][-1]["content"].endswith("本文")  # 検索の指示を付けた 1 回目は呼ばない
    assert p.links == [{"title": "記事", "url": "https://example.com/a"}]
    assert {"t": "links", "items": p.links} in run.emits  # 読んだ URL を画面の URL 欄に出す


def test_pipeline_falls_back_to_screen():
    p, seen, spoken = run(lambda: {"title": "記事", "url": "https://example.com/a", "text": ""},
                          shot=lambda: ("data:image/png;base64,AA", "記事 - Google Chrome"))
    assert spoken == [PAGE_NOTICE, "面白い記事だね。"]
    parts = seen[0][-1]["content"]
    assert "「記事」" in parts[0]["text"] and parts[1]["image_url"]["url"] == "data:image/png;base64,AA"
    assert p.links == [{"title": "記事", "url": "https://example.com/a"}]
    _, seen, _ = run(lambda: None, shot=lambda: ("data:image/png;base64,AA", "メモ帳"))
    assert "「メモ帳」" in seen[0][-1]["content"][0]["text"]


def test_pipeline_page_not_found():
    _, seen, spoken = run(lambda: None)
    assert spoken == [PAGE_NOTICE, PAGE_FAIL] and seen == []
    assert not [m for m in run.emits if m["t"] == "links"]
    _, seen, spoken = run(lambda: None, shot=lambda: None)
    assert spoken == [PAGE_NOTICE, PAGE_FAIL] and seen == []


def test_pipeline_ignores_page_without_request():
    called = []
    _, seen, spoken = run(lambda: called.append(1), user="今日はいい天気だね")
    assert called == [] and spoken == ["面白い記事だね。"] and len(seen) == 1


def test_wants_summary_and_url():
    assert wants_summary("このページ要約して") and wants_summary("この記事をまとめて")
    assert not wants_summary("このページどう思う？") and not wants_summary("要約して")
    assert summary_url("https://example.com/a?b=1") == (
        "https://chatgpt.com/?q=" + quote("https://example.com/a?b=1 を日本語、小学生にも分かるように箇条書きで要約して", safe=""))


def test_pipeline_opens_chatgpt_summary():
    opened = []
    r, calls = reader([(4, "記事")], {4: "example.com/a"}, {})
    r._open = opened.append
    p, seen, spoken = run(r, user="このページ要約して")
    assert opened == [summary_url("https://example.com/a")]
    assert spoken == [SUMMARY_NOTICE] and seen == [] and calls == []  # LLM も r.jina.ai も呼ばない
    assert {"t": "links", "items": [{"title": "記事", "url": "https://example.com/a"}]} in run.emits
    r, _ = reader([], {}, {})
    r._open = opened.append
    _, _, spoken = run(r, user="このページ要約して")
    assert spoken == [PAGE_FAIL] and len(opened) == 1


def test_load_search_read_window(tmp_path):
    from idle_talk.config import load_search
    f = tmp_path / "s.toml"
    f.write_text("[search]\non = true\n", encoding="utf-8")
    assert load_search(f).read_window is True
    f.write_text("[search]\nread_window = false\n", encoding="utf-8")
    assert load_search(f).read_window is False
    f.write_text("[search]\nread_window = 1\n", encoding="utf-8")
    try:
        load_search(f)
    except ValueError:
        pass
    else:
        raise AssertionError("read_window must be bool")
