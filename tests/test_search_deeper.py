"""深掘り検索の制御・資料の引継ぎ・中止を外部サービスなしで検証する。"""

import asyncio

import pytest

from idle_talk.pipeline import ResponsePipeline
from idle_talk.web_search import (
    RESEARCH_ROUND_MAX_CHARS,
    SEARCH_INCOMPLETE_NOTICE,
    SEARCH_MAX_ROUNDS,
    SEARCH_NOTICE,
    research_messages,
)

MESSAGES = [{"role": "system", "content": "PERSONA"},
            {"role": "user", "content": "青葉ホールの移転後の開館時間は？"}]


def result(title, snippet, url="https://example.test/"):
    return [{"title": title, "snippet": snippet, "url": url}]


def pipeline(outputs):
    calls, audio, events = [], [], []

    def llm(messages):
        calls.append(messages)
        return iter(outputs[len(calls) - 1])

    p = ResponsePipeline(llm, lambda text: b"wav", lambda: None,
                         lambda ep, u, c, text, wav: audio.append(text), events.append)
    return p, calls, audio, events


def respond(p, search):
    async def run():
        await p.respond(MESSAGES, search=search)
        await asyncio.sleep(0)
    asyncio.run(run())


def test_followup_search_preserves_original_question_and_prior_evidence():
    p, calls, audio, events = pipeline([
        ["検索: 青葉ホール 移転\n"],
        ["検", "索", ": 青葉文化センター 開館時間\n", "読まない続き"],
        ["移転後は、", "午前9時からです。"],
    ])
    queries = []

    def search(query):
        queries.append(query)
        if len(queries) == 1:
            return result("移転のお知らせ", "移転後の名称は青葉文化センター。", "https://a.test/")
        return result("利用案内", "開館は午前9時。", "https://b.test/")

    respond(p, search)
    assert queries == ["青葉ホール 移転", "青葉文化センター 開館時間"]
    assert calls[2][-1]["content"].startswith(MESSAGES[-1]["content"])
    assert "移転後の名称は青葉文化センター。" in calls[2][-1]["content"]
    assert "開館は午前9時。" in calls[2][-1]["content"]
    assert audio == [SEARCH_NOTICE, "移転後は、", "午前9時からです。"]
    assert [m["active"] for m in events if m["t"] == "search"] == [True, False, True, False]
    assert {item["url"] for item in p.links} == {"https://a.test/", "https://b.test/"}
    assert calls[1][0]["content"].startswith("PERSONA\n")
    assert MESSAGES[0]["content"] == "PERSONA"


def test_empty_results_can_trigger_rephrased_search():
    p, calls, audio, _ = pipeline([["検索: 青葉ホール\n"],
                                   ["検索: 青葉 文化施設 移転先\n"], ["移転先が分かりました。"]])
    queries = []

    def search(query):
        queries.append(query)
        return [] if len(queries) == 1 else result("移転", "青葉文化センターへ移転。")

    respond(p, search)
    assert len(queries) == 2
    assert "青葉ホール" in calls[2][-1]["content"]
    assert audio == [SEARCH_NOTICE, "移転先が分かりました。"]


def test_sufficient_results_finish_without_extra_search():
    p, calls, audio, _ = pipeline([["検索: 青葉ホール 開館時間\n"], ["午前9時からです。"]])
    queries = []
    respond(p, lambda query: queries.append(query) or result("利用案内", "午前9時から。"))
    assert len(queries) == 1 and len(calls) == 2
    assert audio == [SEARCH_NOTICE, "午前9時からです。"]


@pytest.mark.parametrize("next_query", ["検索:  ALPHA   beta\n", "検索:\n"])
def test_repeated_or_empty_followup_stops_without_reading_mark(next_query):
    p, calls, audio, _ = pipeline([["検索: alpha beta\n"], [next_query]])
    queries = []
    respond(p, lambda query: queries.append(query) or [])
    assert queries == ["alpha beta"] and len(calls) == 2
    assert audio == [SEARCH_NOTICE, SEARCH_INCOMPLETE_NOTICE]


def test_search_budget_is_enforced_even_if_model_keeps_requesting_searches():
    p, calls, audio, _ = pipeline([[f"検索: q{i}\n"] for i in range(SEARCH_MAX_ROUNDS + 1)])
    queries = []
    respond(p, lambda query: queries.append(query) or [])
    assert queries == [f"q{i}" for i in range(SEARCH_MAX_ROUNDS)]
    assert len(calls) == SEARCH_MAX_ROUNDS + 1
    assert audio == [SEARCH_NOTICE, SEARCH_INCOMPLETE_NOTICE]


def test_final_round_can_answer_with_partial_findings():
    p, calls, audio, _ = pipeline([["検索: q1\n"], ["検索: q2\n"], ["検索: q3\n"],
                                   ["移転先は分かりましたが、", "開館時間は確認できませんでした。"]])
    respond(p, lambda query: [])
    assert len(calls) == 4
    assert "残り回数: 0回" in calls[-1][0]["content"]
    assert audio == [SEARCH_NOTICE, "移転先は分かりましたが、", "開館時間は確認できませんでした。"]


def test_cancel_during_followup_skips_answer_and_next_search():
    p, calls, audio, events = pipeline([["検索: q1\n"], ["検索: q2\n"], ["古い返答。"]])
    queries = []

    def search(query):
        queries.append(query)
        if len(queries) == 2:
            p.epoch.bump()
        return []

    respond(p, search)
    assert queries == ["q1", "q2"] and len(calls) == 2
    assert audio == [SEARCH_NOTICE]
    assert [m["active"] for m in events if m["t"] == "search"] == [True, False, True, False]


def test_cancel_while_model_requests_followup_does_not_start_search():
    audio, calls, queries = [], [], []
    p = None

    def llm(messages):
        calls.append(messages)
        if len(calls) == 2:
            p.epoch.bump()
            yield "検索: q2\n"
        else:
            yield "検索: q1\n"

    p = ResponsePipeline(llm, lambda text: b"wav", lambda: None,
                         lambda ep, u, c, text, wav: audio.append(text), lambda msg: None)
    respond(p, lambda query: queries.append(query) or [])
    assert queries == ["q1"] and audio == [SEARCH_NOTICE]


def test_failed_followup_clears_search_indicator():
    p, _, _, events = pipeline([["検索: q1\n"], ["検索: q2\n"]])

    def search(query):
        if query == "q2":
            raise OSError("offline")
        return []

    respond(p, search)
    assert [m["active"] for m in events if m["t"] == "search"] == [True, False, True, False]
    assert any(m["t"] == "error" for m in events)


def test_research_context_is_bounded_and_keeps_each_search():
    searches = [(f"query{i}", result(f"title{i}", "x" * 20000)) for i in range(SEARCH_MAX_ROUNDS)]
    out = research_messages(MESSAGES, searches, 0)
    assert len(out[-1]["content"]) < SEARCH_MAX_ROUNDS * (RESEARCH_ROUND_MAX_CHARS + 100) + 200
    assert all(f"query{i}" in out[-1]["content"] for i in range(SEARCH_MAX_ROUNDS))


def test_followup_stream_closes_before_another_search():
    closed = []
    calls = []

    def llm(messages):
        calls.append(messages)
        if len(calls) == 1:
            yield "検索: q1\n"
        elif len(calls) == 2:
            try:
                yield "検索: q2\n"
                raise AssertionError("検索指示の後は生成を続けない")
            finally:
                closed.append(True)
        else:
            yield "答えです。"

    p = ResponsePipeline(llm, lambda text: b"wav", lambda: None,
                         lambda *args: None, lambda msg: None)

    def search(query):
        if query == "q2":
            assert closed == [True]
        return []

    respond(p, search)
    assert len(calls) == 3
