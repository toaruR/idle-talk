import asyncio
import logging
import re
import threading
import time
from pathlib import Path

from idle_talk.clause import ClauseSplitter
from idle_talk.clients import TtsFailed
from idle_talk.llm_supervisor import FALLBACK_PHRASE
from idle_talk.pipeline import ResponsePipeline


def make(llm, tts=None, order=None):
    order = order if order is not None else []
    audio, emits = [], []

    def default_tts(text):
        return b"wav:" + text.encode()

    def cancel_tts():
        order.append("cancel_tts")

    def emit(m):
        order.append(("emit", m["t"]))
        emits.append(m)

    p = ResponsePipeline(llm, tts or default_tts, cancel_tts,
                         lambda ep, u, c, t, w: audio.append((ep, u, c, t, w)), emit)
    return p, audio, emits, order


def test_clause_first_synthesizable_clause_goes_to_tts():
    sp = ClauseSplitter()
    assert sp.feed("こんに") == []
    assert sp.feed("ちは、今日") == ["こんにちは、"]
    assert sp.feed("は晴れ。") == ["今日は晴れ。"]


def test_clause_punct_only_not_synthesizable():
    sp = ClauseSplitter()
    assert sp.feed("、。") == []
    assert sp.feed("はい") == []
    assert sp.flush() == ["、。はい"]


def test_clause_no_tts_before_clause_confirmed():
    calls = []
    p, audio, _, _ = make(lambda m: iter(["こんに", "ちは"]),
                          tts=lambda t: calls.append(t) or b"x")
    seen = []

    def llm(m):
        for tok in ["こんに", "ちは"]:
            seen.append(len(calls))  # 節確定前の TTS 呼び出しは 0 件
            yield tok
        yield "、元気"
        seen.append(len(calls))  # 「こんにちは、」確定で 1 件
        yield "です。"

    p._llm = llm
    asyncio.run(p.respond([]))
    assert seen[:2] == [0, 0]
    assert calls == ["こんにちは、", "元気です。"]
    assert [a[3] for a in audio] == calls


def test_serial_tts_max_concurrency_one():
    state = {"cur": 0, "max": 0}
    lock = threading.Lock()

    def tts(text):
        with lock:
            state["cur"] += 1
            state["max"] = max(state["max"], state["cur"])
        time.sleep(0.03)
        with lock:
            state["cur"] -= 1
        return b"x"

    p, audio, _, _ = make(None, tts=tts)

    async def run():
        await asyncio.gather(p.play_utterance(["あ。", "い。"]), p.play_utterance(["う。", "え。"]),
                             p.play_utterance(["お。"]))

    asyncio.run(run())
    assert state["max"] == 1
    assert len(audio) == 5


def test_cancel_order_before_blocked_llm_completes():
    release = threading.Event()
    order = []

    def llm(m):
        order.append("llm_blocked")
        release.wait(5)
        order.append("llm_done")
        yield "遅い応答。"

    p, audio, _emits, _ = make(llm, order=order)

    async def run():
        task = asyncio.create_task(p.respond([]))
        await asyncio.sleep(0.1)
        assert "llm_blocked" in order
        await p.cancel()
        order.append("cancel_returned")
        release.set()
        await task

    asyncio.run(run())
    assert order.index("cancel_tts") < order.index("llm_done")
    assert order.index("cancel_returned") < order.index("llm_done")
    assert audio == []


def test_stale_epoch_result_discarded(caplog):
    gate = threading.Event()
    cancels = []
    audio = []
    emits = []

    def tts(text):
        gate.wait(5)
        return b"late"

    p = ResponsePipeline(None, tts, lambda: cancels.append(1),
                         lambda *a: audio.append(a), emits.append)

    async def run():
        task = asyncio.create_task(p.play_utterance(["こんにちは。"]))
        await asyncio.sleep(0.1)
        await p.cancel()
        gate.set()
        await task

    with caplog.at_level(logging.INFO, logger="idle_talk.pipeline"):
        asyncio.run(run())
    assert len(cancels) == 1
    assert audio == []
    assert p.discarded == 1
    assert len([r for r in caplog.records if "discard" in r.getMessage()]) == 1
    assert p.epoch.current == 1


def test_llm_failure_5001_with_fallback_phrase():
    def llm(m):
        raise ConnectionRefusedError("killed")
        yield

    p, audio, emits, _ = make(llm)
    asyncio.run(p.respond([]))
    assert [e["code"] for e in emits if e["t"] == "error"] == [5001]
    assert audio[0][3] == FALLBACK_PHRASE


def test_tts_failure_5002_skips_sentence_text_only():
    def tts(text):
        if "壊れ" in text:
            raise TtsFailed("boom")
        return b"ok"

    p, audio, emits, _ = make(lambda m: iter(["壊れた。", "次の文。"]), tts=tts)
    asyncio.run(p.respond([]))
    assert [a[3] for a in audio] == ["次の文。"]
    assert [e["code"] for e in emits if e["t"] == "error"] == [5002]
    texts = [e for e in emits if e["t"] == "text"]
    assert len(texts) == 1 and texts[0]["text"] == "壊れた。"


def test_single_play_entry_point():
    src = (Path(__file__).parent.parent / "src" / "idle_talk" / "pipeline.py").read_text(encoding="utf-8")
    assert len(re.findall(r"def play_utterance\(", src)) == 1


def test_all_kinds_go_through_play_utterance():
    p, audio, _, _ = make(lambda m: iter(["通常。"]))
    calls = []
    orig = p.play_utterance

    async def spy(c):
        calls.append(1)
        return await orig(c)

    p.play_utterance = spy

    async def run():
        await p.respond([])
        await p.play_utterance(["自発発話です。"])
        await p.play_utterance(["うん"])

    asyncio.run(run())
    assert len(calls) == 3  # respond も含め全て play_utterance 経由
    assert len(audio) == 3
