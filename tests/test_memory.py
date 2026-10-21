"""長期記憶: 抽出結果の整形・保存・プロンプト注入と Director 結線 (設計書 3.4)。"""
import asyncio
import threading

import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk import memory
from idle_talk.clients import AsrClient
from idle_talk.director import IdleTalkDirector
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

FACT = "利用者は毎朝コーヒーを飲む。"


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db", db_backup=False)
    yield s
    s.close()


def test_parse_extracted_keeps_only_subject_lines():
    text = "- 利用者は猫を飼っている。\n2. 利用者は3月に引っ越す。\nなし\n今日は晴れです。\n\n- 利用者は猫を飼っている。"
    assert memory.parse_extracted(text) == ["利用者は猫を飼っている。", "利用者は3月に引っ越す。"]


def test_parse_extracted_limits():
    assert memory.parse_extracted("利用者は" + "あ" * memory.MAX_CHARS) == []
    lines = "\n".join(f"利用者は{i}が好き。" for i in range(5))
    assert len(memory.parse_extracted(lines)) == memory.MAX_PER_UTTERANCE


def test_remember_dedupes_and_updates(store):
    memory.remember(store, [FACT], 1, 100)
    assert memory.remember(store, ["利用者は毎朝 コーヒーを飲む"], 2, 200) == []
    rows = memory.list_memories(store)
    assert len(rows) == 1 and rows[0]["updated_at"] == 200 and rows[0]["evidence_utterance_id"] == 2


def test_remember_caps_oldest_deletable(store):
    memory.add_memory(store, "利用者は固定。", 1, 0, deletable=0)
    for i in range(memory.MAX_MEMORIES):
        memory.remember(store, [f"利用者は{i}番目。"], 1, 10 + i)
    assert store.count("memories") == memory.MAX_MEMORIES
    contents = [r["content"] for r in memory.list_memories(store)]
    assert "利用者は固定。" in contents and "利用者は0番目。" not in contents


def test_memories_are_per_persona(store):
    memory.add_memory(store, "利用者は共通。", 1, 1)  # v2 以前の記憶 (persona NULL) は全員に見える
    memory.remember(store, ["利用者は猫好き。"], 1, 10, "a")
    memory.remember(store, ["利用者は犬好き。"], 2, 20, "b")
    assert "猫" in memory.memory_prompt(store, "a") and "犬" not in memory.memory_prompt(store, "a")
    assert "犬" in memory.memory_prompt(store, "b") and "共通" in memory.memory_prompt(store, "b")
    assert memory.remember(store, ["利用者は猫好き"], 3, 30, "b") != []  # 別のペルソナには別に覚える
    for i in range(memory.MAX_MEMORIES):  # 上限はペルソナごと
        memory.remember(store, [f"利用者は{i}番目。"], 1, 100 + i, "a")
    assert "犬" in memory.memory_prompt(store, "b")
    assert len(memory.list_memories(store, "b")) == 3


def test_memory_prompt(store):
    assert memory.memory_prompt(store) == ""
    memory.remember(store, [FACT], 1, 100)
    assert f"- {FACT}" in memory.memory_prompt(store)


def make(tmp_path, gate=None):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    chats: list[list[dict]] = []
    extracts: list[list[dict]] = []

    def llm(messages):
        if messages[0]["content"] == memory.EXTRACT_PROMPT:
            extracts.append(messages)
            if gate is not None:
                gate.wait(10)
            yield f"- {FACT}\n"
            return
        chats.append(messages)
        yield "いいですね。"

    asr = AsrClient(_asr(clock))
    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(), asr,
                         clock=clock, now_ms=lambda: 1000)
    b = MockBrowser()
    b.attach(d.core)
    return d, b, store, chats, extracts


def _asr(clock):
    def send(req, timeout):
        return 200, {"utterance_id": req["utterance_id"], "text": "毎朝コーヒーを飲んでいます",
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}
    return send


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def test_extracted_after_response_and_injected(tmp_path):
    async def run():
        d, b, store, chats, extracts = make(tmp_path)
        try:
            b.speak()
            await until(lambda: store.count("memories") == 1)
            assert FACT not in chats[0][0]["content"]  # 抽出は応答の再生完了後
            assert "毎朝コーヒーを飲んでいます" in extracts[0][1]["content"]
            assert d._mem_pending == []
            b.speak()
            await until(lambda: len(chats) == 2)
            assert FACT in chats[1][0]["content"]
        finally:
            store.close()
    asyncio.run(run())


def test_user_speech_cancels_extraction_and_retries(tmp_path):
    async def run():
        gate = threading.Event()
        d, b, store, chats, extracts = make(tmp_path, gate)
        try:
            b.speak()
            await until(lambda: len(extracts) == 1)
            b.speak()  # 抽出中に利用者が話す: 取消して結果は保存しない
            assert d._mem_task.cancelled() or d._mem_stop.is_set()
            gate.set()
            await until(lambda: len(extracts) == 2)  # 次の応答完了後に未処理分をまとめて再試行
            await until(lambda: store.count("memories") == 1)
            assert len(extracts[1][1]["content"].splitlines()) == 2
        finally:
            gate.set()
            store.close()
    asyncio.run(run())


def test_short_utterance_not_extracted(tmp_path):
    async def run():
        d, b, store, chats, extracts = make(tmp_path)
        d.ingress.asr = AsrClient(mock_asr_send(FakeClock()))  # 「こんにちは」(5 文字)
        try:
            b.speak()
            await until(lambda: len(chats) == 1 and d.fsm.ctx.state.value == "idle")
            await asyncio.sleep(0.05)
            assert extracts == [] and store.count("memories") == 0
        finally:
            store.close()
    asyncio.run(run())
