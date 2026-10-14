"""ペルソナ (設計書 6.6): personas.toml の読み込み・切り替え・system プロンプトへの反映。"""
import asyncio
import json
import os
import tomllib
from pathlib import Path

import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.director import IdleTalkDirector
from idle_talk.persona import DEFAULT, STYLE_RULES, PersonaBook, parse_personas
from idle_talk.protocol import validate_message
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

TOML = """
default = "b"
[a]
name = "エー"
prompt = "あなたはエーです。"
[b]
prompt = "あなたはビーです。"
"""


def write(path, text, t):
    path.write_text(text, encoding="utf-8")
    os.utime(path, (t, t))  # mtime の分解能に依存しないよう明示的に変える


def test_parse_keeps_order_and_default():
    personas, default = parse_personas(tomllib.loads(TOML))
    assert [(p.id, p.name) for p in personas] == [("a", "エー"), ("b", "b")]
    assert default == "b"
    assert parse_personas(tomllib.loads('[x]\nprompt = "p"\n'))[1] == "x"  # default 省略は先頭


@pytest.mark.parametrize("text", [
    "",  # 1 人もいない
    'default = "z"\n[a]\nprompt = "p"\n',  # 既定が無い
    '[a]\nprompt = ""\n',
    '[a]\nprompt = "p"\nspeaker = 1\n',  # 未知キー (声は画面の「声」で選ぶ)
    f'["{"x" * 65}"]\nprompt = "p"\n',
])
def test_parse_rejects(text):
    with pytest.raises(ValueError):
        parse_personas(tomllib.loads(text))


def test_book_without_file_uses_builtin(tmp_path):
    book = PersonaBook(tmp_path / "none.toml")
    assert book.current == DEFAULT and book.choices() == [{"id": "default", "name": DEFAULT.name}]
    assert book.system_prompt() == DEFAULT.prompt + "\n" + STYLE_RULES


def test_book_select_and_reload(tmp_path):
    f = tmp_path / "personas.toml"
    write(f, TOML, 1)
    book = PersonaBook(f)
    assert book.current.id == "b"
    assert book.choices() == [{"id": "a", "name": "エー"}, {"id": "b", "name": "b"}]
    book.select("a")
    assert book.system_prompt().startswith("あなたはエーです。\n")
    with pytest.raises(ValueError):
        book.select("zz")
    assert book.current.id == "a"
    write(f, TOML.replace("あなたはエーです。", "あなたは新エーです。"), 2)  # 保存すれば次の応答から効く
    assert book.current.prompt == "あなたは新エーです。"
    write(f, "[a]\nprompt = ''\n", 3)  # 不正: 前の内容を保つ
    assert book.current.prompt == "あなたは新エーです。"
    write(f, '[b]\nprompt = "ビー"\n', 4)  # 選んでいたものが消えたら既定へ
    assert book.current.id == "b"


def test_shipped_personas_file_is_valid():
    personas, default = parse_personas(tomllib.loads(
        (Path(__file__).resolve().parents[1] / "personas.toml").read_text(encoding="utf-8")))
    assert personas and default in {p.id for p in personas}


def test_ui_persona_schema():
    assert validate_message({"t": "ui", "persona": "a"})
    assert not validate_message({"t": "ui", "persona": ""})
    assert not validate_message({"t": "ui", "persona": 1})


def test_ui_persona_switches_system_prompt(tmp_path):
    async def run():
        clock = FakeClock()
        store = Store(tmp_path / "t.db", db_backup=False)
        seen = []

        def llm(messages):
            seen.append(messages[0]["content"])
            yield "はい。"

        f = tmp_path / "personas.toml"
        write(f, TOML, 1)
        d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                             AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
        d.personas = PersonaBook(f)
        b = MockBrowser()
        b.attach(d.core)
        try:
            d.core.on_text(b, json.dumps({"t": "ui", "persona": "a"}))
            d.core.on_text(b, json.dumps({"t": "ui", "persona": "zz"}))
            assert d.personas.current.id == "a" and [m["code"] for m in b.of("error")] == [5006]
            b.speak()
            end = asyncio.get_running_loop().time() + 5
            while not seen:
                assert asyncio.get_running_loop().time() < end, "timeout"
                await asyncio.sleep(0.01)
            assert seen[0].startswith("あなたはエーです。\n" + STYLE_RULES)
        finally:
            d.close()
            store.close()

    asyncio.run(run())
