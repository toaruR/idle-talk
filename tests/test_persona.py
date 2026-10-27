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
from idle_talk.persona import DEFAULT, NAME_PROMPT, PersonaBook, parse_personas
from idle_talk.protocol import validate_message
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

TOML = """
default = "b"
style_rules = "共通の話し方です。"
[a]
label = "エー"
prompt = "あなたはエーです。"
[b]
prompt = "あなたはビーです。"
"""


def write(path, text, t):
    path.write_text(text, encoding="utf-8")
    os.utime(path, (t, t))  # mtime の分解能に依存しないよう明示的に変える


def test_parse_keeps_order_and_default():
    personas, default = parse_personas(tomllib.loads(TOML))
    assert [(p.id, p.name, p.label) for p in personas] == [("a", "", "エー"), ("b", "", "b")]
    assert default == "b"
    assert parse_personas(tomllib.loads('[x]\nprompt = "p"\n'))[1] == "x"  # default 省略は先頭


@pytest.mark.parametrize("text", [
    "",  # 1 人もいない
    'default = "z"\n[a]\nprompt = "p"\n',  # 既定が無い
    '[a]\nprompt = ""\n',
    '[a]\nprompt = "p"\nname = ""\n',
    '[a]\nprompt = "p"\nlabel = 1\n',
    '[a]\nprompt = "p"\nspeaker = 1\n',  # 未知キー (声は画面の「声」で選ぶ)
    f'["{"x" * 65}"]\nprompt = "p"\n',
    'style_rules = 1\n[a]\nprompt = "p"\n',
    'style_rules = false\n[a]\nprompt = "p"\n',
    'style_rules = []\n[a]\nprompt = "p"\n',
    'style_rules = {}\n[a]\nprompt = "p"\n',
])
def test_parse_rejects(text):
    with pytest.raises(ValueError):
        parse_personas(tomllib.loads(text))


def test_book_without_file_uses_builtin(tmp_path):
    book = PersonaBook(tmp_path / "none.toml")
    assert book.current == DEFAULT and book.choices() == [{"id": "default", "name": DEFAULT.label}]
    assert book.system_prompt() == DEFAULT.prompt


def test_book_select_and_reload(tmp_path):
    f = tmp_path / "personas.toml"
    write(f, TOML, 1)
    book = PersonaBook(f)
    assert book.current.id == "b"
    assert book.choices() == [{"id": "a", "name": "エー"}, {"id": "b", "name": "b"}]
    book.select("a")
    assert book.system_prompt() == "あなたはエーです。\n共通の話し方です。"
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
    path = Path(__file__).resolve().parents[1] / "personas.toml"
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    personas, default = parse_personas(raw)
    assert personas and default in {p.id for p in personas}
    book = PersonaBook(path)
    assert raw["style_rules"].strip()
    assert all(p.name and p.name in p.label for p in personas)
    assert book.system_prompt() == (NAME_PROMPT.format(name=book.current.name) + book.current.prompt
                                    + "\n" + raw["style_rules"].strip())


@pytest.mark.parametrize("header", ["", 'style_rules = ""\n', 'style_rules = "  \\n  "\n'])
def test_omitted_or_empty_style_rules_uses_persona_only(tmp_path, header):
    path = tmp_path / "personas.toml"
    path.write_text(header + '[a]\nprompt = "人物設定"\n', encoding="utf-8")
    assert PersonaBook(path).system_prompt() == "人物設定"


def test_style_rules_reload_keeps_selection_and_rejects_invalid_config(tmp_path):
    path = tmp_path / "personas.toml"
    write(path, TOML, 1)
    book = PersonaBook(path)
    assert book.system_prompt() == "あなたはビーです。\n共通の話し方です。"
    book.select("a")

    updated = TOML.replace("共通の話し方です。", "新しい共通ルールです。")
    write(path, updated, 2)
    assert book.system_prompt() == "あなたはエーです。\n新しい共通ルールです。"
    assert book.current_id == "a"

    invalid = TOML.replace('style_rules = "共通の話し方です。"', 'style_rules = []')
    invalid = invalid.replace('default = "b"', 'default = "a"').replace("あなたはエーです。", "変更済み")
    write(path, invalid, 3)
    assert book.system_prompt() == "あなたはエーです。\n新しい共通ルールです。"
    assert book.default_id == "b" and book.current_id == "a"

    write(path, TOML.replace('prompt = "あなたはエーです。"', 'prompt = ""'), 4)
    assert book.system_prompt() == "あなたはエーです。\n新しい共通ルールです。"

    write(path, TOML.replace('style_rules = "共通の話し方です。"\n', ''), 5)
    assert book.system_prompt() == "あなたはエーです。"


def test_name_and_label_are_separate(tmp_path):
    path = tmp_path / "personas.toml"
    path.write_text('[a]\nname = "ミナ"\nlabel = "ミナ (先生)"\nprompt = "人物設定"\n'
                    '[b]\nname = "ソラ"\nprompt = "p"\n', encoding="utf-8")
    book = PersonaBook(path)
    assert book.choices() == [{"id": "a", "name": "ミナ (先生)"}, {"id": "b", "name": "ソラ"}]
    prompt = book.system_prompt()
    assert prompt.startswith("あなたの名前は「ミナ」です。") and prompt.endswith("人物設定")
    assert "先生" not in prompt  # 表示名は LLM に渡さない


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
            assert seen[0].startswith("あなたはエーです。\n共通の話し方です。")
        finally:
            d.close()
            store.close()

    asyncio.run(run())


def test_question_note_after_recent_question(tmp_path):
    """RATIO_Q: 直近 2 回の再生済み AI 発話に質問終わりがあれば、今回は質問で締めないよう system に足す。"""
    from idle_talk.director import NO_QUESTION_NOTE
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    d = IdleTalkDirector(store, lambda m: iter(()), lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
    try:
        assert NO_QUESTION_NOTE not in d._chat_messages()[0]["content"]
        store.add_turn(1, "ai", "温かいまま飲むと？", 1)
        store.add_turn(2, "user", "冷たくして", 1)
        assert d._chat_messages()[0]["content"].endswith(NO_QUESTION_NOTE)
        store.add_turn(3, "ai", "冷たいのもよかね。", 1)
        assert NO_QUESTION_NOTE in d._chat_messages()[0]["content"]  # 3 応答に 1 回以下
        store.add_turn(4, "ai", "うちも好きよ。", 1)
        assert NO_QUESTION_NOTE not in d._chat_messages()[0]["content"]
    finally:
        d.close()
        store.close()


def test_mode_section_in_system(tmp_path):
    """通常応答の system に今のモード (画面・カメラ・ドラマ・ブラウザ補正なし) を添える。"""
    from idle_talk.camera import Cameras, Source
    from idle_talk.director import (MODE_CAMERA_PROMPT, MODE_DRAMA_PROMPT, MODE_PC_PROMPT, MODE_SCREEN_PROMPT)
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    d = IdleTalkDirector(store, lambda m: iter(()), lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
    try:
        assert "モードです" not in d._chat_messages()[0]["content"]
        d.screen_on = True
        d.screen.mark_shown(None, "メモ帳")
        system = d._chat_messages()[0]["content"]
        assert MODE_SCREEN_PROMPT in system and "メモ帳" in system
        d.camera_on = True
        d.cameras.c = Cameras((Source("駐車場", "rtsp://x"),))
        assert MODE_CAMERA_PROMPT.format(labels="駐車場") in d._chat_messages()[0]["content"]
        d.screen_on = d.camera_on = False
        d.drama_on = True
        assert MODE_DRAMA_PROMPT in d._chat_messages()[0]["content"]
        d.drama_on = False
        d.pc_on = True
        assert MODE_PC_PROMPT in d._chat_messages()[0]["content"]
    finally:
        d.pc_on = False
        d.close()
        store.close()
