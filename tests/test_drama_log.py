"""[drama].log_prompt: 一言・感想・要約に LLM へ渡した台詞とメモを talk.log に残す (既定は残さない)。"""
import asyncio
import json
import logging

import pytest
from test_drama import SETTINGS, drama_on, drama_speak, make, until

from idle_talk.config import load_drama
from idle_talk.drama import DramaListener
from idle_talk.history import talk_log
from idle_talk.vad import EnergyVad


@pytest.fixture
def talk_lines():
    lines: list[str] = []

    class H(logging.Handler):
        def emit(self, record):
            lines.append(record.getMessage())
    h = H()
    talk_log.addHandler(h)
    level, talk_log.level = talk_log.level, logging.INFO
    yield lines
    talk_log.removeHandler(h)
    talk_log.setLevel(level)


def test_load_log_prompt(tmp_path):
    p = tmp_path / "s.toml"
    assert load_drama(p).log_prompt is False
    p.write_text("[drama]\nlog_prompt = true\n", encoding="utf-8")
    assert load_drama(p).log_prompt is True
    p.write_text("[drama]\nlog_prompt = 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_drama(p)


async def react(d, b):
    drama_on(d, b)
    drama_speak(b)
    drama_speak(b)
    await until(lambda: len(d.drama.lines) == 2)
    d.clock_.advance(60)
    assert d.spontaneous_tick() == "drama"
    await until(lambda: not d.spontaneous.pending)


def test_reaction_scene_logged_when_on(tmp_path, talk_lines):
    async def run():
        d, b, _ = make(tmp_path, ["まさか！"], ["犯人はお前だ。", "違う、俺じゃない。"])
        d.drama = DramaListener(SETTINGS._replace(log_prompt=True), EnergyVad(), 200, d.clock_)
        await react(d, b)
        assert "DRAMA(一言): 犯人はお前だ。 / 違う、俺じゃない。" in talk_lines  # 複数行は 1 行にまとめる
        assert "AI: まさか！" in talk_lines
    asyncio.run(run())


def test_nothing_logged_when_off(tmp_path, talk_lines):
    async def run():
        d, b, _ = make(tmp_path, ["まさか！"], ["犯人はお前だ。", "違う、俺じゃない。"])
        await react(d, b)
        assert not any(line.startswith("DRAMA") for line in talk_lines)
    asyncio.run(run())


def test_digest_and_finale_logged(tmp_path, talk_lines):
    async def run():
        d, b, _ = make(tmp_path, ["兄が弟を疑った。", "最後は泣けたね。"], ["犯人はお前だ、そうだろう。", "違う、俺じゃない。"])
        d.drama = DramaListener(SETTINGS._replace(digest_chars=10, finale_gap_s=120, finale_min_chars=10,
                                                  log_prompt=True), EnergyVad(), 200, d.clock_)
        d.drama_vision = True
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.spontaneous_tick()
        await until(lambda: d.drama.notes)
        assert "DRAMA(要約・台詞): 犯人はお前だ、そうだろう。" in talk_lines
        assert "DRAMA(要約・メモ): 兄が弟を疑った。" in talk_lines
        drama_speak(b)
        await until(lambda: len(d.drama.lines) == 2)
        d.drama.mark_reacted()
        d.clock_.advance(120)
        assert d.spontaneous_tick() == "drama_finale"
        await until(lambda: any(m["t"] == "snap" for m in b.texts))
        snap = next(m for m in b.texts if m["t"] == "snap")
        d.core.on_text(b, json.dumps({"t": "snap", "id": snap["id"], "jpeg": "A" * 4096}))
        await until(lambda: not d.spontaneous.pending)
        assert "DRAMA(感想・メモ): 兄が弟を疑った。" in talk_lines
        assert "DRAMA(感想・台詞): 違う、俺じゃない。 [画像あり 3 KB]" in talk_lines  # 画像そのものは残さない
    asyncio.run(run())
