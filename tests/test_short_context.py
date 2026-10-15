"""文脈の無い短い認識結果 (物音の聞き違い) を捨てる: [asr].short_chars / short_reply_s。"""
import asyncio

import pytest
from test_integration_interrupt import asr_settled, make, state, until

from idle_talk.config import Asr, load_asr
from idle_talk.director import SHORT_KIND


def user_rows(store):
    return [r[0] for r in store.query("SELECT text FROM turns WHERE role='user'")]


def short_events(store):
    return store.query("SELECT COUNT(*) FROM events WHERE kind=?", (SHORT_KIND,))[0][0]


def test_short_without_ai_voice_is_discarded(tmp_path):
    async def run():
        d, b, store = make(tmp_path, texts=["はい", "ハハ", "こんにちは"])
        b.speak()  # 食器の音が「はい」に: AI はまだ話していない
        await asr_settled(d)
        b.speak()
        await asr_settled(d)
        assert user_rows(store) == [] and short_events(store) == 2 and state(d) == "idle"
        assert d.calls["n"] == 0  # 応答しない
        b.speak()  # 3 字以上は文脈がなくても通る
        await asr_settled(d)
        await until(lambda: d.calls["n"] == 1)
        assert user_rows(store) == ["こんにちは"]
        store.close()
    asyncio.run(run())


def test_short_reply_right_after_ai_voice_passes(tmp_path):
    async def run():
        d, b, store = make(tmp_path, texts=["こんにちは", "うん", "はい"])
        b.speak()
        await asr_settled(d)
        await until(lambda: state(d) == "idle" and d.calls["n"] == 1 and len(b.chunks) == 2)  # AI が話し終えた
        d.clock_.advance(20)
        b.speak()  # 直後 (short_reply_s ちょうど) の返事は通す
        await asr_settled(d)
        assert user_rows(store)[-1] == "うん"
        await until(lambda: state(d) == "idle" and d.calls["n"] == 2)
        d.clock_.advance(20.1)
        b.speak()  # short_reply_s を過ぎた短い結果は捨てる
        await asr_settled(d)
        assert user_rows(store)[-1] == "うん" and short_events(store) == 1
        store.close()
    asyncio.run(run())


def test_short_disabled(tmp_path):
    async def run():
        d, b, store = make(tmp_path, texts=["はい"])
        d.short_chars = 0
        b.speak()
        await asr_settled(d)
        assert user_rows(store) == ["はい"] and short_events(store) == 0
        store.close()
    asyncio.run(run())


def test_load_asr_short_settings(tmp_path):
    f = tmp_path / "s.toml"
    assert (Asr().short_chars, Asr().short_reply_s) == (2, 20)
    f.write_text("[asr]\nshort_chars = 0\nshort_reply_s = 60\n", encoding="utf-8")
    assert load_asr(f)[-2:] == (0, 60)
    for bad in ("short_chars = -1", "short_chars = 11", 'short_reply_s = "20"', "short_reply_s = 3601"):
        f.write_text(f"[asr]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_asr(f)
