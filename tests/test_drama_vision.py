"""ドラマモードの画面 (設計書 6.5): 一言・感想の直前に画面 1 枚を求め、LLM に画像として添える。"""
import asyncio
import json

from test_drama import SETTINGS, drama_on, drama_speak, make, until

from idle_talk import director as director_mod
from idle_talk.clients import build_llm_request
from idle_talk.drama import DramaListener
from idle_talk.protocol import validate_message
from idle_talk.vad import EnergyVad

JPEG = "QUJD"  # base64 の中身は見ない


def test_snap_and_image_content_schemas():
    assert validate_message({"t": "snap", "id": 1, "jpeg": JPEG})
    assert validate_message({"t": "snap", "id": 1})  # 映像が無いときは jpeg を省く
    assert not validate_message({"t": "snap", "jpeg": JPEG})
    assert not validate_message({"t": "snap", "id": 1, "jpeg": "x" * (1048576 + 1)})
    parts = [{"type": "text", "text": "見て"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}}]
    assert build_llm_request([{"role": "user", "content": parts}])["messages"][0]["content"] == parts
    for bad in ([], [{"type": "image_url", "image_url": {"url": "http://example.com/a.jpg"}}], [{"type": "audio"}]):
        try:
            build_llm_request([{"role": "user", "content": bad}])
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")


async def react(d, b):
    drama_on(d, b)
    drama_speak(b)
    await until(lambda: d.drama.lines)
    d.clock_.advance(60)
    assert d.spontaneous_tick() == "drama"


def answer_snap(d, b, jpeg=JPEG):
    snap = next(m for m in b.texts if m["t"] == "snap")
    d.core.on_text(b, json.dumps({"t": "snap", "id": snap["id"], **({"jpeg": jpeg} if jpeg else {})}))


def test_reaction_attaches_screen(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["え、泣いてる！"], ["犯人はお前だ、そうだろう。"])
        d.drama_vision = True
        await react(d, b)
        await until(lambda: any(m["t"] == "snap" for m in b.texts))
        answer_snap(d, b)
        await until(lambda: not d.spontaneous.pending)
        content = d.seen_[0][1]["content"]
        assert content[0]["type"] == "text" and "犯人はお前だ" in content[0]["text"] and "画像" in content[0]["text"]
        assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + JPEG}}
        assert [r[0] for r in store.query("SELECT text FROM turns WHERE role='ai'")] == ["え、泣いてる！"]
        assert d._snap_wait is None  # 使ったら捨てる
    asyncio.run(run())


def test_no_screen_without_vision_or_video(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, ["え、泣いてる！"], ["犯人はお前だ、そうだろう。"])
        await react(d, b)  # drama_vision は既定でオフ: 求めない
        await until(lambda: not d.spontaneous.pending)
        assert not any(m["t"] == "snap" for m in b.texts) and isinstance(d.seen_[0][1]["content"], str)

        d2, b2, _ = make(tmp_path / "2", ["まさか！"], ["犯人はお前だ、そうだろう。"])
        d2.drama_vision = True
        await react(d2, b2)
        await until(lambda: any(m["t"] == "snap" for m in b2.texts))
        answer_snap(d2, b2, jpeg=None)  # 映像が無い: 文章だけで言う
        await until(lambda: not d2.spontaneous.pending)
        assert isinstance(d2.seen_[0][1]["content"], str)
    (tmp_path / "2").mkdir()
    asyncio.run(run())


def test_screen_timeout_goes_without_image(tmp_path, monkeypatch):
    monkeypatch.setattr(director_mod, "SNAP_TIMEOUT_S", 0.05)

    async def run():
        d, b, store = make(tmp_path, ["まさか！"], ["犯人はお前だ、そうだろう。"])
        d.drama_vision = True
        await react(d, b)
        await until(lambda: not d.spontaneous.pending)
        assert any(m["t"] == "snap" for m in b.texts) and isinstance(d.seen_[0][1]["content"], str)
        assert [r[0] for r in store.query("SELECT text FROM turns WHERE role='ai'")] == ["まさか！"]
        snap = next(m for m in b.texts if m["t"] == "snap")
        d.core.on_text(b, json.dumps({"t": "snap", "id": snap["id"], "jpeg": JPEG}))  # 遅れて届いた返事は捨てる
    asyncio.run(run())


def test_image_failure_retries_without_image(tmp_path):
    async def run():
        d, b, store = make(tmp_path, [], ["犯人はお前だ、そうだろう。"])
        calls = []

        def llm(messages):
            calls.append(messages[-1]["content"])
            if isinstance(messages[-1]["content"], list):
                raise OSError("HTTP 500: image input is not supported")  # mmproj 無しの llama-server
            yield "まさか！"
        d._llm = llm
        d.drama_vision = True
        await react(d, b)
        await until(lambda: any(m["t"] == "snap" for m in b.texts))
        answer_snap(d, b)
        await until(lambda: not d.spontaneous.pending)
        assert isinstance(calls[0], list) and isinstance(calls[1], str)
        assert "画像" not in calls[1] and "犯人はお前だ" in calls[1]
        assert [r[0] for r in store.query("SELECT text FROM turns WHERE role='ai'")] == ["まさか！"]
    asyncio.run(run())


def test_finale_attaches_screen(tmp_path):
    async def run():
        d, b, _ = make(tmp_path, ["最後は泣けたね。あなたはどうだった？"], ["犯人はお前だ、そうだろう。"])
        d.drama = DramaListener(SETTINGS._replace(digest_chars=2000, finale_gap_s=120, finale_min_chars=10),
                                EnergyVad(), 200, d.clock_)
        d.drama_vision = True
        drama_on(d, b)
        drama_speak(b)
        await until(lambda: d.drama.lines)
        d.drama.mark_reacted()
        d.clock_.advance(120)
        assert d.spontaneous_tick() == "drama_finale"
        await until(lambda: any(m["t"] == "snap" for m in b.texts))
        answer_snap(d, b)
        await until(lambda: not d.spontaneous.pending)
        content = d.seen_[-1][-1]["content"]
        assert content[1]["image_url"]["url"].endswith(JPEG) and "犯人はお前だ" in content[0]["text"]
    asyncio.run(run())


def test_cancel_while_waiting_for_screen(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["まさか！"], ["犯人はお前だ、そうだろう。"])
        d.drama_vision = True
        await react(d, b)
        await until(lambda: any(m["t"] == "snap" for m in b.texts))
        d.core.on_text(b, json.dumps({"t": "ui", "drama": False}))  # 共有を止めた: 一言も取消す
        await until(lambda: not d.spontaneous.pending)
        answer_snap(d, b)
        await asyncio.sleep(0.05)
        assert d.seen_ == [] and store.query("SELECT text FROM turns WHERE role='ai'") == []
    asyncio.run(run())
