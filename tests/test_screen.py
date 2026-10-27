"""画面モード (設計書 6.7): 共有した画面を interval_s ごとに見て、前に話したときから変わっていれば一言。"""
import asyncio
import base64
import json

import pytest
from mocks import FakeClock
from test_drama import make, until

from idle_talk.config import Screen, load_screen
from idle_talk.protocol import validate_message
from idle_talk.screen import THUMB_H, THUMB_W, ScreenWatcher, decode_thumb, thumb_diff

JPEG = "QUJD"  # base64 の中身は見ない
N = THUMB_W * THUMB_H


def thumb(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


DARK = thumb(bytes(N))
SLIGHT = thumb(bytes(N - 6) + bytes([255]) * 6)  # 6 画素だけ変わった (約 1%)
BRIGHT = thumb(bytes([200]) * N)


# ---- 設定・プロトコル ----
def test_load_screen_and_schemas(tmp_path):
    p = tmp_path / "s.toml"
    assert load_screen(p) == Screen()
    p.write_text("[screen]\ninterval_s = 10\nmin_diff = 0\n", encoding="utf-8")
    assert load_screen(p) == Screen(10, 0)
    for bad in ("interval_s = 9", "min_diff = 101", "min_diff = 2.5", "foo = 1"):
        p.write_text(f"[screen]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_screen(p)
    assert validate_message({"t": "ui", "screen": True})
    assert not validate_message({"t": "ui", "screen": 1})
    assert validate_message({"t": "snap", "id": 1, "jpeg": JPEG, "thumb": DARK})
    assert not validate_message({"t": "snap", "id": 1, "thumb": "x" * 1025})


# ---- ScreenWatcher ----
def test_thumb_diff_and_decode():
    assert decode_thumb(DARK) == bytes(N)
    for bad in (None, "", "!!", thumb(bytes(N - 1))):
        assert decode_thumb(bad) is None
    assert thumb_diff(bytes(N), bytes([255]) * N) == 100
    assert round(thumb_diff(decode_thumb(DARK), decode_thumb(SLIGHT)), 2) == round(600 / N, 2)


def test_watcher_due_and_changed():
    clock = FakeClock()
    w = ScreenWatcher(Screen(interval_s=60, min_diff=3), clock)
    assert not w.due()
    clock.advance(60)
    assert w.due()
    w.mark_checked()
    assert not w.due()
    assert w.changed(decode_thumb(DARK))  # 初回は比べる相手が無い: 変わったとみなす
    w.mark_shown(decode_thumb(DARK))
    assert not w.changed(decode_thumb(SLIGHT)) and w.changed(decode_thumb(BRIGHT))
    assert w.changed(None)  # 縮小画像が無ければ比べられない
    w.stop()
    assert w.changed(decode_thumb(DARK))


# ---- Director ----
def screen_on(d, b):
    d.drama_vision = True
    d.core.on_text(b, json.dumps({"t": "ui", "screen": True}))
    assert d.screen_on and not d.drama_on


async def look(d, b, th, jpeg=JPEG):
    """interval_s 進めて画面を見に行かせ、求められた画面を返す。"""
    n = sum(m["t"] == "snap" for m in b.texts)
    d.clock_.advance(60)
    assert d.spontaneous_tick() == "screen"
    await until(lambda: sum(m["t"] == "snap" for m in b.texts) > n)
    snap = [m for m in b.texts if m["t"] == "snap"][-1]
    d.core.on_text(b, json.dumps({"t": "snap", "id": snap["id"], **({"jpeg": jpeg} if jpeg else {}), "thumb": th}))
    await until(lambda: not d.spontaneous.pending)


def ai_turns(store):
    return [r[0] for r in store.query("SELECT text FROM turns WHERE role='ai'")]


def test_comments_only_when_screen_changed(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["面白そうな記事だね！", "お、動画に変えたんだ。"], [])
        screen_on(d, b)
        assert d.spontaneous_tick() is None  # interval_s 前は見に行かない
        await look(d, b, DARK)
        content = d.seen_[0][1]["content"]
        assert content[0]["type"] == "text" and "画面" in content[0]["text"]
        assert content[1] == {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + JPEG}}
        assert d.seen_[0][0]["role"] == "system" and len(d.seen_[0]) == 2  # 会話履歴は渡さない
        assert ai_turns(store) == ["面白そうな記事だね！"]

        await look(d, b, SLIGHT)  # ほとんど変わっていない: LLM を呼ばずに黙る
        assert len(d.seen_) == 1 and ai_turns(store) == ["面白そうな記事だね！"]
        skip = store.query("SELECT payload_json FROM events WHERE kind='screen_skip'")
        assert len(skip) == 1 and json.loads(skip[0][0])["diff"] < 3

        await look(d, b, BRIGHT)  # 前に話した画面 (DARK) から大きく変わった
        assert len(d.seen_) == 2 and ai_turns(store) == ["面白そうな記事だね！", "お、動画に変えたんだ。"]
    asyncio.run(run())


def test_silent_on_nothing_or_no_image(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["なし"], [])
        screen_on(d, b)
        await look(d, b, DARK, jpeg=None)  # 画面が取れない: LLM を呼ばない
        assert d.seen_ == []
        await look(d, b, DARK)  # 「なし」: 黙るが、この画面は見せたので次は比べる
        assert len(d.seen_) == 1 and ai_turns(store) == []
        await look(d, b, DARK)
        assert len(d.seen_) == 1
        assert len(store.query("SELECT 1 FROM events WHERE kind='screen_skip'")) == 3
    asyncio.run(run())


def test_screen_off_cancels_and_resets(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["見て見て"], [])
        screen_on(d, b)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "screen"
        await until(lambda: any(m["t"] == "snap" for m in b.texts))
        d.core.on_text(b, json.dumps({"t": "ui", "screen": False}))  # 共有を止めた: 一言も取消す
        await until(lambda: not d.spontaneous.pending)
        snap = next(m for m in b.texts if m["t"] == "snap")
        d.core.on_text(b, json.dumps({"t": "snap", "id": snap["id"], "jpeg": JPEG, "thumb": DARK}))
        await asyncio.sleep(0.05)
        assert d.seen_ == [] and ai_turns(store) == [] and not d.screen_on
    asyncio.run(run())


# ---- カメラモード (ui.camera) ----
def test_camera_uses_browser_snap_and_camera_prompt(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["お、コーヒー飲んでるの？"], [])
        d.drama_vision = True
        d.screen_capture = lambda: pytest.fail("カメラは前面のウィンドウを撮らない")
        assert validate_message({"t": "ui", "camera": True}) and not validate_message({"t": "ui", "camera": 1})
        d.core.on_text(b, json.dumps({"t": "ui", "camera": True}))
        assert d.screen_on and d.camera_on
        await look(d, b, DARK)
        text = d.seen_[0][1]["content"][0]["text"]
        assert "カメラ" in text and "ウィンドウのタイトル" not in text
        assert ai_turns(store) == ["お、コーヒー飲んでるの？"]
        d.core.on_text(b, json.dumps({"t": "ui", "screen": True, "camera": False}))  # カメラから画面へ: 見直す (比べる相手を捨てる)
        assert d.screen_on and not d.camera_on and d.screen.changed(None)
        d.core.on_text(b, json.dumps({"t": "ui", "screen": False}))
        assert not d.screen_on and not d.camera_on
    asyncio.run(run())


# ---- 前面のウィンドウ ([screen].follow_window) ----
def test_load_follow_window(tmp_path):
    p = tmp_path / "s.toml"
    assert not load_screen(p).follow_window and load_screen(p).exclude == ()  # 既定はオフ
    p.write_text('[screen]\nfollow_window = true\nexclude = ["KeePass"]\n', encoding="utf-8")
    assert load_screen(p)[2:4] == (True, ("KeePass",))
    assert load_screen(p)[4:] == (1280, 5)
    for bad in ("follow_window = 1", 'exclude = "KeePass"', 'exclude = [""]', "width = 1281", "color_bits = 3"):
        p.write_text(f"[screen]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_screen(p)


def test_wincap_helpers():
    import struct
    import zlib

    from idle_talk.wincap import bgra_to_gray, bgra_to_png, excluded, fit, posterize
    assert fit(1920, 1080) == (640, 360) and fit(1000, 1000) == (360, 360) and fit(100, 50) == (100, 50)
    assert excluded("", ()) and excluded("idle-talk - Google Chrome", ())  # デスクトップ・自分自身
    assert not excluded("screen.py - idle-talk - Visual Studio Code", ())  # フォルダ名が入るだけなら撮る
    assert excluded("Vault - keepass", ("KeePass",)) and not excluded("YouTube", ("KeePass",))
    bgra = bytes([0, 0, 255, 255, 255, 0, 0, 255])  # 赤, 青 (1x2 ではなく 2x1)
    png = bgra_to_png(bgra, 2, 1)
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", png[16:24]) == (2, 1)
    idat = png[png.index(b"IDAT") + 4:png.index(b"IEND") - 8]
    assert zlib.decompress(idat) == b"\x00" + bytes([255, 0, 0, 0, 0, 255])
    assert bgra_to_gray(bgra) == bytes([76, 29])
    assert posterize(bytes([0, 7, 8, 255]), 5) == bytes([4, 4, 12, 252]) and posterize(bgra, 8) == bgra


def test_follow_window_uses_foreground_and_title(tmp_path):
    from idle_talk.wincap import Shot

    async def run():
        d, b, store = make(tmp_path, ["その動画面白そう！", "今度はメールか。", "まだ動画見てるね。"], [])
        shots = [Shot("UE5H", bytes(N), "猫の動画 - YouTube"), Shot("UE5H", bytes(N), "受信トレイ - Outlook"),
                 None, Shot("UE5H", bytes(N), "受信トレイ - Outlook")]
        d.screen_capture = lambda: shots.pop(0)
        screen_on(d, b)
        for _ in range(4):
            d.clock_.advance(60)
            assert d.spontaneous_tick() == "screen"
            await until(lambda: not d.spontaneous.pending)
        assert not any(m["t"] == "snap" for m in b.texts)  # ブラウザには求めない
        text, image = d.seen_[0][1]["content"]
        assert "猫の動画 - YouTube" in text["text"] and image["image_url"]["url"] == "data:image/png;base64,UE5H"
        # 画面が同じでもウィンドウが替われば話す。撮れない (除外等) と同じウィンドウのままは黙る
        assert len(d.seen_) == 2 and ai_turns(store) == ["その動画面白そう！", "今度はメールか。"]
        assert len(store.query("SELECT 1 FROM events WHERE kind='screen_skip'")) == 2
    asyncio.run(run())


@pytest.mark.skipif(__import__("sys").platform != "win32", reason="Windows only")
def test_window_capture_smoke():
    from idle_talk.wincap import WindowCapture
    shot = WindowCapture(()).grab()  # 前面が除外・最小化なら None
    assert shot is None or (base64.b64decode(shot.png)[:4] == b"\x89PNG" and len(shot.thumb) == N)
