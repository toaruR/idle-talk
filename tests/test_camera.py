"""カメラモードのソース (設計書 6.9): cameras.toml、ffmpeg での取得、ソースを順に見る Director。"""
import asyncio
import base64
import json
import shutil

import pytest
from test_drama import make, until
from test_screen import DARK, JPEG, N, ai_turns, look

import idle_talk.director as dm
from idle_talk import camera
from idle_talk.camera import BROWSER, Cameras, CameraCycle, Source, load_cameras, mask_url


def test_load_cameras(tmp_path):
    p = tmp_path / "cameras.toml"
    assert load_cameras(p) == Cameras()  # 無ければブラウザの Web カメラだけ
    p.write_text('ffmpeg = "C:/ff.exe"\n[[source]]\nlabel = "部屋"\nurl = "browser"\n'
                 '[[source]]\nlabel = "駐車場"\nurl = "rtsp://u:p@h/s"\n[[source]]\ndevice = "Cam"\n', encoding="utf-8")
    c = load_cameras(p)
    assert c.ffmpeg == "C:/ff.exe" and c.sources == (Source("部屋", BROWSER), Source("駐車場", "rtsp://u:p@h/s"), Source("", "", "Cam"))
    for bad in ("", "x = 1\n[[source]]\nurl = 'browser'", "[[source]]\nlabel = 'a'",
                "[[source]]\nurl = 'browser'\ndevice = 'Cam'", "[[source]]\nurl = 'browser'\nfoo = 1",
                f"[[source]]\nurl = 'browser'\nlabel = '{'a' * 41}'"):
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            load_cameras(p)


def test_mask_url():
    assert mask_url("rtsp://user:secret@192.168.0.1:554/s") == "rtsp://user:***@192.168.0.1:554/s"
    assert mask_url("rtsp://192.168.0.1/s") == "rtsp://192.168.0.1/s"
    assert "secret" not in camera.describe(Source("", "rtsp://u:secret@h/s"))


def test_cycle_compares_per_source():
    cyc = CameraCycle(Cameras((Source("a", BROWSER), Source("b", "rtsp://h"))))
    assert cyc.needs_browser and not CameraCycle(Cameras((Source("b", "rtsp://h"),))).needs_browser
    assert cyc.next().label == "a" and cyc.diff(bytes(N)) == 100
    cyc.mark_shown(bytes(N))
    assert cyc.next().label == "b" and cyc.diff(bytes(N)) == 100  # 別のソースとは比べない
    cyc.mark_shown(bytes([255]) * N)
    assert cyc.next().label == "a" and cyc.diff(bytes(N)) == 0
    cyc.reset()
    assert cyc.next().label == "a" and cyc.diff(bytes(N)) == 100


def test_grab_builds_ffmpeg_commands(monkeypatch):
    calls = []

    def run(cmd, data=None):
        calls.append(cmd)
        return b"JPG" if data is None else bytes(N)
    monkeypatch.setattr(camera, "_run", run)
    assert camera.grab(Source("", "rtsp://h/s"), "ff") == (base64.b64encode(b"JPG").decode(), bytes(N))
    assert calls[0][0] == "ff" and calls[0][calls[0].index("-i") - 1] == "tcp" and "rtsp://h/s" in calls[0]
    camera.grab(Source("", "", "Cam"))
    assert calls[2][calls[2].index("-i") - 2:calls[2].index("-i") + 2] == ["-f", "dshow", "-i", "video=Cam"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg がない")
def test_grab_real_ffmpeg(monkeypatch):
    orig = camera._run

    def run(cmd, data=None):  # 入力を ffmpeg の合成映像 (-f lavfi -i testsrc) に差し替える
        if data is None:
            i = cmd.index("-i")
            cmd = cmd[:i] + ["-f", "lavfi", "-i", "testsrc=size=1280x720"] + cmd[i + 2:]
        return orig(cmd, data)
    monkeypatch.setattr(camera, "_run", run)
    jpeg, thumb = camera.grab(Source("", "dummy"))
    assert base64.b64decode(jpeg)[:2] == bytes([0xFF, 0xD8]) and len(thumb) == N


def test_director_cycles_sources_with_labels(tmp_path, monkeypatch):
    async def run():
        d, b, store = make(tmp_path, ["お、何か作ってるの？", "車が増えたね。"], [])
        d.drama_vision = True
        d.cameras = CameraCycle(Cameras((Source("部屋のカメラ", BROWSER), Source("駐車場", "rtsp://u:pw@h/s"))))
        grabbed = []
        monkeypatch.setattr(dm, "grab_camera", lambda src, ff: (grabbed.append(src.label), (JPEG, bytes([200]) * N))[1])
        d.core.on_text(b, json.dumps({"t": "ui", "camera": True}))
        await look(d, b, DARK)  # 1 つ目はブラウザの Web カメラ (snap)
        assert "このカメラ: 部屋のカメラ" in d.seen_[0][1]["content"][0]["text"]
        n = sum(m["t"] == "snap" for m in b.texts)
        d.clock_.advance(60)
        assert d.spontaneous_tick() == "screen"
        await until(lambda: not d.spontaneous.pending)
        assert grabbed == ["駐車場"] and sum(m["t"] == "snap" for m in b.texts) == n  # 2 つ目は ffmpeg、snap は求めない
        assert "このカメラ: 駐車場" in d.seen_[1][1]["content"][0]["text"]
        assert ai_turns(store) == ["お、何か作ってるの？", "車が増えたね。"]
        await look(d, b, DARK)  # 3 回目は部屋のカメラに戻り、前の部屋の画像と同じなので黙る
        assert len(d.seen_) == 2
    asyncio.run(run())
