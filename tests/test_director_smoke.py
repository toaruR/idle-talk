"""Director 結線の結合試験 (模擬ブラウザ + 模擬 LLM/TTS/ASR)。"""
import asyncio
import json
import os
import threading

import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk.clients import AsrClient
from idle_talk.director import IdleTalkDirector, wav_to_pcm
from idle_talk.errors import EXIT_DUPLICATE_LAUNCH
from idle_talk.main import main
from idle_talk.protocol import validate_message
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

STATE_ENUM = {"idle", "user_speaking", "eot_check", "preparing", "ai_playing", "interrupt_check", "rest"}


def make(tmp_path, tokens=("今日はいい天気ですね。", "散歩に行きましょう。"), gate=None, auto_playback=True):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    calls = {"n": 0}

    def llm(messages):
        calls["n"] += 1
        if gate is not None and calls["n"] == 1:
            gate.wait(10)
        yield from tokens

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
    b = MockBrowser(auto_playback=auto_playback)
    b.attach(d.core)
    return d, b, store


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def ai_rows(store, completed):
    return store.query("SELECT * FROM turns WHERE role='ai' AND completed=?", (completed,))


def test_one_turn(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        try:
            b.speak()
            await until(lambda: d.fsm.ctx.state.value == "idle" and ai_rows(store, 1))
            assert len(b.chunks) == 2 and all(c[3] for c in b.chunks)
            assert len(ai_rows(store, 1)) == 1
            assert ai_rows(store, 1)[0]["text"] == "今日はいい天気ですね。散歩に行きましょう。"
            assert store.query("SELECT COUNT(*) FROM turns WHERE completed=1 AND role='ai'")[0][0] == 1
        finally:
            store.close()
    asyncio.run(run())


def test_state_names(tmp_path):
    async def run():
        _d, b, store = make(tmp_path)
        try:
            b.speak()
            await until(lambda: ai_rows(store, 1))
            states = b.of("state")
            assert [s["state"] for s in states] == [
                "idle", "user_speaking", "eot_check", "preparing", "ai_playing", "idle"]
            assert all(s["state"] in STATE_ENUM and validate_message(s) for s in states)
        finally:
            store.close()
    asyncio.run(run())


def test_same_path(tmp_path):
    async def run():
        gate = threading.Event()
        d, b, store = make(tmp_path, gate=gate)
        try:
            b.speak()  # 1 回目: LLM がゲートで待機 -> preparing
            await until(lambda: d.fsm.ctx.state.value == "preparing")
            await until(lambda: ("response", "play_utterance") in d.path_calls)
            await asyncio.sleep(0.1)  # LLM スレッドがゲートに到達するまで
            b.speak()  # 準備中に再発話 -> cancel 経路
            await until(lambda: d.fsm.ctx.state.value == "idle" and ai_rows(store, 1))
            assert len(b.of("cancel")) == 1
            assert all(c[0] == d.epoch.current for c in b.chunks)
            pc = d.path_calls
            funcs = lambda route: {f for r, f in pc if r == route}
            assert ("response", "play_utterance") in pc and ("interrupt", "cancel") in pc
            assert ("response", "_finalize_utterance") in pc and ("interrupt", "_finalize_utterance") in pc
            assert {"_finalize_utterance"} <= funcs("response") & funcs("interrupt")
        finally:
            gate.set()
            store.close()
    asyncio.run(run())


def test_duplicate_start(tmp_path):
    pid = tmp_path / "idle_talk.pid"
    pid.write_text(str(os.getpid()))  # 生存中の PID を 2 つ目の起動から見せる
    with pytest.raises(SystemExit) as e:
        main(["--pid-file", str(pid), "--data-dir", str(tmp_path)])
    assert e.value.code == EXIT_DUPLICATE_LAUNCH == 4


def test_wav_to_pcm():
    wav = b"RIFF" + b"\0\0\0\0" + b"WAVE" + b"fmt " + (4).to_bytes(4, "little") + b"abcd" + b"data" + (2).to_bytes(4, "little") + b"xy"
    assert wav_to_pcm(wav) == b"xy"
    assert wav_to_pcm(b"RIFF....WAVE") == b"RIFF....WAVE"


def test_main_ports_from_settings_file(monkeypatch, tmp_path):
    from idle_talk import main as main_mod
    seen = {}

    class D:
        def close(self):
            pass

    monkeypatch.setattr(main_mod, "build_director", lambda d, args, cfg: seen.update(vars(args)) or D())
    monkeypatch.setattr(main_mod, "_run", lambda director, port, sup=None, stop_file=None, topic_interval_s=0: seen.update(run_port=port) or asyncio.sleep(0))
    monkeypatch.setattr(main_mod, "start_web", lambda port, ws_port, ui, speakers, personas=None: seen.update(web=(port, ws_port), ui=ui))
    monkeypatch.setattr(main_mod.TtsClient, "speakers", lambda self: [])
    f = tmp_path / "s.toml"
    f.write_text("[ports]\nws = 9000\nllm = 9001\ntts = 9002\nasr = 9003\nweb = 9004\n"
                 "[spontaneous]\non = true\ninterval_s = 30\n[tts]\nspeaker = 42\n", encoding="utf-8")
    base = ["--config", str(f), "--data-dir", str(tmp_path), "--pid-file", str(tmp_path / "p.pid")]
    assert main_mod.main(base) == 0
    assert seen["run_port"] == 9000
    assert (seen["llm_url"], seen["tts_url"], seen["asr_url"]) == (
        "http://127.0.0.1:9001", "http://127.0.0.1:9002", "http://127.0.0.1:9003")
    assert seen["speaker"] == 42
    assert seen["web"] == (9004, 9000)
    assert seen["ui"] == {"on": True, "radio": False, "interval_s": 30, "quiet_start": "22:00", "quiet_end": "07:00",
                          "speaker": 42, "persona": "default"}
    assert main_mod.main(base + ["--port", "7000", "--llm-url", "http://h:1", "--speaker", "7"]) == 0
    assert seen["run_port"] == 7000 and seen["llm_url"] == "http://h:1" and seen["speaker"] == 7
    assert main_mod.main(base + ["--web-port", "7001"]) == 0
    assert seen["web"] == (7001, 9000)


def test_start_web_serves_dir_and_ws_port(tmp_path):
    import http.client
    from idle_talk.main import start_web
    (tmp_path / "index.html").write_text("hi", encoding="utf-8")
    httpd = start_web(0, 9123, {"on": True}, [{"id": 1, "name": "a b"}], root=tmp_path)
    try:
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        c.request("GET", "/config.json")
        r = c.getresponse()
        assert (r.status, json.loads(r.read())) == (200, {"ws_port": 9123, "ui": {"on": True},
                                                          "speakers": [{"id": 1, "name": "a b"}],
                                                          "personas": []})
        c.request("GET", "/")
        r = c.getresponse()
        assert (r.status, r.read(), r.getheader("Cache-Control")) == (200, b"hi", "no-cache")
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_start_web_refetches_speakers_until_available(tmp_path):
    import http.client
    from idle_talk.main import SpeakerChoices, start_web
    answers = [[], [{"id": 1, "name": "a b"}], [{"id": 2, "name": "c d"}]]  # AivisSpeech が後から起動
    choices = SpeakerChoices(lambda: answers.pop(0))
    httpd = start_web(0, 9123, {}, choices, root=tmp_path)
    try:
        c = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
        got = []
        for _ in range(3):
            c.request("GET", "/config.json")
            got.append(json.loads(c.getresponse().read())["speakers"])
        assert got == [[], [{"id": 1, "name": "a b"}], [{"id": 1, "name": "a b"}]]  # 取れたら覚える
        assert 1 in choices and 2 not in choices and choices
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_request_rest_moves_to_rest_and_notifies(tmp_path):
    from idle_talk.fsm import State
    d, b, store = make(tmp_path)
    try:
        d.request_rest()  # LLM_RETRY 超過時に main.llm_watch から呼ばれる
        assert d.fsm.ctx.state is State.REST
        assert any(m.get("t") == "state" and m.get("state") == "rest" for m in b.texts)
    finally:
        store.close()


def test_ui_without_mute_wakes_from_rest(tmp_path):
    import json
    from idle_talk.fsm import State
    d, b, store = make(tmp_path)
    try:
        d.request_rest()  # 無操作 (IDLE_TIMEOUT) と同じく muted=False の休止
        d.core.on_text(b, json.dumps({"t": "ui", "mute": False}))  # 画面の「開始」は接続時に ui を送る
        assert d.fsm.ctx.state is State.IDLE
        d.core.on_text(b, json.dumps({"t": "ui", "mute": True}))
        assert d.fsm.ctx.state is State.REST
    finally:
        store.close()


def test_hello_gets_current_state(tmp_path):
    d, b, store = make(tmp_path)  # make() が attach (hello 送信) 済み
    try:
        assert b.of("state") == [{"t": "state", "state": "idle", "muted": False}]
    finally:
        store.close()


def test_tts_settings_reloaded_on_file_change(tmp_path, caplog):
    import os
    from idle_talk.clients import TtsClient
    from idle_talk.main import TtsSettingsReloader
    f = tmp_path / "s.toml"
    f.write_text("[tts]\nspeaker = 1\nspeed = 1.0\n", encoding="utf-8")
    tts = TtsClient(speaker=1, voice={"speedScale": 1.0}, post=lambda url, body, timeout: b"{}")
    r = TtsSettingsReloader(tts, f)
    assert r.reload_if_changed() is False
    f.write_text("[tts]\nspeaker = 2\nspeed = 0.8\n", encoding="utf-8")
    os.utime(f, (1, 1))  # mtime の分解能に依存しないよう明示的に変える
    assert r.reload_if_changed() is True
    assert (tts.speaker, tts.voice) == (2, {"speedScale": 0.8})
    f.write_text("[tts]\nspeaker = 3\nspeed = 9\n", encoding="utf-8")  # 範囲外: 前の値を保つ
    os.utime(f, (2, 2))
    assert r.reload_if_changed() is False
    assert (tts.speaker, tts.voice) == (2, {"speedScale": 0.8})
    fixed = TtsSettingsReloader(TtsClient(speaker=7), f, fixed_speaker=True)
    f.write_text("[tts]\nspeaker = 4\n", encoding="utf-8")
    os.utime(f, (3, 3))
    assert fixed.reload_if_changed() is True and fixed.tts.speaker == 7


def test_ui_speaker_kept_until_file_speaker_changes(tmp_path):
    from idle_talk.clients import TtsClient
    from idle_talk.main import TtsSettingsReloader
    f = tmp_path / "s.toml"
    f.write_text("[tts]\nspeaker = 1\n", encoding="utf-8")
    tts = TtsClient(speaker=1)
    r = TtsSettingsReloader(tts, f, speaker_ids=frozenset({1, 2, 3}))
    r.set_speaker(2)
    assert tts.speaker == 2
    with pytest.raises(ValueError):
        r.set_speaker(99)
    assert tts.speaker == 2
    f.write_text("[tts]\nspeaker = 1\nspeed = 1.2\n", encoding="utf-8")  # 声パラメータだけの編集
    os.utime(f, (1, 1))
    assert r.reload_if_changed() is True and (tts.speaker, tts.voice) == (2, {"speedScale": 1.2})
    f.write_text("[tts]\nspeaker = 3\n", encoding="utf-8")
    os.utime(f, (2, 2))
    assert r.reload_if_changed() is True and tts.speaker == 3


def test_ui_speaker_message_reaches_tts(tmp_path):
    d = IdleTalkDirector(Store(tmp_path / "t.db", db_backup=False), lambda m: iter(()), lambda t: b"",
                         lambda: None, EnergyVad(), None)
    seen = []

    def set_speaker(s):
        if s == 99:
            raise ValueError("unknown speaker: 99")
        seen.append(s)

    d.set_speaker = set_speaker
    b = MockBrowser()
    b.attach(d.core)
    d.core.on_text(b, json.dumps({"t": "ui", "speaker": 5}))
    d.core.on_text(b, json.dumps({"t": "ui", "speaker": 99}))
    assert seen == [5] and [m["code"] for m in b.of("error")] == [5006]
    d.close()
