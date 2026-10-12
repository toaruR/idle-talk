"""Director 結線の結合試験 (模擬ブラウザ + 模擬 LLM/TTS/ASR)。"""
import asyncio
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
                "user_speaking", "eot_check", "preparing", "ai_playing", "idle"]
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
    monkeypatch.setattr(main_mod, "_run", lambda director, port, sup=None: seen.update(run_port=port) or asyncio.sleep(0))
    f = tmp_path / "s.toml"
    f.write_text("[ports]\nws = 9000\nllm = 9001\ntts = 9002\nasr = 9003\n[tts]\nspeaker = 42\n", encoding="utf-8")
    base = ["--config", str(f), "--data-dir", str(tmp_path), "--pid-file", str(tmp_path / "p.pid")]
    assert main_mod.main(base) == 0
    assert seen["run_port"] == 9000
    assert (seen["llm_url"], seen["tts_url"], seen["asr_url"]) == (
        "http://127.0.0.1:9001", "http://127.0.0.1:9002", "http://127.0.0.1:9003")
    assert seen["speaker"] == 42
    assert main_mod.main(base + ["--port", "7000", "--llm-url", "http://h:1", "--speaker", "7"]) == 0
    assert seen["run_port"] == 7000 and seen["llm_url"] == "http://h:1" and seen["speaker"] == 7


def test_request_rest_moves_to_rest_and_notifies(tmp_path):
    from idle_talk.fsm import State
    d, b, store = make(tmp_path)
    try:
        d.request_rest()  # LLM_RETRY 超過時に main.llm_watch から呼ばれる
        assert d.fsm.ctx.state is State.REST
        assert any(m.get("t") == "state" and m.get("state") == "rest" for m in b.texts)
    finally:
        store.close()
