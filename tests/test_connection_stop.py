"""接続オフで認識・生成・機器を止め、同じサーバに再接続できる。"""
import asyncio
import json
import threading

from mocks_browser import LOUD
from test_integration_interrupt import asr_settled, make, until
from test_pc import make as make_pc, pc_on

from idle_talk.fsm import State


def test_disconnect_drops_late_asr_and_reconnects(tmp_path):
    async def scenario():
        d, browser, store = make(tmp_path, texts=["停止前の音声です", "再接続して話します"])
        try:
            held = []
            d.ingress.offload = lambda work, done: held.append((work, done))
            browser.speak()
            assert len(held) == 1 and d.ingress.asr_pending == 1
            work, done = held.pop()
            result = work()  # 認識は済んでいるが、結果は切断後に届く
            d.core.disconnect(browser)
            done(result)
            assert d.fsm.ctx.state is State.REST
            assert d.ingress.asr_pending == 0 and d.ingress.shm_count == 0
            assert not store.query("SELECT * FROM turns WHERE role='user'")
            browser.speak()
            assert not held
            browser.attach(d.core)
            d.core.on_text(browser, json.dumps({"t": "ui", "mute": False}))
            assert d.fsm.ctx.state is State.IDLE
            browser.speak()
            work, done = held.pop()
            done(work())
            d._pump()
            await until(lambda: bool(browser.chunks))
            assert [row[0] for row in store.query("SELECT text FROM turns WHERE role='user'")] == ["再接続して話します"]
        finally:
            d.core.disconnect(browser)
            await asyncio.sleep(0)
            d.close()
    asyncio.run(scenario())


def test_disconnect_clears_partial_audio_and_pending_queue(tmp_path):
    async def scenario():
        texts = ["まだ認識していない音声"]
        d, browser, _store = make(tmp_path, texts=texts)
        try:
            held = []
            d.ingress.offload = lambda work, done: held.append((work, done))
            browser.speak()
            for _ in range(15):
                browser.send_frame(LOUD)
            assert d.ingress._buf is not None
            d.core.disconnect(browser)
            assert d.ingress._buf is None and not d.ingress._pre
            assert not d.ingress.vad._speaking
            work, done = held.pop()
            done(work())  # 未開始の認識は ASR を呼ばずに捨てる
            assert texts == ["まだ認識していない音声"]
            assert d.calls["n"] == 0
            assert d.ingress.shm_count == 0
            assert not d._down_q and not d._down_sent
            assert d.spontaneous_tick() is None
        finally:
            d.close()
    asyncio.run(scenario())


def test_disconnect_cancels_generation_and_modes(tmp_path):
    async def scenario():
        gate = threading.Event()
        d, browser, _store = make(tmp_path, gate=gate)
        try:
            browser.speak()
            await asr_settled(d)
            await until(lambda: d.calls["n"] == 1)
            d._set_drama(True)
            d._set_screen(True)
            d.core.disconnect(browser)
            assert d._task.cancelled() or d._task.cancelling()
            assert not d.drama_on and not d.screen_on
            assert d.fsm.ctx.state is State.REST
            gate.set()
            await asyncio.sleep(0.05)
            assert not browser.chunks
        finally:
            gate.set()
            d.close()
    asyncio.run(scenario())


def test_disconnect_stops_server_microphone_and_monitor(tmp_path):
    async def scenario():
        d, browser, _store = make_pc(tmp_path, [], [])
        try:
            pc_on(d, browser)
            monitor, mic = d.monitor, d.mic
            assert d.pc_on and mic is not None
            d.core.disconnect(browser)
            assert not d.pc_on and d.mic is None and d.monitor is None
            assert not monitor.running and not mic.running
        finally:
            d.close()
    asyncio.run(scenario())
