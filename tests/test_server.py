import asyncio
import json
import logging
import subprocess
import sys

import pytest
from mocks import FakeClock

from idle_talk.errors import EXIT_DUPLICATE_LAUNCH
from idle_talk.portlock import DuplicateLaunch, PidLock, pid_alive
from idle_talk.protocol import pack_up_frame
from idle_talk.server import WsCore, serve

OK = "http://localhost:8765"


class FakeConn:
    def __init__(self):
        self.texts, self.bins, self.closed = [], [], []

    def send_text(self, s):
        self.texts.append(json.loads(s))

    def send_bytes(self, b):
        self.bins.append(b)

    def close(self, code):
        self.closed.append(code)

    def of(self, t):
        return [m for m in self.texts if m["t"] == t]


def hello(**kw):
    d = {"t": "hello", "version": 1, "sample_rate": 16000, "client_id": "c"}
    d.update(kw)
    return json.dumps(d)


def setup(**kw):
    clock = FakeClock()
    core = WsCore(clock=clock, **kw)
    c = FakeConn()
    assert core.connect(c, OK)
    return core, clock, c


def test_bind_host_only_loopback():
    WsCore(host="127.0.0.1")
    with pytest.raises(ValueError):
        WsCore(host="0.0.0.0")


def test_origin_rejected_4003(caplog):
    core = WsCore()
    c = FakeConn()
    with caplog.at_level(logging.WARNING, logger="idle_talk.server"):
        assert not core.connect(c, "http://evil.example.com")
    assert c.closed == [4003]
    assert len([r for r in caplog.records if "origin rejected" in r.message]) == 1
    assert core.conn is None


def test_replace_old_connection_4001(caplog):
    core, _, a = setup()
    b = FakeConn()
    with caplog.at_level(logging.INFO, logger="idle_talk.server"):
        assert core.connect(b, OK)
    assert a.closed == [4001]
    assert core.conn is b
    assert len([r for r in caplog.records if "replacing" in r.message]) == 1
    core.on_text(b, hello())
    assert core.hello_done


def test_version_mismatch_4002_and_bad_sample_rate():
    core, _, c = setup()
    core.on_text(c, hello(version=2))
    assert c.closed == [4002]
    core, _, c = setup()
    core.on_text(c, hello(sample_rate=44100))
    assert c.closed and not core.hello_done


def test_ping_detection_within_10s_and_epoch_bump():
    core, clock, c = setup()
    assert len(c.of("ping")) == 1
    e0 = core.epoch.current
    for _ in range(9):
        clock.advance(1)
        assert not core.tick()
    assert core.conn is c and c.closed == []  # 9 秒時点では未切断
    clock.advance(1)
    assert core.tick()  # 10 秒
    assert core.conn is None and c.closed
    assert len(c.of("ping")) == 2  # 5s ごと
    c2 = FakeConn()
    core.connect(c2, OK)
    assert core.epoch.current == e0 + 1
    assert c2.of("cancel")[0]["new_epoch"] == e0 + 1


def test_pong_keeps_connection():
    core, clock, c = setup()
    for _ in range(6):
        clock.advance(5)
        core.on_text(c, json.dumps({"t": "pong", "ts": 1}))
        assert not core.tick()
    assert core.conn is c


def test_ui_range_error_5006_keeps_settings():
    core, _, c = setup()
    before = dict(core.settings)
    core.on_text(c, json.dumps({"t": "ui", "freq_per_hour": 13}))
    assert c.of("error")[0]["code"] == 5006
    assert core.settings == before
    core.on_text(c, json.dumps({"t": "ui", "freq_per_hour": 5}))
    assert core.settings["freq_per_hour"] == 5


def test_stale_playback_epoch_dropped(caplog):
    got = []
    core, _, c = setup(on_playback=got.append)
    pb = {"t": "playback", "epoch": 7, "utterance_id": 1, "chunk_id": 0,
          "position_ms": 10, "state": "started"}
    with caplog.at_level(logging.INFO, logger="idle_talk.server"):
        core.on_text(c, json.dumps(pb))
    assert got == [] and any("stale playback" in r.message for r in caplog.records)
    core.on_text(c, json.dumps({**pb, "epoch": core.epoch.current}))
    assert len(got) == 1


def test_oversize_binary_dropped(caplog):
    got = []
    core, _, c = setup(on_audio=lambda s, p: got.append((s, p)))
    core.on_binary(c, pack_up_frame(5, b"\x01\x02"))
    assert got == [(5, b"\x01\x02")]
    with caplog.at_level(logging.WARNING, logger="idle_talk.server"):
        core.on_binary(c, b"\x00" * 1048577)
    assert len(got) == 1
    assert len([r for r in caplog.records if "oversize" in r.message]) == 1


def test_portlock_dead_reclaimed_alive_rejected(tmp_path):
    p = tmp_path / "d.pid"
    p.write_text("99999")
    lock = PidLock(p, pid=1234, is_alive=lambda pid: False)
    lock.acquire()
    assert p.read_text() == "1234"
    with pytest.raises(DuplicateLaunch):
        PidLock(p, pid=5678, is_alive=lambda pid: True).acquire()
    lock.release()
    assert not p.exists()


def test_portlock_real_pid_alive():
    assert pid_alive(subprocess.os.getpid())
    pr = subprocess.Popen([sys.executable, "-c", "pass"])
    pr.wait()
    assert not pid_alive(pr.pid)


def test_portlock_second_process_exit_4(tmp_path):
    p = tmp_path / "d.pid"
    first = PidLock(p)
    first.acquire()
    code = ("import sys;from idle_talk.portlock import acquire_or_exit;"
            f"acquire_or_exit({str(p)!r})")
    r = subprocess.run([sys.executable, "-c", code], check=False,
                       env={**__import__("os").environ, "PYTHONPATH": "src"})
    assert r.returncode == EXIT_DUPLICATE_LAUNCH == 4
    first.release()


def test_real_websocket_origin_and_replace():
    from websockets.asyncio.client import connect

    async def run():
        core = WsCore()
        server, task = await serve(core, 0)
        port = server.sockets[0].getsockname()[1]
        url = f"ws://127.0.0.1:{port}/ws"
        try:
            async with connect(url, origin="http://evil.example.com") as w:
                with pytest.raises(Exception) as ei:
                    await w.recv()
                assert ei.value.rcvd.code == 4003
            async with connect(url, origin=OK) as a:
                await a.send(hello())
                async with connect(url, origin=OK) as b:
                    with pytest.raises(Exception) as ei:
                        while True:
                            await a.recv()
                    assert ei.value.rcvd.code == 4001
                    await b.send(hello())
                    await asyncio.sleep(0.1)
                    assert core.hello_done
        finally:
            task.cancel()
            server.close()
            await server.wait_closed()

    asyncio.run(run())
