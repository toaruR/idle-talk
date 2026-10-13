"""ctl (起動・停止ラッパー) と main.stop_watch の試験。実サービスは起動せず、PID/停止ファイルの取り決めを検証する。"""
import asyncio
import os
import subprocess
import sys
import textwrap

import pytest

from idle_talk import asr_worker, ctl
from idle_talk.errors import EXIT_DUPLICATE_LAUNCH
from idle_talk.main import PID_FILE, STOP_FILE, stop_watch


def run_ctl(tmp_path, *cmd):
    return ctl.main([*cmd, "--data-dir", str(tmp_path), "--config", str(tmp_path / "none.toml")])


def test_stop_watch_removes_file_and_stops(tmp_path):
    f = tmp_path / STOP_FILE
    calls = []

    async def sleep(_):
        if not calls and not f.exists():
            f.write_text("")

    asyncio.run(stop_watch(f, lambda: calls.append(1), sleep=sleep))
    assert calls == [1] and not f.exists()


def test_start_rejects_when_main_alive(tmp_path, monkeypatch, capsys):
    (tmp_path / PID_FILE).write_text(str(os.getpid()))
    monkeypatch.setattr(ctl, "spawn", lambda *a: pytest.fail("must not spawn"))
    assert run_ctl(tmp_path, "start") == EXIT_DUPLICATE_LAUNCH
    assert "already running" in capsys.readouterr().out


def test_status_reports_stale_pid_as_stopped(tmp_path, capsys):
    (tmp_path / PID_FILE).write_text("0")
    assert run_ctl(tmp_path, "status") == 3
    assert "idle-talk: stopped" in capsys.readouterr().out


def test_stop_requests_graceful_exit_via_stop_file(tmp_path, capsys):
    code = textwrap.dedent(f"""
        import os, pathlib, time
        d = pathlib.Path({str(tmp_path)!r})
        (d / {PID_FILE!r}).write_text(str(os.getpid()))
        while not (d / {STOP_FILE!r}).exists():
            time.sleep(0.05)
        (d / "graceful").write_text("")
        (d / {PID_FILE!r}).unlink()
    """)
    child = subprocess.Popen([sys.executable, "-c", code])
    try:
        assert ctl.wait_until(lambda: (tmp_path / PID_FILE).exists(), 10)
        assert run_ctl(tmp_path, "stop") == 0
        child.wait(10)
    finally:
        child.kill()
    assert (tmp_path / "graceful").exists()
    assert not (tmp_path / STOP_FILE).exists()
    assert "idle-talk: stopped" in capsys.readouterr().out


def test_asr_worker_pid_file_rejects_duplicate(tmp_path):
    pid_file = tmp_path / "asr_worker.pid"
    pid_file.write_text(str(os.getpid()))
    with pytest.raises(SystemExit) as e:
        asr_worker.main(["--pid-file", str(pid_file)], load_model=lambda: pytest.fail("must not load"))
    assert e.value.code == EXIT_DUPLICATE_LAUNCH


def test_memories_list_delete_clear(tmp_path, capsys):
    from idle_talk import memory
    from idle_talk.main import DB_FILE
    from idle_talk.store import Store
    s = Store(tmp_path / DB_FILE, db_backup=False)
    a = memory.add_memory(s, "利用者は猫を飼っている。", 1, 1)
    memory.add_memory(s, "利用者は固定。", 1, 1, deletable=0)
    memory.add_memory(s, "利用者は朝型。", 1, 1)
    s.close()
    assert run_ctl(tmp_path, "memories") == 0
    out = capsys.readouterr().out
    assert "猫を飼っている" in out and "(固定)" in out
    assert run_ctl(tmp_path, "memories", "--delete", str(a)) == 0
    assert run_ctl(tmp_path, "memories", "--delete", str(a)) == 1
    assert run_ctl(tmp_path, "memories", "--clear") == 0
    capsys.readouterr()
    run_ctl(tmp_path, "memories")
    out = capsys.readouterr().out
    assert "固定" in out and "朝型" not in out and "猫" not in out
