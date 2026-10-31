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


def test_start_launches_aivis_engine_then_stop_kills_it(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "s.toml"
    cfg.write_text('[ports]\ntts = 10199\n[aivis]\nengine = "C:/a/run.exe"\n', encoding="utf-8")
    ready = []
    launched, killed = [], []

    class Proc:
        pid = os.getpid()  # 生存中の PID (stop で生存確認される)

        def poll(self):
            return None

    def spawn_exe(argv, log_path):
        launched.append(argv)
        ready.append(True)  # 起動後に応答し始める
        return Proc()

    monkeypatch.setattr(ctl, "tts_ready", lambda url: bool(ready))
    monkeypatch.setattr(ctl, "spawn_exe", spawn_exe)
    args = ctl.argparse.Namespace(config=cfg)
    ctl.start_aivis(args, tmp_path)
    assert launched == [["C:/a/run.exe", "--host", "127.0.0.1", "--port", "10199", "--no-use_gpu", "--output_log_utf8"]]
    assert (tmp_path / ctl.AIVIS_PID_FILE).read_text() == str(os.getpid())
    ctl.start_aivis(args, tmp_path)  # 応答中なら二重に起動しない
    assert len(launched) == 1 and "already running" in capsys.readouterr().out

    monkeypatch.setattr(ctl, "kill_tree", killed.append)
    monkeypatch.setattr(ctl, "wait_until", lambda cond, t: True)
    assert ctl.cmd_stop(args, tmp_path) == 0
    assert killed == [os.getpid()] and not (tmp_path / ctl.AIVIS_PID_FILE).exists()


def test_start_aivis_skips_without_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ctl, "tts_ready", lambda url: False)
    monkeypatch.setattr(ctl, "spawn_exe", lambda *a: pytest.fail("must not spawn"))
    ctl.start_aivis(ctl.argparse.Namespace(config=tmp_path / "none.toml"), tmp_path)
    assert "start it manually" in capsys.readouterr().out


def test_set_asr_engine_rewrites_only_asr_section_and_keeps_crlf(tmp_path):
    f = tmp_path / "s.toml"
    f.write_bytes('[aivis]\r\nengine = "C:/AivisSpeech/run.exe"\r\n\r\n[asr]  # 音声認識\r\n'
                  'engine = "reazon"  # 既定\r\nthreads = 2\r\n\r\n[tts]\r\nspeaker = 1\r\n'.encode("utf-8"))
    ctl.set_asr_engine(f, "haas")
    assert f.read_bytes().decode("utf-8") == (
        '[aivis]\r\nengine = "C:/AivisSpeech/run.exe"\r\n\r\n[asr]  # 音声認識\r\n'
        'engine = "haas"  # 既定\r\nthreads = 2\r\n\r\n[tts]\r\nspeaker = 1\r\n')


def test_set_asr_engine_adds_key_or_section(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text("[asr]\nthreads = 4\n", encoding="utf-8")
    ctl.set_asr_engine(f, "haas")
    assert f.read_text(encoding="utf-8") == '[asr]\nengine = "haas"\nthreads = 4\n'
    g = tmp_path / "t.toml"
    g.write_text("[tts]\nspeaker = 1", encoding="utf-8")
    ctl.set_asr_engine(g, "haas")
    assert g.read_text(encoding="utf-8") == '[tts]\nspeaker = 1\n\n[asr]\nengine = "haas"\n'


def test_asr_command_shows_and_switches_without_starting_when_stopped(tmp_path, monkeypatch, capsys):
    f = tmp_path / "s.toml"
    f.write_text('[asr]\nengine = "reazon"\n', encoding="utf-8")
    monkeypatch.setattr(ctl, "start_one", lambda *a: pytest.fail("started while stopped"))
    args = ["--data-dir", str(tmp_path), "--config", str(f)]
    assert ctl.main(["asr", *args]) == 0
    assert "asr: reazon (asr_worker stopped)" in capsys.readouterr().out
    assert ctl.main(["asr", "haas", *args]) == 0
    assert "reazon -> haas" in capsys.readouterr().out
    assert 'engine = "haas"' in f.read_text(encoding="utf-8")


def test_asr_command_restarts_running_worker(tmp_path, monkeypatch, capsys):
    f = tmp_path / "s.toml"
    f.write_text('[asr]\nengine = "reazon"\n', encoding="utf-8")
    (tmp_path / ctl.ASR_PID_FILE).write_text(str(os.getpid()))
    calls = []
    monkeypatch.setattr(ctl, "stop_asr_worker", lambda d: calls.append("stop"))
    monkeypatch.setattr(ctl, "start_one", lambda name, module, args, pid_file, log: calls.append(module) or 1)
    assert ctl.main(["asr", "haas", "--data-dir", str(tmp_path), "--config", str(f)]) == 0
    assert calls == ["stop", "idle_talk.asr_worker"]
