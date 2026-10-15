import asyncio
from types import SimpleNamespace

from mocks import FakeClock

from idle_talk.config import LlmLaunch
from idle_talk.llm_supervisor import (
    FALLBACK_PHRASE,
    NO_THINK_KWARGS,
    LlmSupervisor,
    build_llama_argv,
    build_llama_command,
    wait_ready,
    watch,
)
from idle_talk.pipeline import ResponsePipeline


def make():
    clock = FakeClock()
    sleeps, starts, rests, warns = [], [], [], []
    sup = LlmSupervisor(lambda: starts.append(clock()) or object(), clock=clock,
                        sleep=lambda s: (sleeps.append(s), clock.advance(s)),
                        request_rest=lambda: rests.append(1), warn=warns.append)
    return sup, clock, sleeps, starts, rests, warns


def test_backoff_1_2_4_then_rest_on_4th():
    sup, _clock, sleeps, starts, rests, warns = make()
    assert [sup.on_crash() for _ in range(3)] == ["restarted"] * 3
    assert sleeps == [1, 2, 4]
    assert len(starts) == 3 and rests == []
    assert sup.on_crash() == "rest"
    assert len(starts) == 3 and rests == [1] and len(warns) == 1


def test_window_reset_after_10_minutes():
    sup, clock, sleeps, _starts, rests, _warns = make()
    for _ in range(3):
        sup.on_crash()
    clock.advance(601)
    assert sup.on_crash() == "restarted"
    assert sleeps[-1] == 1
    assert rests == []


def test_kill_gives_fallback_phrase_within_http_timeout():
    clock = FakeClock()

    def llm(m):
        raise ConnectionRefusedError("llama-server killed")
        yield

    audio, emits = [], []
    p = ResponsePipeline(llm, lambda t: b"w", lambda: None,
                         lambda ep, u, c, t, w: audio.append((t, clock())), emits.append)
    asyncio.run(p.respond([]))
    assert audio[0][0] == FALLBACK_PHRASE
    assert audio[0][1] <= 5  # HTTP_TIMEOUT.llm_first
    assert any(e["t"] == "error" and e["code"] == 5001 for e in emits)


def test_kill_then_restart_ready_for_next_turn():
    sup, _clock, sleeps, _starts, _rests, _warns = make()
    assert sup.on_crash() == "restarted"
    assert sleeps == [1] and sup.proc is not None


def test_llama_command_disables_thinking():
    cmd = build_llama_command("model.gguf")
    assert "--jinja" in cmd
    assert "enable_thinking" in cmd and "false" in cmd


class FakeProc:
    def __init__(self):
        self.code = None
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = 1

    def wait(self, timeout=None):
        return self.code


def make_procs():
    clock = FakeClock()
    procs, rests = [], []

    def start():
        procs.append(FakeProc())
        return procs[-1]
    sup = LlmSupervisor(start, clock=clock, sleep=clock.advance, request_rest=lambda: rests.append(1))
    sup.start()
    return sup, procs, rests


def test_watch_restarts_on_exit_and_returns_rest_over_limit():
    sup, procs, rests = make_procs()
    ticks = []

    async def sleep(s):
        ticks.append(s)
        if len(ticks) % 2 == 0:  # 2 回に 1 回、監督中のプロセスを落とす
            procs[-1].code = -9

    assert asyncio.run(watch(sup, sleep=sleep)) == "rest"
    assert len(procs) == 4 and rests == [1]  # 初回起動 + 再起動 3 回、4 回目の終了で rest


def test_exited_and_stop():
    sup, procs, _rests = make_procs()
    assert not sup.exited()
    sup.stop()
    assert procs[0].terminated and sup.exited()
    assert not LlmSupervisor(lambda: None).exited()  # 未起動


def test_llm_watch_requests_director_rest(monkeypatch):
    from idle_talk import main as main_mod
    rests = []

    async def fake_watch(sup):
        return "rest"
    monkeypatch.setattr(main_mod, "watch", fake_watch)
    asyncio.run(main_mod.llm_watch(SimpleNamespace(request_rest=lambda: rests.append(1)), object()))
    assert rests == [1]


def test_build_llama_argv_matches_command():
    argv = build_llama_argv(LlmLaunch(server="srv.exe", model="m.gguf", ngl=10, ctx=2048), 8091)
    assert argv[0] == "srv.exe" and argv[argv.index("-m") + 1] == "m.gguf"
    assert argv[argv.index("--port") + 1] == "8091" and argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("-ngl") + 1] == "10" and argv[argv.index("-c") + 1] == "2048"
    assert "--jinja" in argv and argv[argv.index("--chat-template-kwargs") + 1] == NO_THINK_KWARGS
    assert argv[argv.index("-sm") + 1] == "none" and argv[argv.index("-mg") + 1] == "0"  # 既定は GPU0 固定


def test_build_llama_argv_gpu_choice():
    argv = build_llama_argv(LlmLaunch(model="m.gguf", main_gpu=1), 8091)
    assert argv[argv.index("-sm") + 1] == "none" and argv[argv.index("-mg") + 1] == "1"
    auto = build_llama_argv(LlmLaunch(model="m.gguf", main_gpu=-1), 8091)
    assert "-sm" not in auto and "-mg" not in auto  # 自動配置は llama-server に任せる


def test_build_llama_argv_vision():
    argv = build_llama_argv(LlmLaunch(model="m.gguf", mmproj="p.gguf", image_max_tokens=256), 8091)
    assert argv[argv.index("--mmproj") + 1] == "p.gguf" and argv[argv.index("--image-max-tokens") + 1] == "256"
    plain = build_llama_argv(LlmLaunch(model="m.gguf", image_max_tokens=256), 8091)
    assert "--mmproj" not in plain and "--image-max-tokens" not in plain  # mmproj 無しなら画像の指定もしない
    assert "--image-max-tokens" not in build_llama_argv(LlmLaunch(model="m.gguf", mmproj="p.gguf"), 8091)


def test_has_vision_reads_props(monkeypatch):
    import io
    import json as json_mod

    from idle_talk import llm_supervisor as sup_mod
    bodies = {"on": {"modalities": {"vision": True}}, "off": {"modalities": {"vision": False}}, "old": {}}

    def fake_open(url, timeout):
        assert url.endswith("/props")
        key = url.split("//")[1].split("/")[0]
        if key == "down":
            raise OSError("refused")
        return io.BytesIO(json_mod.dumps(bodies[key]).encode()) if key in bodies else io.BytesIO(b"<html>")
    monkeypatch.setattr(sup_mod.urllib.request, "urlopen", fake_open)
    assert sup_mod.has_vision("http://on/")
    assert not any(sup_mod.has_vision(f"http://{k}") for k in ("off", "old", "down", "bad"))


def test_llama_env_orders_gpus_by_pci_bus():
    from idle_talk.llm_supervisor import llama_env
    env = llama_env()
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and "PATH" in {k.upper() for k in env}


def test_wait_ready_polls_until_ready_or_timeout():
    clock = FakeClock()
    answers = iter([False, False, True])
    assert wait_ready("u", 10, 0.5, clock, clock.advance, lambda u: next(answers))
    assert clock() == 1.0
    clock2 = FakeClock()
    assert not wait_ready("u", 2, 0.5, clock2, clock2.advance, lambda u: False)


def test_start_llm_skips_when_unconfigured_or_already_running(monkeypatch, tmp_path):
    from idle_talk import main as main_mod
    from idle_talk.config import Config
    args = SimpleNamespace(config=str(tmp_path / "none.toml"), llm_url="http://127.0.0.1:8091")
    assert main_mod.start_llm(args, tmp_path, Config()) is None  # [llm].model 未設定 = 手動起動
    f = tmp_path / "s.toml"
    f.write_text('[llm]\nmodel = "m.gguf"\n', encoding="utf-8")
    args.config = str(f)
    monkeypatch.setattr(main_mod, "is_ready", lambda url: True)
    assert main_mod.start_llm(args, tmp_path, Config()) is None  # 既に応答中なら二重起動しない


def test_start_llm_launches_and_waits(monkeypatch, tmp_path):
    from idle_talk import main as main_mod
    from idle_talk.config import Config
    f = tmp_path / "s.toml"
    f.write_text('[llm]\nserver = "srv.exe"\nmodel = "m.gguf"\n', encoding="utf-8")
    seen = {}
    monkeypatch.setattr(main_mod, "is_ready", lambda url: False)
    monkeypatch.setattr(main_mod, "popen_starter", lambda argv, log: seen.update(argv=argv, log=log) or FakeProc)
    monkeypatch.setattr(main_mod, "wait_ready", lambda url, t: seen.update(wait=(url, t)) or True)
    sup = main_mod.start_llm(SimpleNamespace(config=str(f), llm_url="http://127.0.0.1:8091"), tmp_path, Config())
    assert isinstance(sup.proc, FakeProc)
    assert seen["argv"][:3] == ["srv.exe", "-m", "m.gguf"] and "8091" in seen["argv"]
    assert seen["log"] == tmp_path / "llama-server.log"
    assert seen["wait"] == ("http://127.0.0.1:8091", main_mod.LLM_READY_TIMEOUT_S)
