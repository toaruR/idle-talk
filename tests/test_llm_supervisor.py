import asyncio

from mocks import FakeClock

from idle_talk.llm_supervisor import FALLBACK_PHRASE, LlmSupervisor, build_llama_command
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
