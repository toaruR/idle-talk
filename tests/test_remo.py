import asyncio
import io
import json
import threading
import urllib.error

import pytest

from idle_talk.pipeline import ResponsePipeline
from idle_talk.remo import (LocalRemo, RemoControl, _NoRedirect, load_control, main,
                            parse_intent, validate_profile, validate_signal)

SIGNAL = {"freq": 38, "data": [2500, 1200, 600], "format": "us"}


def intent(device="ac", action="set", value=26, mode="cool", unit="degree"):
    return dict(device=device, action=action, value=value, mode=mode, unit=unit)


class Device:
    def __init__(self):
        self.sent = []
        self.fail = False

    def send(self, signal):
        self.sent.append(signal)
        if self.fail:
            raise TimeoutError()


def make(keys=("ac:cool:26", "ac:cool:25", "ac:heat:26", "ac:off", "light:level:50", "light:level:100"), **kw):
    device = Device()
    control = RemoControl(device, {k: dict(SIGNAL, data=[n + 1]) for n, k in enumerate(keys)}, **kw)
    return control, device


def call(control, i, text="エアコンを操作して", current=lambda: True):
    return control(text, lambda messages: [json.dumps(i)], current)


def test_absolute_and_relative_ac_send_registered_complete_settings():
    control, device = make()
    assert "冷房26度" in call(control, intent())
    assert "冷房25度" in call(control, intent(action="adjust", value=-1, mode=None), "暑いから温度下げて")
    assert device.sent == [control.signals["ac:cool:26"], control.signals["ac:cool:25"]]
    assert "冷房26度" in call(control, intent(value=26, mode=None))


def test_unknown_expired_and_off_state_do_not_guess_temperature():
    now = [100]
    control, device = make(clock=lambda: now[0], state_ttl_s=10)
    relative = intent(action="adjust", value=-1, mode=None)
    assert "分かりません" in call(control, relative)
    call(control, intent())
    now[0] += 10
    assert "分かりません" in call(control, relative)
    call(control, intent())
    call(control, intent(action="off", value=None, mode=None, unit=None))
    assert "分かりません" in call(control, relative)
    assert len(device.sent) == 3


def test_relative_retains_heat_mode_and_does_not_change_mode_implicitly():
    control, device = make(("ac:heat:26", "ac:heat:25", "ac:cool:25"))
    call(control, intent(mode="heat"))
    assert "暖房25度" in call(control, intent(action="adjust", value=-1, mode=None))
    before = len(device.sent)
    assert "分かりません" in call(control, intent(action="adjust", value=1, mode="cool"))
    assert len(device.sent) == before


def test_light_absolute_steps_percent_and_unknown_state():
    control, device = make(("light:level:25", "light:level:50", "light:level:100", "light:down"))
    dim = intent("light", "adjust", -1, None, "step")
    assert "一段階暗く" in call(control, dim, "ちょっと暗くして")
    call(control, intent("light", "set", 100, None, "percent"))
    assert "50%" in call(control, dim)
    assert "25%" in call(control, intent("light", "adjust", -25, None, "percent"))
    assert len(device.sent) == 4
    assert "一段階暗く" in call(control, dim)
    assert len(device.sent) == 5
    fresh, new_device = make()
    assert "light:down" in call(fresh, dim)
    assert not new_device.sent


def test_single_absolute_light_level_does_not_block_registered_dimmer_button():
    control, device = make(("light:level:100", "light:down"))
    call(control, intent("light", "set", 100, None, "percent"))
    assert "一段階暗く" in call(control, intent("light", "adjust", -1, None, "step"))
    assert device.sent == [control.signals["light:level:100"], control.signals["light:down"]]
    assert "分かりません" in call(control, intent("light", "adjust", -10, None, "percent"))


@pytest.mark.parametrize("i", [intent(value=33), intent(value=26.2), intent(mode=None),
    intent(action="adjust", value=0), intent(action="adjust", value=6),
    intent("light", "set", 0, None, "percent"), intent("light", "set", 101, None, "percent"),
    intent("light", "set", 50, "cool", "percent"), intent(action="on", value=26, unit="degree"),
    intent(action="clarify"), intent(action="set", value=None)])
def test_ambiguous_or_unsupported_intent_does_not_send(i):
    control, device = make()
    assert call(control, i)
    assert not device.sent


def test_unregistered_profile_and_timeout_do_not_claim_success_or_retry():
    control, device = make()
    assert "未登録" in call(control, intent(value=24))
    assert not device.sent
    call(control, intent())
    device.fail = True
    assert "届いた可能性" in call(control, intent(value=25))
    assert len(device.sent) == 2
    device.fail = False
    assert "分かりません" in call(control, intent(action="adjust", value=-1, mode=None))
    assert len(device.sent) == 2


@pytest.mark.parametrize("text", ['{}', 'done', json.dumps(intent(value=True)),
    json.dumps(intent(value=float("nan"))), json.dumps(dict(intent(), url="http://outside/")),
    json.dumps(intent(action="shell")), json.dumps([intent()])])
def test_invalid_llm_output_is_rejected(text):
    control, device = make()
    answer = control("照明消して", lambda m: [text], lambda: True)
    assert "読み取れません" in answer
    assert not device.sent


def test_normal_conversation_does_not_classify_or_send():
    control, device = make()
    assert control("こんにちは", lambda m: pytest.fail("should not call LLM"), lambda: True) is None
    none = intent("none", "none", None, None, None)
    assert call(control, none, "照明の仕組みを教えて") is None
    assert not device.sent


def test_cancellation_while_classifying_prevents_network_send():
    control, device = make()
    active = [True]

    def llm(messages):
        yield '{"device":'
        active[0] = False
        yield '"ac"}'

    assert control("冷房26度にして", llm, lambda: active[0]) == ""
    assert not device.sent


def test_pipeline_uses_latest_user_text_and_speaks_actual_result():
    control, device = make()
    seen, audio = [], []

    def llm(messages):
        seen.extend(messages)
        yield json.dumps(intent())

    p = ResponsePipeline(llm, lambda t: b"wav", lambda: None,
                         lambda ep, u, c, t, w: audio.append(t), lambda m: None)
    messages = [{"role": "system", "content": "past context"},
                {"role": "user", "content": "照明消して"},
                {"role": "assistant", "content": "古い会話"},
                {"role": "user", "content": "冷房26度にして"}]
    asyncio.run(p.respond(messages, control=control))
    assert len(device.sent) == 1
    assert audio == ["冷房26度の信号を送りました。"]
    assert seen[-1] == messages[-1]
    assert "past context" not in str(seen)


def test_pipeline_cancel_during_intent_prevents_sending_and_audio():
    started, release = threading.Event(), threading.Event()
    control, device = make()
    audio = []

    def llm(messages):
        started.set()
        release.wait(5)
        yield json.dumps(intent())

    p = ResponsePipeline(llm, lambda t: b"wav", lambda: None,
                         lambda *args: audio.append(args), lambda m: None)

    async def run():
        task = asyncio.create_task(p.respond([{"role": "user", "content": "冷房26度にして"}], control=control))
        await asyncio.to_thread(started.wait, 2)
        await p.cancel()
        release.set()
        await task

    asyncio.run(run())
    assert not device.sent and not audio


def test_non_command_falls_back_to_chat_with_one_system_message():
    control, device = make()
    calls, audio = [], []

    def llm(messages):
        calls.append(messages)
        yield json.dumps(intent("none", "none", None, None, None)) if len(calls) == 1 else "説明します。"

    p = ResponsePipeline(llm, lambda t: b"wav", lambda: None,
                         lambda ep, u, c, t, w: audio.append(t), lambda m: None)
    original = [{"role": "system", "content": "人物設定"}, {"role": "user", "content": "照明の仕組みは？"}]
    asyncio.run(p.respond(original, control=control))
    assert len([m for m in calls[1] if m["role"] == "system"]) == 1
    assert "今回は家電操作を実行していません" in calls[1][0]["content"]
    assert original[0]["content"] == "人物設定"
    assert audio == ["説明します。"] and not device.sent


def test_cancel_before_first_iterator_step_does_not_send_old_request():
    control, device = make()
    p = ResponsePipeline(lambda m: [json.dumps(intent())], lambda t: b"wav", lambda: None,
                         lambda *args: None, lambda m: None)
    started, release = threading.Event(), threading.Event()
    original = p._llm_clauses

    def delayed(*args):
        started.set()
        release.wait(5)
        yield from original(*args)

    p._llm_clauses = delayed

    async def run():
        task = asyncio.create_task(p.respond([{"role": "user", "content": "冷房26度にして"}], control=control))
        await asyncio.to_thread(started.wait, 2)
        await p.cancel()
        release.set()
        await task

    asyncio.run(run())
    assert not device.sent


def test_burst_voice_commands_cancel_stale_operations_and_keep_accepting_audio(tmp_path):
    from test_integration_interrupt import asr_settled, make as make_director, state, until

    async def run():
        phrases = ["照明消して", "照明つけて", "照明消して", "照明つけて", "照明消して", "照明つけて"]
        d, browser, store = make_director(tmp_path, texts=list(phrases))
        control, device = make(("light:on", "light:off"))
        d.remo = control
        entered, release = threading.Event(), threading.Event()
        calls = []

        def classify(messages):
            calls.append(messages[-1]["content"])
            if len(calls) == 1:
                entered.set()
                release.wait(5)
            action = "on" if "つけて" in messages[-1]["content"] else "off"
            yield json.dumps(intent("light", action, None, None, None))

        d.pipeline._llm = classify
        try:
            browser.speak()
            await asr_settled(d)
            await until(entered.is_set)
            for _ in phrases[1:]:
                browser.speak()
                await asr_settled(d)
                await asyncio.sleep(0.01)
            release.set()
            await until(lambda: state(d) == "idle" and len(device.sent) == 1)
            assert device.sent == [control.signals["light:on"]]
            d.texts_q.append("照明消して")
            browser.speak()
            await asr_settled(d)
            await until(lambda: state(d) == "idle" and len(device.sent) == 2)
            assert device.sent[-1] == control.signals["light:off"]
            assert not d.ingress.asr_pending
            assert len(store.query("SELECT id FROM turns WHERE role='user'")) == 7
        finally:
            release.set()
            if d._task and not d._task.done():
                d._task.cancel()
                await asyncio.gather(d._task, return_exceptions=True)
            store.close()

    asyncio.run(run())


def test_ctl_forwards_remo_arguments_without_starting_services(monkeypatch):
    from idle_talk import ctl

    seen = []
    monkeypatch.setattr("idle_talk.remo.main", lambda args: seen.append(args) or 0)
    assert ctl.main(["remo", "--config", "other/remo.toml", "learn", "ac:cool:26"]) == 0
    assert seen == [["--config", "other/remo.toml", "learn", "ac:cool:26"]]


def test_local_transport_uses_json_required_header_and_no_post_retry():
    client = LocalRemo("192.168.11.123")
    calls = []

    class Reply(io.BytesIO):
        status = 200

    class Opener:
        def open(self, req, *, timeout):
            calls.append(req)
            assert timeout == 5
            return Reply(json.dumps(SIGNAL).encode() if req.data is None else b"")

    client._opener = Opener()
    assert client.receive() == SIGNAL
    client.send(SIGNAL)
    assert [c.get_method() for c in calls] == ["GET", "POST"]
    assert calls[1].get_header("X-requested-with") == "local"
    assert json.loads(calls[1].data) == SIGNAL
    assert _NoRedirect().redirect_request(None, None, 302, "", {}, "http://example.com") is None


@pytest.mark.parametrize("host", ["8.8.8.8", "127.0.0.1", "0.0.0.0", "::1", "example.com", "192.168.0.1/path", "http://192.168.0.1"])
def test_only_explicit_lan_ipv4_address_is_accepted(host):
    with pytest.raises(ValueError):
        LocalRemo(host)


@pytest.mark.parametrize("signal", [{}, dict(SIGNAL, freq=True), dict(SIGNAL, freq=0),
    dict(SIGNAL, format="raw"), dict(SIGNAL, data=[]), dict(SIGNAL, data=[False]),
    dict(SIGNAL, data=[-1]), dict(SIGNAL, data="abc")])
def test_bad_received_or_saved_signals_are_rejected(signal):
    with pytest.raises(ValueError):
        validate_signal(signal)


def test_config_opt_in_and_signals_are_loaded_beside_config(tmp_path):
    cfg = tmp_path / "remo.toml"
    assert load_control(cfg) is None
    cfg.write_text('on = false\nhost = "not configured"\n')
    assert load_control(cfg) is None
    cfg.write_text('on = true\nhost = "192.168.11.123"\nstate_ttl_s = 60\n')
    (tmp_path / "remo-signals.json").write_text(json.dumps({"ac:cool:26": SIGNAL}))
    control = load_control(cfg)
    assert control.ttl == 60 and control.signals == {"ac:cool:26": SIGNAL}
    cfg.write_text('on = true\nhost = "192.168.11.123"\nstate_ttl_s = nan\n')
    with pytest.raises(ValueError):
        load_control(cfg)


def test_learning_requires_fresh_signal_and_never_sends(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "remo.toml"
    cfg.write_text('host = "192.168.11.123"\non = false\n')
    readings = iter([SIGNAL, dict(SIGNAL, data=[9]), SIGNAL, SIGNAL])
    monkeypatch.setattr(LocalRemo, "receive", lambda self: next(readings))
    monkeypatch.setattr(LocalRemo, "send", lambda *args: pytest.fail("learn must not transmit"))
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert main(["--config", str(cfg), "learn", "ac:cool:26"]) == 0
    assert "on = true" in capsys.readouterr().out
    path = tmp_path / "remo-signals.json"
    before = path.read_bytes()
    assert json.loads(before)["ac:cool:26"]["data"] == [9]
    assert main(["--config", str(cfg), "learn", "ac:cool:25"]) == 1
    assert path.read_bytes() == before


def test_learning_does_not_overwrite_existing_profile(tmp_path, monkeypatch):
    cfg = tmp_path / "remo.toml"
    cfg.write_text('host = "192.168.11.123"\n')
    (tmp_path / "remo-signals.json").write_text(json.dumps({"light:on": SIGNAL}))
    monkeypatch.setattr(LocalRemo, "receive", lambda self: pytest.fail("already registered"))
    assert main(["--config", str(cfg), "learn", "light:on"]) == 1


@pytest.mark.parametrize("key", ["ac:cool:32.5", "ac:heat:15", "light:level:0", "light:level:101", "../file"])
def test_invalid_profile_name_is_rejected(key):
    with pytest.raises(ValueError):
        validate_profile(key)
