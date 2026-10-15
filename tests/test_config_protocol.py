import copy
import dataclasses
import tomllib
from pathlib import Path

import pytest

from idle_talk import errors, protocol
from idle_talk.config import Config

ROOT = Path(__file__).resolve().parent.parent


def test_defaults_time():
    c = Config()
    assert (c.T_silence, c.T_confirm, c.MIN_SEG, c.SIL_BACKCHANNEL) == (400, 400, 200, 300)
    assert (c.BC_COOLDOWN, c.PING_INTERVAL, c.IDLE_TIMEOUT) == (5, 5, 1800)


def test_defaults_perf():
    c = Config()
    assert (c.TTS_GPU_THRESH, c.OVERLAP_MAX, c.VAR_SUM, c.G2_SUM) == (300, 300, 750, 130)
    assert (c.TTS_RATIO_MAX, c.DUCK_GAIN) == (0.7, 0.2)


def test_defaults_structs():
    c = Config()
    assert (c.LLM_RETRY.max, c.LLM_RETRY.window_s, c.LLM_RETRY.backoff_s) == (3, 600, (1, 2, 4))
    assert (c.STRONG_UTT.volume_ratio, c.STRONG_UTT.window_s) == (2.0, 1.0)
    assert (c.ASR_HALLU.no_speech_prob, c.ASR_HALLU.compression_ratio) == (0.6, 2.4)
    h = c.HTTP_TIMEOUT
    assert (h.llm_first, h.llm_total, h.tts_sentence) == (5, 20, 3)
    assert h.asr(3.0) == 8.0
    e = c.E_PARAMS
    assert (e.silence_input_s, e.device_lost_s, e.tts_cpu_settle_s, e.tts_cpu_delta_pt) == (60, 3, 2, 5)
    assert (c.TEST_N, c.DB_BACKUP, c.N_TURNS, c.RATIO_Q) == (20, 1, 8, (1, 3))
    assert c.STOP_WORDS == ("待って", "違う", "ちょっと", "ストップ")


def test_defaults_spontaneous():
    c = Config()
    assert c.spontaneous_enabled is False
    assert (c.spont_interval_s, c.quiet_start, c.quiet_end) == (1200, "22:00", "07:00")


@pytest.mark.parametrize("v", [9, 3601, 1.5, True])
def test_spont_interval_range(v):
    with pytest.raises(ValueError):
        Config(spont_interval_s=v)
    assert Config(spont_interval_s=10).spont_interval_s == 10


def test_defaults_field_names():
    expected = {
        "T_silence", "TTS_GPU_THRESH", "PING_INTERVAL", "LLM_RETRY", "IDLE_TIMEOUT", "STRONG_UTT",
        "OVERLAP_MAX", "HTTP_TIMEOUT", "DUCK_GAIN", "VAR_SUM", "G2_SUM", "TTS_RATIO_MAX",
        "TEST_N", "E_PARAMS", "DB_BACKUP", "MIN_SEG", "T_confirm", "STOP_WORDS", "N_TURNS",
        "SIL_BACKCHANNEL", "BC_COOLDOWN", "RATIO_Q", "ASR_HALLU",
        "spontaneous_enabled", "spont_interval_s", "quiet_start", "quiet_end", "radio",
    }
    assert len(expected) == 28
    assert {f.name for f in dataclasses.fields(Config)} == expected


def test_range_t_confirm():
    for v in (300, 500):
        assert Config(T_confirm=v).T_confirm == v
    for v in (299, 501):
        with pytest.raises(ValueError):
            Config(T_confirm=v)


def test_range_sil_backchannel():
    for v in (250, 350):
        assert Config(SIL_BACKCHANNEL=v).SIL_BACKCHANNEL == v
    for v in (249, 351):
        with pytest.raises(ValueError):
            Config(SIL_BACKCHANNEL=v)
    with pytest.raises(ValueError):
        Config(T_silence=300, SIL_BACKCHANNEL=300)
    with pytest.raises(ValueError):
        Config(T_silence=250, SIL_BACKCHANNEL=300)


POSITIVE = [
    {"t": "hello", "version": 1, "sample_rate": 16000, "client_id": "abc"},
    {"t": "playback", "epoch": 1, "utterance_id": 2, "chunk_id": 3, "position_ms": 600000, "state": "started"},
    {"t": "ui", "mute": True, "on": False, "radio": True, "interval_s": 3600, "quiet_start": "22:00", "quiet_end": "07:00"},
    {"t": "ping", "ts": 123},
    {"t": "pong", "ts": 123},
    {"t": "duck", "epoch": 1, "gain": 0.2},
    {"t": "cancel", "new_epoch": 2},
    {"t": "resume", "epoch": 2},
    {"t": "state", "state": "idle", "muted": False},
    {"t": "error", "code": 5001, "message": "x"},
]


def test_schema_positive():
    assert len(POSITIVE) == 10
    for m in POSITIVE:
        assert protocol.validate_message(m), m


def test_schema_negative():
    neg = [
        {"t": "ui", "interval_s": 3601},
        {"t": "ui", "interval_s": 9},
        {"t": "playback", "epoch": 1, "utterance_id": 2, "chunk_id": 3, "position_ms": 600001, "state": "started"},
        {"t": "error", "code": 3999, "message": "x"},
        {"t": "state", "state": "foo", "muted": False},
        {"t": "hello", "version": 1, "sample_rate": 44100, "client_id": "a"},
    ]
    for m in neg:
        assert not protocol.validate_message(m), m


def test_schema_error_code_range():
    for c, ok in ((4000, True), (5999, True), (3999, False), (6000, False)):
        assert protocol.validate_message({"t": "error", "code": c, "message": "m"}) is ok


def test_schema_required_keys():
    seen = set()
    for m in POSITIVE:
        required = set(next(
            s["required"] for s in protocol.MESSAGE_SCHEMA["oneOf"]
            if m["t"] in (s["properties"]["t"].get("enum") or [s["properties"]["t"].get("const")])
        )) - {"t"}
        for k in required:
            bad = copy.deepcopy(m)
            del bad[k]
            assert not protocol.validate_message(bad), (m["t"], k)
            seen.add(k)
    expected = {"version", "sample_rate", "client_id", "epoch", "utterance_id", "chunk_id",
                "position_ms", "state", "ts", "gain", "new_epoch", "muted", "code", "message"}
    assert expected <= seen
    # ui の mute/on/quiet_* は required に無い(スキーマ上 t のみ必須)。欠落は t のみ検証
    assert not protocol.validate_message({"mute": True})


def test_state_names():
    assert list(protocol.STATE_NAMES_JA) == [
        "idle", "user_speaking", "eot_check", "preparing", "ai_playing", "interrupt_check", "rest"]
    assert len(set(protocol.STATE_NAMES_JA.values())) == 7
    assert protocol.STATE_NAMES_JA["idle"] == "待機" and protocol.STATE_NAMES_JA["rest"] == "休止"


def test_header_roundtrip():
    h = protocol.pack_down_header(1, 2, 3, 4)
    assert len(h) == 16
    assert protocol.unpack_down_header(h) == (1, 2, 3, 4)
    m = protocol.U32_MAX
    assert protocol.unpack_down_header(protocol.pack_down_header(m, m, m, m)) == (m, m, m, m)
    for i in range(4):
        args = [0, 0, 0, 0]
        args[i] = m + 1
        with pytest.raises(ValueError):
            protocol.pack_down_header(*args)


def test_up_seq_bounds():
    assert protocol.unpack_up_frame(protocol.pack_up_frame(4294967295, b"ab")) == (4294967295, b"ab")
    with pytest.raises(ValueError):
        protocol.pack_up_frame(4294967296, b"")


def test_chunk_size_limit():
    assert protocol.pack_down_chunk(0, 0, 0, bytes(1048576))
    with pytest.raises(ValueError):
        protocol.pack_down_chunk(0, 0, 0, bytes(1048577))


def test_error_codes():
    assert (errors.CLOSE_4001, errors.CLOSE_VERSION_MISMATCH, errors.CLOSE_ORIGIN_DENIED) == (4001, 4002, 4003)
    assert (errors.ERR_LLM_TIMEOUT, errors.ERR_TTS_FAILED, errors.ERR_ASR_FAILED,
            errors.ERR_DB, errors.ERR_MIC_LOST, errors.ERR_OUT_OF_RANGE) == (5001, 5002, 5003, 5004, 5005, 5006)
    assert errors.EXIT_DUPLICATE_LAUNCH == 4


def test_pytest_config():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("integration:") for m in cfg["markers"])
    assert cfg["addopts"] == "-m 'not integration' --strict-markers"


def test_load_ports_defaults_when_missing(tmp_path):
    from idle_talk.config import Ports, load_ports
    assert load_ports(tmp_path / "none.toml") == Ports() == (8765, 8090, 8080, 10101, 8000)


def test_load_ports_partial_override(tmp_path):
    from idle_talk.config import load_ports
    f = tmp_path / "s.toml"
    f.write_text("[ports]\nws = 9000\ntts = 50021\n", encoding="utf-8")
    assert load_ports(f)._asdict() == {"ws": 9000, "asr": 8090, "llm": 8080, "tts": 50021, "web": 8000}


@pytest.mark.parametrize("body", ["[ports]\nwss = 1\n", "[ports]\nws = \"1\"\n", "[ports]\nws = 70000\n",
                                  "[ports]\nws = true\n"])
def test_load_ports_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_ports
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_ports(f)


def test_repo_settings_file_is_valid():
    from idle_talk.config import DEFAULT_SPEAKER, SETTINGS_FILE, load_ports, load_speaker
    assert DEFAULT_SPEAKER == 888753760
    load_ports(ROOT / SETTINGS_FILE)  # 利用者が値を変えてもよいので、読めることだけ確認
    load_speaker(ROOT / SETTINGS_FILE)
    from idle_talk.config import load_voice
    load_voice(ROOT / SETTINGS_FILE)
    from idle_talk.config import load_spontaneous
    Config(**load_spontaneous(ROOT / SETTINGS_FILE))
    from idle_talk.config import load_topics
    load_topics(ROOT / SETTINGS_FILE)
    from idle_talk.config import load_vad
    load_vad(ROOT / SETTINGS_FILE)


def test_load_speaker(tmp_path):
    from idle_talk.config import DEFAULT_SPEAKER, load_speaker
    f = tmp_path / "s.toml"
    assert load_speaker(f) == DEFAULT_SPEAKER
    f.write_text("[tts]\nspeaker = 1234\n", encoding="utf-8")
    assert load_speaker(f) == 1234


@pytest.mark.parametrize("body", ["[tts]\nspeeker = 1\n", "[tts]\nspeaker = \"1\"\n", "[tts]\nspeaker = -1\n"])
def test_load_speaker_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_speaker
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_speaker(f)


def test_load_voice(tmp_path):
    from idle_talk.config import load_speaker, load_voice
    f = tmp_path / "s.toml"
    assert load_voice(f) == {}
    f.write_text("[tts]\nspeaker = 7\nspeed = 1.3\npitch = -0.05\ntempo_dynamics = 2\npost_silence = 0\n",
                 encoding="utf-8")
    assert load_speaker(f) == 7
    assert load_voice(f) == {"speedScale": 1.3, "pitchScale": -0.05, "tempoDynamicsScale": 2.0,
                             "postPhonemeLength": 0.0}


@pytest.mark.parametrize("body", ["[tts]\nspeed = 0.4\n", "[tts]\nspeed = 2.1\n", "[tts]\nspeed = \"1\"\n",
                                  "[tts]\nspeed = true\n", "[tts]\npitch = 0.2\n", "[tts]\npause = -1\n",
                                  "[tts]\nspeedScale = 1.0\n"])
def test_load_voice_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_voice
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_voice(f)


def test_load_spontaneous(tmp_path):
    from idle_talk.config import load_spontaneous
    f = tmp_path / "s.toml"
    assert load_spontaneous(f) == {}
    f.write_text('[spontaneous]\non = true\ninterval_s = 30\nquiet_start = "23:00"\nradio = true\n', encoding="utf-8")
    kw = load_spontaneous(f)
    assert kw == {"spontaneous_enabled": True, "spont_interval_s": 30, "quiet_start": "23:00", "radio": True}
    c = Config(**kw)
    assert (c.spontaneous_enabled, c.spont_interval_s, c.quiet_end) == (True, 30, "07:00")


@pytest.mark.parametrize("body", ["[spontaneous]\non = 1\n", "[spontaneous]\ninterval_s = 30.0\n",
                                  "[spontaneous]\nfreq_per_hour = 3\n", "[spontaneous]\nquiet_start = 2300\n",
                                  "[spontaneous]\nradio = 1\n"])
def test_load_spontaneous_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_spontaneous
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_spontaneous(f)


def test_load_llm_launch(tmp_path):
    from idle_talk.config import LlmLaunch, load_llm_launch
    f = tmp_path / "s.toml"
    assert load_llm_launch(f) == LlmLaunch() and LlmLaunch().model == ""  # 既定は監督なし
    f.write_text('[llm]\nmodel = "m.gguf"\nctx = 2048\n', encoding="utf-8")
    assert load_llm_launch(f) == LlmLaunch(model="m.gguf", ctx=2048)
    assert LlmLaunch().main_gpu == 0  # 既定は GPU0 だけに載せる
    f.write_text('[llm]\nmain_gpu = -1\n', encoding="utf-8")
    assert load_llm_launch(f).main_gpu == -1  # 自動配置
    f.write_text('[llm]\nmmproj = "p.gguf"\nimage_max_tokens = 256\n', encoding="utf-8")
    assert load_llm_launch(f)[-2:] == ("p.gguf", 256)


@pytest.mark.parametrize("body", ['[llm]\nmodle = "m"\n', "[llm]\nmodel = 1\n", '[llm]\nngl = "9"\n',
                                  "[llm]\nctx = -1\n", "[llm]\nmain_gpu = -2\n",
                                  "[llm]\nmmproj = 1\n", "[llm]\nimage_max_tokens = -1\n"])
def test_load_llm_launch_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_llm_launch
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_llm_launch(f)


def test_load_topics(tmp_path):
    from idle_talk.config import Topics, load_topics
    f = tmp_path / "s.toml"
    assert load_topics(f) == Topics((), 30)
    f.write_text('[topics]\nfeeds = ["https://example.test/rss"]\ninterval_min = 60\n', encoding="utf-8")
    assert load_topics(f) == Topics(("https://example.test/rss",), 60)


@pytest.mark.parametrize("body", ['[topics]\nfeeds = "https://example.test/rss"\n', '[topics]\nfeeds = ["file:///x"]\n',
                                  "[topics]\ninterval_min = 0\n", "[topics]\ninterval = 30\n"])
def test_load_topics_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_topics
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_topics(f)


def test_load_vad(tmp_path):
    from idle_talk.config import VadSettings, load_vad
    f = tmp_path / "s.toml"
    assert load_vad(f) == VadSettings(500.0, 100, True, 3.0, 100.0)
    f.write_text("[vad]\nthreshold = 800\nonset_ms = 160\nadaptive = false\nratio = 4\nmin_threshold = 50\n",
                 encoding="utf-8")
    assert load_vad(f) == VadSettings(800.0, 160, False, 4.0, 50.0)


@pytest.mark.parametrize("body", ["[vad]\nthreshold = 0\n", "[vad]\nthreshold = \"500\"\n", "[vad]\nonset_ms = 10\n",
                                  "[vad]\nonset_ms = 100.0\n", "[vad]\nonset = 100\n", "[vad]\nadaptive = 1\n",
                                  "[vad]\nratio = 1\n", "[vad]\nmin_threshold = -1\n"])
def test_load_vad_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_vad
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_vad(f)


def test_load_aivis_launch(tmp_path):
    from idle_talk.config import AivisLaunch, load_aivis_launch
    f = tmp_path / "s.toml"
    assert load_aivis_launch(f) == AivisLaunch() and AivisLaunch().engine == ""  # 既定は手動起動
    f.write_text('[aivis]\nengine = "run.exe"\nuse_gpu = true\n', encoding="utf-8")
    assert load_aivis_launch(f) == AivisLaunch(engine="run.exe", use_gpu=True)


@pytest.mark.parametrize("body", ['[aivis]\nengin = "r"\n', "[aivis]\nengine = 1\n", "[aivis]\nuse_gpu = 1\n"])
def test_load_aivis_launch_rejects_invalid(tmp_path, body):
    from idle_talk.config import load_aivis_launch
    f = tmp_path / "s.toml"
    f.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError):
        load_aivis_launch(f)
