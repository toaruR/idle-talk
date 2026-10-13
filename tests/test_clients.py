import json
import threading
import urllib.error
import urllib.request

import pytest
from jsonschema import Draft7Validator
from mocks import FakeClock, mock_asr_send, mock_llm_stream, mock_tts_post

from idle_talk.asr_worker import handle_request, is_hallucination, make_server
from idle_talk.clients import (
    LLM_MAX_TOKENS,
    LLM_REQ_SCHEMA,
    TTS_RATE,
    AsrClient,
    AsrFailed,
    AsrTimeout,
    LlmClient,
    LlmTimeout,
    TtsClient,
    TtsTimeout,
    WARMUP_TIMEOUT_S,
    build_llm_request,
    parse_sse,
)

MSGS = [{"role": "user", "content": "やあ"}]


def test_llm_request_schema_and_done():
    clk, cap = FakeClock(), []
    c = LlmClient(clock=clk, stream=mock_llm_stream(clk, ["あ", "い"], captured=cap))
    assert "".join(c.stream_chat(MSGS)) == "あい"
    url, body = cap[0]
    assert url.endswith("/v1/chat/completions")
    assert Draft7Validator(LLM_REQ_SCHEMA).is_valid(body)
    assert body["model"] == "local" and body["stream"] is True and body["max_tokens"] == LLM_MAX_TOKENS
    assert set(body) == {"model", "messages", "stream", "max_tokens"}
    # [DONE] で終端: 以降の行は読まない
    lines = ["data: [DONE]\n", 'data: {"choices":[{"delta":{"content":"x"}}]}\n']
    assert list(parse_sse(lines)) == []


def test_llm_bad_message_role_rejected():
    with pytest.raises(ValueError):
        build_llm_request([{"role": "tool", "content": "x"}])


def test_timeout_llm_first_token():
    clk = FakeClock()
    c = LlmClient(clock=clk, stream=mock_llm_stream(clk, ["a"], delays=[5.001]))
    with pytest.raises(LlmTimeout) as e:
        list(c.stream_chat(MSGS))
    assert e.value.code == 5001


def test_timeout_llm_first_token_boundary_ok():
    clk = FakeClock()
    c = LlmClient(clock=clk, stream=mock_llm_stream(clk, ["a"], delays=[5.0]))
    assert list(c.stream_chat(MSGS)) == ["a"]


def test_timeout_llm_total():
    clk = FakeClock()
    c = LlmClient(clock=clk, stream=mock_llm_stream(clk, ["a", "b", "c"], delays=[1, 10, 9.5]))
    with pytest.raises(LlmTimeout) as e:
        list(c.stream_chat(MSGS))
    assert e.value.code == 5001


def test_timeout_tts_sentence():
    clk = FakeClock()
    t = TtsClient(clock=clk, post=mock_tts_post(clk, delay=1.6))  # 2 呼び出しで 3.2s
    with pytest.raises(TtsTimeout) as e:
        t.synthesize("こんにちは")
    assert e.value.code == 5002
    clk2 = FakeClock()
    ok = TtsClient(clock=clk2, post=mock_tts_post(clk2, delay=1.5))  # 3.0s は許容
    assert ok.synthesize("a").startswith(b"RIFF")


def test_synthesis_endpoint_and_skip_failed_sentence():
    clk = FakeClock()
    post = mock_tts_post(clk, fail_texts=("二文目",))
    t = TtsClient(clock=clk, post=post)
    out = t.synthesize_sentences(["一文目", "二文目", "三文目"])
    assert [w is not None for _, w in out] == [True, False, True]
    assert any("/synthesis?" in u for u in post.calls)
    assert not any("/cancellable_synthesis" in u for u in post.calls)  # AivisSpeech 1.2.0 では 501


def test_llm_null_content_chunk_is_skipped():
    # llama-server は最初のチャンクで delta.content を null にする
    lines = ['data: {"choices":[{"delta":{"role":"assistant","content":null}}]}\n',
             'data: {"choices":[{"delta":{"content":"やあ"}}]}\n', "data: [DONE]\n"]
    assert list(parse_sse(lines)) == ["やあ"]
    with pytest.raises(ValueError):
        list(parse_sse(['data: {"choices":[{"delta":{"content":1}}]}\n']))


def test_tts_voice_overrides_audio_query():
    sent = []

    def post(url, body, timeout):
        if "audio_query" in url:
            return b'{"speedScale": 1.0, "pitchScale": 0.0}'
        sent.append(json.loads(body))
        return b"RIFF"
    fixed = {"outputSamplingRate": TTS_RATE, "outputStereo": False}
    TtsClient(post=post, voice={"speedScale": 1.4}).synthesize("a")
    assert sent == [{"speedScale": 1.4, "pitchScale": 0.0, **fixed}]
    sent.clear()
    TtsClient(post=post).synthesize("a")
    assert sent == [{"speedScale": 1.0, "pitchScale": 0.0, **fixed}]


def test_tts_rate_matches_web_player():
    import pathlib
    import re
    js = (pathlib.Path(__file__).resolve().parents[1] / "web" / "player.js").read_text(encoding="utf-8")
    assert int(re.search(r"DEFAULT_PCM_RATE = (\d+);", js).group(1)) == TTS_RATE


def test_tts_warmup_uses_long_timeout_and_swallows_failure():
    seen = []

    def post(url, body, timeout):
        seen.append((url.split("?")[0], timeout))
        return b"{}" if "audio_query" in url else b"RIFF"
    assert TtsClient(post=post).warmup() is True
    assert seen == [("http://127.0.0.1:10101/audio_query", WARMUP_TIMEOUT_S),
                    ("http://127.0.0.1:10101/synthesis", WARMUP_TIMEOUT_S)]

    def down(url, body, timeout):
        raise OSError("connection refused")
    assert TtsClient(post=down).warmup() is False


def test_timeout_asr_discard():
    clk = FakeClock()
    ok = AsrClient(mock_asr_send(clk, delay=8.0), clock=clk)
    assert ok.transcribe(1, "shm", 3.0)["text"]  # 3s 区間 -> 上限 8s、ちょうどは許容
    bad = AsrClient(mock_asr_send(clk, delay=8.01), clock=clk)
    with pytest.raises(AsrTimeout) as e:
        bad.transcribe(1, "shm", 3.0)
    assert e.value.code == 5003


@pytest.mark.parametrize("exc,want", [(TimeoutError("timed out"), AsrTimeout),
                                       (ConnectionResetError("reset"), AsrFailed)])
def test_asr_socket_errors_become_client_errors(exc, want):
    """urllib のソケット例外を素通しすると WS の受信処理ごと落ちる。発話破棄用の例外に変換する。"""
    def send(req, timeout):
        raise exc
    with pytest.raises(want):
        AsrClient(send).transcribe(1, "shm", 1.0)


def _ok_transcribe(uid, ref):
    return {"utterance_id": uid, "text": "t", "no_speech_prob": 0.1, "compression_ratio": 1.0}


def test_bad_request_asr():
    good = json.dumps({"utterance_id": 1, "pcm_ref": "shm1", "lang": "ja"}).encode()
    assert handle_request(good, _ok_transcribe)[0] == 200
    for bad in ({"utterance_id": 1, "pcm_ref": "s", "lang": "en"},
                {"utterance_id": -1, "pcm_ref": "s", "lang": "ja"},
                {"utterance_id": 1, "pcm_ref": "s", "lang": "ja", "x": 1}):
        st, body = handle_request(json.dumps(bad).encode(), _ok_transcribe)
        assert st == 400
        assert set(body["error"]) == {"code", "message"} and body["error"]["message"]
    assert handle_request(b"{", _ok_transcribe)[0] == 400


def test_bad_request_asr_over_http():
    srv = make_server(_ok_transcribe)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        req = urllib.request.Request(url, data=json.dumps(
            {"utterance_id": 1, "pcm_ref": "s", "lang": "en"}).encode())
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=5)
        assert e.value.code == 400
        assert "message" in json.loads(e.value.read())["error"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_hallu_threshold():
    assert is_hallucination(0.61, 1.0) is True
    assert is_hallucination(0.6, 1.0) is False
    assert is_hallucination(0.1, 2.41) is True
    assert is_hallucination(0.1, 2.4) is False


def test_hallu_client_discards():
    clk = FakeClock()

    def body(p):
        return {"utterance_id": 1, "text": "x", "no_speech_prob": p, "compression_ratio": 1.0}

    assert AsrClient(mock_asr_send(clk, body=body(0.61)), clock=clk).transcribe(1, "s", 1) is None
    assert AsrClient(mock_asr_send(clk, body=body(0.6)), clock=clk).transcribe(1, "s", 1) is not None
