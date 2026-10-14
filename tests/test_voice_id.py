"""話者識別 (設計書 6.4): 特徴量・声の登録と照合・名前の抜き出し・Director での名前の尋ね方。"""
import asyncio

import numpy as np
import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import MockBrowser

from idle_talk import ctl
from idle_talk.clients import AsrClient
from idle_talk.config import VoiceId, load_voice_id
from idle_talk.director import IdleTalkDirector
from idle_talk.main import DB_FILE, enable_voice_id
from idle_talk.store import Store
from idle_talk.vad import EnergyVad
from idle_talk.voice_id import (ASK_MAX, ASK_PROMPT, ASKED_PROMPT, ENROLL_N, NAME_PROMPT, VoiceBook, fbank,
                                parse_name)

LONG = 100  # 2s の発話 (MIN_VOICE_MS 以上)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "v.db", db_backup=False)
    yield s
    s.close()


def vec(*head):
    v = np.zeros(8, dtype=np.float32)
    v[:len(head)] = head
    return v


# ---- 特徴量 ----
def test_fbank_shape_and_mean_removed():
    t = np.arange(16000) / 16000
    pcm = (3000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16).tobytes()
    f = fbank(pcm)
    assert f.shape == (1 + (16000 - 400) // 160, 80) and f.dtype == np.float32
    assert np.allclose(f.mean(axis=0), 0, atol=1e-3)


def test_fbank_too_short_is_empty():
    assert fbank(b"\x00\x00" * 399).shape == (0, 80)


def test_fbank_peak_follows_pitch():
    t = np.arange(16000) / 16000

    def peak(hz):  # 前半だけ鳴らす: 平均を引いた後も、その周波数のメル帯域が最も大きく変化する
        x = 3000 * np.sin(2 * np.pi * hz * t) * (t < 0.5)
        x = x + np.random.default_rng(0).normal(0, 10, len(t))
        return int(np.argmax(fbank(x.astype(np.int16).tobytes()).std(axis=0)))

    assert peak(300) < peak(1000) < peak(4000)


# ---- 登録と照合 ----
def test_unknown_voice_enrolled_after_consistent_utterances(store):
    book = VoiceBook(store, threshold=0.5)
    a = vec(1, 0.1)
    assert all(book.identify(a, 1) is None for _ in range(ENROLL_N - 1))
    sid = book.identify(a, 2)
    assert sid is not None and book.name(sid) is None
    assert book.identify(vec(0.9, 0.2), 3) == sid  # 同じ人の少し違う声
    assert store.query("SELECT n FROM speakers WHERE id=?", (sid,))[0][0] == ENROLL_N + 1


def test_one_off_noise_is_not_enrolled(store):
    book = VoiceBook(store, threshold=0.5)
    assert book.identify(vec(1), 1) is None
    assert book.identify(vec(0, 1), 2) is None  # 1 回目と別の声: 数え直し
    assert store.count("speakers") == 0


def test_two_people_are_told_apart_and_persist(store, tmp_path):
    book = VoiceBook(store, threshold=0.5)
    for _ in range(ENROLL_N):
        a = book.identify(vec(1), 1)
    for _ in range(ENROLL_N):
        b = book.identify(vec(0, 1), 1)
    assert a is not None and b is not None and a != b
    book.set_name(a, "たろう", 2)
    again = VoiceBook(store, threshold=0.5)  # 再起動後も覚えている
    assert again.identify(vec(1, 0.1), 3) == a and again.name(a) == "たろう"
    assert again.identify(vec(0.1, 1), 3) == b


def test_same_name_merges_voices(store):
    book = VoiceBook(store, threshold=0.5)
    for _ in range(ENROLL_N):
        a = book.identify(vec(1), 1)
    book.set_name(a, "たろう", 1)
    for _ in range(ENROLL_N):
        b = book.identify(vec(0, 1), 1)  # マイクが変わって別人扱いになった
    assert book.set_name(b, "たろう", 2) == a
    assert store.count("speakers") == 1 and book.name(b) is None
    assert book.identify(vec(0.6, 1), 3) == a  # まとめた声のどちらにも近い


# ---- 名前の抜き出し ----
@pytest.mark.parametrize("text,name", [
    ("たろう", "たろう"), ("名前: 花子", "花子"), ("「ゆうき」です。", "ゆうき"), ("山田さん\n", "山田"),
    ("やまだ", "やまだ"), ("なし", None), ("名前はなし", None), ("。", None), ("", None),
    ("とても長い説明的な文章が返ってきた場合", None),
])
def test_parse_name(text, name):
    assert parse_name(text) == name


# ---- 設定 ----
def write(tmp_path, body):
    p = tmp_path / "s.toml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_voice_id(tmp_path):
    assert load_voice_id(tmp_path / "none.toml") == VoiceId()
    assert load_voice_id(write(tmp_path, "[voice_id]\non = true\nthreshold = 0.6\n")) == \
        VoiceId(True, VoiceId().model, 0.6)
    for bad in ("on = 1", "threshold = 1.0", "threshold = 0", 'model = ""', "x = 1"):
        with pytest.raises(ValueError):
            load_voice_id(write(tmp_path, f"[voice_id]\n{bad}\n"))


def test_missing_model_disables_voice_id(tmp_path, store):
    d = IdleTalkDirector(store, lambda m: iter(()), lambda t: b"", lambda: None, EnergyVad())
    cfg = write(tmp_path, f'[voice_id]\non = true\nmodel = "{(tmp_path / "no.onnx").as_posix()}"\n')
    assert enable_voice_id(d, cfg) is False and d.voices is None


# ---- Director: 名前を尋ね、返事から覚え、以後は名前で呼ぶ ----
def make(tmp_path, embed, name_reply="たろう"):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    systems: list[str] = []

    def llm(messages):
        system = messages[0]["content"]
        if system == NAME_PROMPT:
            yield name_reply
            return
        systems.append(system)
        yield "そうなんだ。"

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
    d.enable_voice_id(embed, 0.5)
    b = MockBrowser()
    b.attach(d.core)
    return d, b, store, systems


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


async def turn(d, b, systems, frames=LONG):
    n = len(systems)
    b.speak(loud_frames=frames)
    await until(lambda: len(systems) > n and d.fsm.ctx.state.value == "idle" and d._cur is None)
    if d._mem_task is not None:
        await d._mem_task
    return systems[-1]


def test_new_voice_is_asked_name_then_called_by_it(tmp_path):
    async def run():
        d, b, store, systems = make(tmp_path, lambda pcm: vec(1))
        try:
            assert ASK_PROMPT not in await turn(d, b, systems)  # 1 回目: まだ登録しない
            assert ASK_PROMPT in await turn(d, b, systems)  # 登録した声: 名前を尋ねる
            assert ASKED_PROMPT in await turn(d, b, systems)  # 返事: 名前で挨拶 (抜き出しは再生後)
            assert store.query("SELECT name FROM speakers")[0][0] == "たろう"
            assert "「たろう」さん" in await turn(d, b, systems)
        finally:
            store.close()
    asyncio.run(run())


def test_short_utterance_keeps_current_speaker(tmp_path):
    async def run():
        calls = []
        d, b, store, systems = make(tmp_path, lambda pcm: calls.append(len(pcm)) or vec(1))
        try:
            for _ in range(3):
                await turn(d, b, systems)
            sid = d._speaker
            assert sid is not None
            n = len(calls)
            await turn(d, b, systems, frames=15)  # 0.3s: 声の特徴を計算しない
            assert len(calls) == n and d._speaker == sid
        finally:
            store.close()
    asyncio.run(run())


def test_name_asked_at_most_ask_max_times(tmp_path):
    async def run():
        d, b, store, systems = make(tmp_path, lambda pcm: vec(1), name_reply="なし")
        try:
            asked = [ASK_PROMPT in await turn(d, b, systems) for _ in range(ASK_MAX * 2 + 3)]
            assert sum(asked) == ASK_MAX
            assert d._ask_sid is None and store.query("SELECT name FROM speakers")[0][0] is None
        finally:
            store.close()
    asyncio.run(run())


def test_embedding_failure_keeps_conversation(tmp_path):
    async def run():
        def boom(pcm):
            raise RuntimeError("onnx failed")
        d, b, store, systems = make(tmp_path, boom)
        try:
            await turn(d, b, systems)
            assert d._speaker is None and store.count("speakers") == 0
        finally:
            store.close()
    asyncio.run(run())


# ---- ctl speakers ----
def test_ctl_speakers_list_and_delete(tmp_path, capsys):
    s = Store(tmp_path / DB_FILE, db_backup=False)
    book = VoiceBook(s, 0.5)
    for _ in range(ENROLL_N):
        sid = book.identify(vec(1), 1)
    book.set_name(sid, "花子", 1)
    s.close()
    assert ctl.main(["speakers", "--data-dir", str(tmp_path)]) == 0
    assert "花子" in capsys.readouterr().out
    assert ctl.main(["speakers", "--data-dir", str(tmp_path), "--delete", str(sid)]) == 0
    assert ctl.main(["speakers", "--data-dir", str(tmp_path), "--delete", str(sid)]) == 1
