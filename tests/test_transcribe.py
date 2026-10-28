"""文字起こしモード (設計書 6.10): AI は話さず、認識結果を声の特徴から付けた仮名つきで残す。"""
import asyncio
import json
import logging
import re

import numpy as np
import pytest
from mocks import FakeClock, mock_asr_send
from mocks_browser import LOUD, MockBrowser

from idle_talk import ctl
from idle_talk.clients import AsrClient
from idle_talk.config import Transcribe, load_transcribe
from idle_talk.director import TRANSCRIBE_SPLIT_MS, UNKNOWN_SPEAKER, IdleTalkDirector, merge_final
from idle_talk.history import history_window
from idle_talk.main import DB_FILE
from idle_talk.protocol import validate_message
from idle_talk.speaker_track import ChangeDetector, SpeakerTrack
from idle_talk.store import Store
from idle_talk.vad import EnergyVad
from idle_talk.voice_id import ENROLL_N, VoiceBook, pitch_track, voice_word

LONG = 100  # 2s の発話 (MIN_VOICE_MS 以上)


def tone(f0: float, vibrato: float = 0.02, seconds: float = 2.0, rate: float = 1.5) -> bytes:
    """基本周波数 f0 の倍音つきの声らしい音。vibrato は F0 の揺れ幅 (割合)。"""
    t = np.arange(int(16000 * seconds)) / 16000
    phase = 2 * np.pi * np.cumsum(f0 * (1 + vibrato * np.sin(2 * np.pi * rate * t))) / 16000
    x = sum(np.sin(k * phase) / k for k in range(1, 8)) * 5000
    return x.astype(np.int16).tobytes()


def vec(*head):
    v = np.zeros(8, dtype=np.float32)
    v[:len(head)] = head
    return v


# ---- 仮名の言葉 ----
@pytest.mark.parametrize("f0", [100, 150, 190, 260])
def test_pitch_track_follows_f0(f0):
    assert abs(float(np.median(pitch_track(tone(f0)))) - f0) < f0 * 0.03


@pytest.mark.parametrize("f0,vibrato,word", [
    (100, 0.02, "低く落ち着いた声"),
    (150, 0.02, "やや低く落ち着いた声"),
    (190, 0.02, "やや高く落ち着いた声"),
    (260, 0.02, "高く落ち着いた声"),
    (260, 0.4, "高く抑揚のある声"),
    (150, 0.25, "やや低い声"),
])
def test_voice_word(f0, vibrato, word):
    assert voice_word(tone(f0, vibrato)) == word


def test_voice_word_without_voice():
    assert voice_word(b"\x00\x00" * 32000) == "声"
    assert voice_word(b"") == "声"


def enroll(book, v):
    for _ in range(ENROLL_N):
        sid = book.identify(v, 1)
    return sid


def test_alias_numbers_are_serial_and_persist(tmp_path):
    s = Store(tmp_path / "v.db", db_backup=False)
    book = VoiceBook(s, 0.5)
    a = enroll(book, vec(1))
    b = enroll(book, vec(0, 1))
    assert a is not None and b is not None and a != b
    assert book.set_alias(a, "低い声", 1) == "低い声1"
    assert book.set_alias(b, "高く抑揚のある声", 1) == "高く抑揚のある声2"  # 言葉が違っても番号は通し
    s.close()
    s = Store(tmp_path / "v.db", db_backup=False)
    book = VoiceBook(s, 0.5)
    assert book.alias(a) == "低い声1"
    c = enroll(book, vec(0, 0, 1))
    assert book.set_alias(c, "声", 1) == "声3"
    s.close()


def test_ctl_speakers_shows_alias(tmp_path, capsys):
    s = Store(tmp_path / DB_FILE, db_backup=False)
    book = VoiceBook(s, 0.5)
    book.set_alias(enroll(book, vec(1)), "低い声", 1)
    s.close()
    assert ctl.main(["speakers", "--data-dir", str(tmp_path)]) == 0
    assert "仮名: 低い声1" in capsys.readouterr().out


def test_ui_transcribe_schema():
    assert validate_message({"t": "ui", "transcribe": True})
    assert not validate_message({"t": "ui", "transcribe": 1})


# ---- Director ----
def make(tmp_path, embed=None):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    llm_calls: list[list[dict]] = []

    def llm(messages):
        llm_calls.append(messages)
        yield "そうなんだ。"

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         AsrClient(mock_asr_send(clock)), clock=clock, now_ms=lambda: 1000)
    if embed is not None:
        d.enable_voice_id(embed, 0.5)
    b = MockBrowser()
    b.attach(d.core)
    return d, b, store, llm_calls


def ui(b, **settings):
    b.core.on_text(b, json.dumps({"t": "ui", **settings}))


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


async def utter(d, b, frames=LONG):
    n = len(b.of("transcript"))
    b.speak(loud_frames=frames)
    await until(lambda: len(b.of("transcript")) > n and d.fsm.ctx.state.value == "idle")
    return b.of("transcript")[-1]


def test_transcribe_does_not_respond_and_labels_speakers(tmp_path):
    async def run():
        voice = {"emb": vec(1)}
        d, b, store, llm_calls = make(tmp_path, lambda pcm: voice["emb"])
        try:
            ui(b, transcribe=True)
            assert d.transcribe_on and not d.ingress.silence_check  # 生のマイクの静かな部屋を E6 と取らない
            first = await utter(d, b)
            assert first["text"]  # 2 秒の発話の窓 (0.5 秒ずつ) で登録が進み、早ければ最初の発話から仮名が付く
            assert first["speaker"] in (UNKNOWN_SPEAKER, "声1")
            assert (await utter(d, b))["speaker"] == "声1"  # 仮名を付ける (無音の模擬音は高さが出ない)
            voice["emb"] = vec(0, 1)
            labels = [(await utter(d, b))["speaker"] for _ in range(2)]
            assert labels[-1] == "声2" and "声1" not in labels  # 知らない声: 前の人にしない
            assert (await utter(d, b, frames=15))["speaker"] == "声2"  # 短い発話は直前の話者を引き継ぐ
            await asyncio.sleep(0.1)
            assert llm_calls == [] and b.chunks == [] and b.of("text") == []  # AI は話さない
            rows = store.query("SELECT role, text FROM turns")
            assert {r["role"] for r in rows} == {"transcript"} and rows[1]["text"].startswith("声1: ")
            assert history_window(store) == []  # 会話の履歴には入れない
            assert d.spontaneous_tick(d.clock() + 10 ** 6) is None  # 自発発話も無操作休止もしない
            assert d.fsm.ctx.state.value == "idle"
        finally:
            store.close()
    asyncio.run(run())


def test_speaker_changes_and_text_go_to_talk_log_separately(tmp_path, caplog):
    """話者の切り替わりは 0.5 秒ずつずらした窓で測って SPEAKER 行、文字は区間の時刻つきの TEXT 行で別々に出す。"""
    async def run():
        voice = {"emb": vec(1)}
        d, b, store, _ = make(tmp_path, lambda pcm: voice["emb"])
        d.wall = lambda: 1_800_000_000.0
        try:
            ui(b, transcribe=True)
            assert d.track is not None
            d.track.close()  # 模擬ブラウザは音を一気に送り、別スレッドの計算が追いつかず窓が飛ぶ: その場で計算する
            d.track = SpeakerTrack(d.ingress.embed, d._on_window, submit=lambda work: work())
            for emb in (vec(1), vec(0, 1)):  # 窓は照合だけ: 区間の声で登録 (2 回) された後の 3 回目で替わる
                voice["emb"] = emb
                for _ in range(3):
                    await utter(d, b, frames=150)  # 3 秒: 窓 (1.5 秒、0.5 秒ずつ) が 4 つ
            await asyncio.sleep(0.2)
        finally:
            store.close()
    with caplog.at_level(logging.INFO, logger="idle_talk.talk"):
        asyncio.run(run())
    lines = [r.getMessage() for r in caplog.records if r.name == "idle_talk.talk"]
    speaker = [m for m in lines if m.startswith("SPEAKER(")]
    assert len(speaker) == 2 and speaker[0].endswith("- → 声1") and speaker[1].endswith("声1 → 声2")
    text = [m for m in lines if m.startswith("TEXT(")]
    assert len(text) == 6 and all(re.match(r"TEXT\(\d\d:\d\d:\d\d\.\d-\d\d:\d\d:\d\d\.\d\): ", m) for m in text)
    assert not any(m.startswith("TRANSCRIPT(") for m in lines)


def test_change_detector_needs_two_windows():
    c = ChangeDetector()
    assert c.push(1, 10) is None and c.push(1, 11) == (None, 1, 10)
    assert c.push(2, 12) is None and c.push(1, 13) is None  # 1 窓だけの揺れでは替わらない
    assert c.push(None, 14) is None  # 知らない声は候補を崩さない
    assert c.push(2, 15) is None and c.push(None, 16) is None and c.push(2, 17) == (1, 2, 15)


def test_transcribe_threshold_and_voice_length(tmp_path):
    """文字起こしは [transcribe].threshold で照合し (会話は 0.6 のまま)、声の特徴は voice_max_s までの頭で測る。"""
    async def run():
        voice = {"emb": vec(1)}
        lengths = []
        d, b, store, _ = make(tmp_path, lambda pcm: lengths.append(len(pcm)) or voice["emb"])
        d.transcribe = Transcribe(threshold=0.4, voice_max_s=1)
        try:
            ui(b, transcribe=True)
            d.track.close()  # 話者の切り替わりの窓 (1.5 秒) は測らない: 区間の声の長さだけを見る
            d.track = None
            await utter(d, b)
            assert (await utter(d, b))["speaker"] == "声1"
            assert max(lengths) <= 1000 * 32  # 1 秒 (16kHz s16le)
            voice["emb"] = vec(1, 1.5)  # 声1 との類似度 0.55: 会話 (0.6) なら別人、文字起こし (0.4) なら同じ人
            assert (await utter(d, b))["speaker"] == "声1"
            ui(b, transcribe=False)
            assert d.ingress.voice_max_ms == 0
        finally:
            store.close()
    asyncio.run(run())


def test_load_transcribe(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text("[transcribe]\nthreshold = 0.5\nvoice_max_s = 6\n", encoding="utf-8")
    assert load_transcribe(f) == Transcribe(0.5, 6)
    f.write_text("[transcribe]\nvad_ratio = 1.5\n", encoding="utf-8")
    assert load_transcribe(f).vad_ratio == 1.5
    assert load_transcribe(tmp_path / "none.toml") == Transcribe()
    for bad in ("threshold = 1.5", "threshold = 0", "voice_max_s = 1", "voice_max_s = 2.5", "vad_ratio = 1",
                "vad_ratio = 11", "x = 1"):
        f.write_text(f"[transcribe]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_transcribe(f)


def test_transcribe_splits_endless_sound(tmp_path):
    """テレビ・ドラマの音は途切れず発話が終わらない (実機 2026-10-27)。TRANSCRIBE_SPLIT_MS で切って確定し、
    溜めている間も PARTIAL_MS ごとに仮の文字を出す。"""
    async def run():
        d, b, store, _ = make(tmp_path)
        try:
            ui(b, transcribe=True)
            assert d.ingress.max_seg_ms == TRANSCRIBE_SPLIT_MS
            for _ in range(TRANSCRIBE_SPLIT_MS // 20 * 2 + 10):  # 鳴り続ける (無音にならない)
                b.send_frame(LOUD)
                await asyncio.sleep(0)
            await until(lambda: len(b.of("transcript")) >= 2)
            assert d.ingress._buf is not None and d.fsm.ctx.state.value == "user_speaking"
            partial = b.of("partial")
            assert len(partial) >= 3 and partial[0]["text"]  # 確定を待たずに出る (前の認識が終わるまでは飛ばす)
            assert {m["seg"] for m in partial} >= {m["uid"] for m in b.of("transcript")}  # 確定と同じ番号
            ui(b, transcribe=False)
            assert d.ingress.max_seg_ms == 0
            n = len(b.of("partial"))
            for _ in range(200):
                b.send_frame(LOUD)
            assert len(b.of("partial")) == n  # 文字起こしでなければ仮の文字は出さない
        finally:
            store.close()
    asyncio.run(run())


def test_transcribe_uses_name_when_known(tmp_path):
    async def run():
        d, b, store, _ = make(tmp_path, lambda pcm: vec(1))
        try:
            ui(b, transcribe=True)
            await utter(d, b)
            await utter(d, b)
            d.voices.set_name(d._speaker, "花子", 1)
            assert (await utter(d, b))["speaker"] == "花子"
        finally:
            store.close()
    asyncio.run(run())


def test_transcribe_without_voice_id_is_unknown(tmp_path):
    async def run():
        d, b, store, llm_calls = make(tmp_path)
        try:
            ui(b, transcribe=True)
            assert (await utter(d, b))["speaker"] == UNKNOWN_SPEAKER
            assert llm_calls == []
        finally:
            store.close()
    asyncio.run(run())


def test_transcribe_off_returns_to_conversation(tmp_path):
    async def run():
        d, b, store, llm_calls = make(tmp_path)
        try:
            ui(b, transcribe=True)
            await utter(d, b)
            ui(b, transcribe=False)
            assert not d.transcribe_on and d.ingress.silence_check
            b.speak(loud_frames=LONG)
            await until(lambda: llm_calls and d.fsm.ctx.state.value == "idle" and d._cur is None)
            assert [m["role"] for m in llm_calls[-1][1:]] == ["user"]  # 文字起こしの発言は履歴に混ざらない
        finally:
            store.close()
    asyncio.run(run())


def test_transcribe_keeps_short_text_and_lowers_vad_ratio(tmp_path):
    """話者が替わる所の「私は」「もう」等の 2 字を捨てず、閾値の倍率を [transcribe].vad_ratio に下げる (実機 2026-10-28)。"""
    async def run():
        d, b, store, _ = make(tmp_path)
        d.short_reply_s = 0  # 会話では 2 字以下を常に捨てる設定
        try:
            assert d._out_of_context(False, "もう")
            ratio = d.ingress.vad.ratio
            ui(b, transcribe=True)
            assert not d._out_of_context(False, "もう") and d.ingress.vad.ratio == d.transcribe.vad_ratio
            ui(b, transcribe=False)
            assert d._out_of_context(False, "もう") and d.ingress.vad.ratio == ratio
        finally:
            store.close()
    asyncio.run(run())


def test_final_shorter_than_partial_keeps_partial(tmp_path, caplog):
    """確定の認識が崩れて最後の仮の文字より大きく短ければ、仮の文字を確定にする (実機 2026-10-28)。"""
    async def run():
        d, b, store, _ = make(tmp_path)
        try:
            ui(b, transcribe=True)
            d._partials[5] = "ゆうべ何枚か撮らせてもらった俺と話し"
            d._transcribe(5, "最高っすね")
            d._partials[6] = "こんにちは"
            d._transcribe(6, "こんにちは世界")
            d._partials[7] = "十八五十二年に初めて"
            d._transcribe(7, "千八五十二年に")  # 少しでも短ければ (9 割未満) 仮の文字
            assert [m["text"] for m in b.of("transcript")] == ["ゆうべ何枚か撮らせてもらった俺と話し 最高っすね", "こんにちは世界",
                                                              "十八五十二年に初めて"]
            assert d._partials == {}
            d._partials[8] = "ははは笑っちゃう"
            assert d.ingress.on_discard(8) and b.of("transcript")[-1]["text"] == "ははは笑っちゃう"  # 確定が捨てられた
            assert not d.ingress.on_discard(9)  # 仮の文字が無ければ補わない
        finally:
            store.close()
    with caplog.at_level(logging.INFO, logger="idle_talk.director"):
        asyncio.run(run())
    assert any("final shorter than partial" in r.getMessage() for r in caplog.records)


def test_hallucinated_final_falls_back_to_partial(tmp_path):
    """確定の認識が幻聴として捨てられても、灰色で出ていた仮の文字を確定にし、次の区間の頭には足さない。"""
    async def run():
        d, b, store, _ = make(tmp_path)
        try:
            ui(b, transcribe=True)
            d._partials[d.ingress._uid + 1] = "あはははは"
            send = d.ingress.asr._send
            d.ingress.asr._send = lambda req, t: (200, {"utterance_id": req["utterance_id"], "text": "",
                                                        "no_speech_prob": 1.0, "compression_ratio": 0.0})
            b.speak(loud_frames=LONG)
            await until(lambda: b.of("transcript"))
            d.ingress.asr._send = send
            assert b.of("transcript")[-1]["text"] == "あはははは" and d.ingress._head is None
        finally:
            store.close()
    asyncio.run(run())


@pytest.mark.parametrize("partial,final,want", [
    # 実機 2026-10-28 の idle_talk.log から
    ("じゃ一瞬しか見れないってことですか", "そうよ", "じゃ一瞬しか見れないってことですか そうよ"),
    ("一年に一回しか咲かないからね", "一回しか", "一年に一回しか咲かないからね"),
    ("夜満開になって最終的に朝大体六時ぐらい周辺からポタポタポタポタ今みたいな音が",
     "最終的に朝大体六時ぐらい周辺からポタポタポタポタ",
     "夜満開になって最終的に朝大体六時ぐらい周辺からポタポタポタポタ今みたいな音が"),
    ("一夜のみ夕方咲き始めて", "一夜のみ夕方", "一夜のみ夕方咲き始めて"),
    ("千八百五十二年", "千八百五十二年に", "千八百五十二年に"),  # 確定の方が長い: 確定
    ("", "こんにちは", "こんにちは"),
    ("こんにちは", "", "こんにちは"),
])
def test_merge_final(partial, final, want):
    assert merge_final(partial, final) == want

