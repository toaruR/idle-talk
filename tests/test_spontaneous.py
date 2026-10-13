"""自発発話・無操作休止 (T020 / 設計書 4.3-5 / 6.1 / 8.1)。"""
import asyncio
import json
import threading

import pytest
from mocks import FakeClock
from mocks_browser import MockBrowser

from idle_talk import topics
from idle_talk.clients import AsrClient
from idle_talk.config import Config
from idle_talk.director import IdleTalkDirector
from idle_talk.fsm import Ev, Event
from idle_talk.spontaneous import Spontaneous, in_quiet_hours
from idle_talk.store import Store
from idle_talk.topic_worker import TopicWorker
from idle_talk.vad import EnergyVad


def sp(minute=12 * 60, **kw):
    s = Spontaneous(minutes=lambda: minute, **kw)
    s.apply_settings({"on": True})
    return s


def test_default_off_and_interval():
    s = Spontaneous()
    assert s.enabled is False
    assert Config().spontaneous_enabled is False
    s.apply_settings({"on": True})
    assert s.enabled is True and s.interval_s == 1200


def test_interval_out_of_range_rejected_5006():
    s = Spontaneous()
    with pytest.raises(ValueError):
        s.apply_settings({"interval_s": 9})
    assert s.interval_s == 1200


def test_interval_9_ui_returns_5006(tmp_path):
    d, b, _ = make(tmp_path)
    d.core.on_text(b, json.dumps({"t": "ui", "interval_s": 9}))
    assert [m["code"] for m in b.of("error")] == [5006]


def fire_count(s, minute_now):
    n = 0
    for i in range(1, 200):
        t = i * 600.0
        if s.should_fire(t, t - 1200.0):
            s.fire(t)
            s.done()
            s.on_user_speech()
            n += 1
    return n


@pytest.mark.parametrize("minute,expected_any", [(23 * 60 + 30, False), (3 * 60, False), (12 * 60, True)])
def test_quiet_hours(minute, expected_any):
    s = sp(minute)
    assert (fire_count(s, minute) > 0) is expected_any


def test_quiet_hours_boundaries():
    assert in_quiet_hours(22 * 60, "22:00", "07:00")
    assert not in_quiet_hours(7 * 60, "22:00", "07:00")
    assert not in_quiet_hours(21 * 60 + 59, "22:00", "07:00")


def test_fires_after_interval_of_silence():
    s = sp()
    s.apply_settings({"interval_s": 30})
    assert not s.should_fire(129.0, 100.0)
    assert s.should_fire(130.0, 100.0)
    assert not s.should_fire(130.0, None)  # IDLE でない


def test_silence_backoff_monotonic_no_consecutive():
    s = sp()
    s.apply_settings({"interval_s": 30})
    idle_since, fires = 0.0, []
    now = 0.0
    while len(fires) < 4:
        now += 1.0
        assert now < 100000
        if s.should_fire(now, idle_since):
            if fires:
                assert now - idle_since >= s.intervals[-1]  # 連続発火なし
            s.fire(now)
            s.done()  # 利用者は無反応。再生終了後に再び IDLE
            fires.append(now)
            idle_since = now
    assert s.intervals == [30.0, 60.0, 120.0, 240.0]
    s.on_user_speech()
    assert s.silent_count == 0 and s.wait_s == 30.0


def test_not_chat_mode_never_fires():
    s = sp(mode="focus")
    assert not any(s.should_fire(i * 1000.0, 0.0) for i in range(1, 50))
    assert s.fired == 0


def make(tmp_path, gate=None, texts=None, auto_playback=True):
    """texts を渡すと ASR を模擬し、順に認識結果として返す。"""
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)

    def llm(messages):
        if gate is not None:
            gate.wait(10)
        yield from ("やあ、元気ですか。",)

    def send(req, timeout):
        return 200, {"utterance_id": req["utterance_id"], "text": texts.pop(0),
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}

    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE", lambda: None, EnergyVad(),
                         None if texts is None else AsrClient(send, clock=clock), clock=clock, now_ms=lambda: 1000)
    d.spontaneous._minutes = lambda: 12 * 60
    b = MockBrowser(auto_playback=auto_playback)
    b.attach(d.core)
    d.clock_ = clock
    return d, b, store


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def test_idle_timeout(tmp_path):
    async def run():
        d, _b, _ = make(tmp_path)
        d.clock_.advance(1799)
        assert d.spontaneous_tick() is None and d.fsm.ctx.state.value == "idle"
        d.clock_.advance(1)  # 1800s
        assert d.spontaneous_tick() == "rest" and d.fsm.ctx.state.value == "rest"
    asyncio.run(run())


def test_idle_timeout_29m59s(tmp_path):
    async def run():
        d, _b, _ = make(tmp_path)
        d.clock_.advance(1799.0)
        d.spontaneous_tick()
        assert d.fsm.ctx.state.value == "idle"
    asyncio.run(run())


def test_spontaneous_plays_via_common_path(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True}))
        assert d.spontaneous_tick() is None  # 初回は 1 周期待つ
        d.clock_.advance(1200)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending)
        assert len(b.chunks) >= 1
        assert ("spontaneous", "play_utterance") in d.path_calls
        assert d.spontaneous.cancelled == 0
    asyncio.run(run())


def test_spontaneous_respects_lookahead_and_fires_again(tmp_path):
    """TTS が再生より速くても先読み 1 チャンクまでしか送らず (超過は dropped で捨てられる)、
    dropped が来ても完了扱いにして次の自発発話が止まらない。"""
    async def run():
        clock = FakeClock()
        store = Store(tmp_path / "t.db", db_backup=False)
        d = IdleTalkDirector(store, lambda m: iter(["一。", "二。", "三。", "四。"]), lambda text: b"RIFF....WAVE",
                             lambda: None, EnergyVad(), None, clock=clock, now_ms=lambda: 1000)
        d.spontaneous._minutes = lambda: 12 * 60
        b = MockBrowser(auto_playback=False)
        b.attach(d.core)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        clock.advance(30)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: d._gen_done)
        assert [c[2] for c in b.chunks] == [0, 1]  # 再生中 1 + 先読み 1
        ep, utt = b.chunks[0][:2]
        b.playback(ep, utt, 0, "finished")
        assert [c[2] for c in b.chunks] == [0, 1, 2]
        b.playback(ep, utt, 1, "dropped")
        b.playback(ep, utt, 2, "finished")
        b.playback(ep, utt, 3, "finished")
        assert not d.spontaneous.pending
        clock.advance(60)  # 無反応なので待ち時間は 2 倍
        assert d.spontaneous_tick() == "spontaneous"
        store.close()
    asyncio.run(run())


def test_cancel_prep(tmp_path):
    async def run():
        gate = threading.Event()
        d, b, _store = make(tmp_path, gate, texts=["ねえ、ちょっといい"])
        d.core.on_text(b, json.dumps({"t": "ui", "on": True}))
        d.spontaneous_tick()
        d.clock_.advance(1200)
        assert d.spontaneous_tick() == "spontaneous"
        await asyncio.sleep(0.05)
        assert d.spontaneous.pending
        ep = d.epoch.current
        b.speak()  # 準備中に利用者発話: 認識で語が出たら取消す
        await until(lambda: d.spontaneous.cancelled == 1)
        gate.set()
        await asyncio.sleep(0.2)
        assert not any(c[0] == ep for c in b.chunks)  # 自発発話は 1 チャンクも流れない
        assert ("spontaneous", "cancel") in d.path_calls
        assert not d.spontaneous.pending
    asyncio.run(run())


def test_noise_during_spontaneous_playback_resumes(tmp_path):
    """自発発話の再生中に拾った音が語にならなければ (雑音)、取消さず duck を戻して最後まで再生する。"""
    async def run():
        d, b, _store = make(tmp_path, texts=["…"], auto_playback=False)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True}))
        d.clock_.advance(1200)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: d._gen_done and b.chunks)
        b.speak()  # 物音
        assert len(b.of("duck")) == 1 and d.spontaneous.pending
        await until(lambda: d.ingress.asr_pending == 0 and b.of("resume"))
        assert d.fsm.ctx.state.value == "idle" and d.spontaneous.cancelled == 0
        for ep, utt, cid, _pcm in list(b.chunks):
            b.playback(ep, utt, cid, "finished")
        assert not d.spontaneous.pending and d.spontaneous.cancelled == 0
    asyncio.run(run())


def test_off_by_default_no_fire(tmp_path):
    async def run():
        d, _b, _ = make(tmp_path)
        d.clock_.advance(1000)
        assert d.spontaneous_tick() is None
        d.clock_.advance(1000)
        assert d.spontaneous.fired == 0
    asyncio.run(run())


def test_silence_measured_from_end_of_speech_with_backoff(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        d.clock_.advance(29)
        assert d.spontaneous_tick() is None
        d.clock_.advance(1)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending and d.fsm.ctx.state.value == "idle")
        d.clock_.advance(59)  # 発話終了から計測。無反応なので 30s x 2
        assert d.spontaneous_tick() is None
        d.clock_.advance(1)
        assert d.spontaneous_tick() == "spontaneous"
        assert d.spontaneous.intervals == [30.0, 60.0]
    asyncio.run(run())


def test_spontaneous_loop_ticks():
    from idle_talk.main import spontaneous_loop
    ticks = []

    class D:
        def spontaneous_tick(self):
            ticks.append(1)

    async def sleep(_s):
        if len(ticks) == 3:
            raise asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(spontaneous_loop(D(), sleep=sleep))
    assert len(ticks) == 3


def test_silence_counted_from_connect(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.clock_.advance(100)  # 接続前の経過は数えない
        b.attach(d.core)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        assert d.spontaneous_tick() is None
        d.clock_.advance(30)
        assert d.spontaneous_tick() == "spontaneous"
    asyncio.run(run())


def test_no_fire_without_client_and_reconnect_clears_stuck(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "on": True, "interval_s": 30}))
        d.core.hello_done = False  # 未接続相当
        d.clock_.advance(100)
        assert d.spontaneous_tick() is None
        d.core.hello_done = True
        d.spontaneous.fire(d.clock_())  # 前の接続で再生されずに残った自発発話
        d.spontaneous.awaiting_reaction, d.spontaneous.silent_count = True, 2
        b.attach(d.core)  # 再接続 (hello)
        assert not d.spontaneous.pending and d.spontaneous.wait_s == 30.0
        d.clock_.advance(30)
        assert d.spontaneous_tick() == "spontaneous"
    asyncio.run(run())


def test_radio_fixed_gap_without_backoff_even_if_off():
    from idle_talk.spontaneous import RADIO_GAP_S
    s = Spontaneous(minutes=lambda: 12 * 60)
    s.apply_settings({"radio": True})  # on はオフのまま
    assert s.should_fire(100.0 + RADIO_GAP_S, 100.0)
    for _ in range(3):
        s.fire(0.0)
        s.done()
    assert s.silent_count == 0 and s.wait_s == RADIO_GAP_S
    s.apply_settings({"radio": False})
    assert not s.should_fire(10000.0, 0.0)


def test_radio_no_question_prompt_and_no_idle_timeout(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        seen = []
        d.pipeline._llm = lambda m: seen.append(m) or iter(["今日はいい天気です。"])
        d.core.on_text(b, json.dumps({"t": "ui", "radio": True}))
        d.clock_.advance(3)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending)
        assert "返事を求めない" in seen[0][-1]["content"]
        d.clock_.advance(1800)  # 無操作でも休止せず話し続ける
        assert d.spontaneous_tick() == "spontaneous"
        assert d.fsm.ctx.state.value == "idle"
    asyncio.run(run())


def test_radio_does_not_wait_for_memory_extraction(tmp_path):
    async def run():
        d, b, _store = make(tmp_path)
        d.core.on_text(b, json.dumps({"t": "ui", "radio": True}))
        d._mem_task = asyncio.get_running_loop().create_task(asyncio.sleep(10))  # 記憶抽出中
        d.clock_.advance(3)
        assert d.spontaneous_tick() == "spontaneous"
        d._mem_task.cancel()
    asyncio.run(run())


def test_spontaneous_uses_topic_card_once(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        cid = topics.add_card(store, "新しい駅が開業", facts="<p>来春に開業予定。</p>", expires_at=10_000)
        seen = []
        d.pipeline._llm = lambda m: seen.append(m) or iter(["そういえば駅の話です。"])
        d.core.on_text(b, json.dumps({"t": "ui", "on": True}))
        d.clock_.advance(1200)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending)
        prompt = seen[0][-1]["content"]
        assert "「新しい駅が開業」" in prompt and "来春に開業予定。" in prompt and "<p>" not in prompt
        assert store.query("SELECT used_at FROM topic_cards WHERE id=?", (cid,))[0][0] == 1000
        assert topics.select_cards(store, 1000) == []  # 使ったカードは次回選ばれない (SPONT_TOPICS に戻る)
    asyncio.run(run())


def test_topic_card_trims_history_and_forbids_repeating_scenery(tmp_path):
    async def run():
        d, b, store = make(tmp_path)
        for i in range(6):
            store.add_turn(i, "ai", f"雨の音が静かに聞こえる夜です {i}", 1, None, i)
        topics.add_card(store, "新しい駅が開業", expires_at=10_000)
        seen = []
        d.pipeline._llm = lambda m: seen.append(m) or iter(["駅の話です。"])
        d.core.on_text(b, json.dumps({"t": "ui", "radio": True}))
        d.clock_.advance(3)
        assert d.spontaneous_tick() == "spontaneous"
        await until(lambda: not d.spontaneous.pending)
        msgs = seen[0]
        assert [m["content"][-1] for m in msgs[1:-1]] == ["4", "5"]  # 直近 CARD_TURNS 件だけ
        assert "繰り返さず" in msgs[-1]["content"] and "返事を求めない" in msgs[-1]["content"]
        assert "直前の話の続き" not in msgs[-1]["content"]
        seen.clear()
        d.clock_.advance(3)
        assert d.spontaneous_tick() == "spontaneous"  # カードが無ければ従来どおり全履歴
        await until(lambda: not d.spontaneous.pending)
        assert len(seen[0]) - 2 > 2
    asyncio.run(run())


def test_topic_worker_follows_state(tmp_path):
    d, _b, store = make(tmp_path)
    d.topic_worker = TopicWorker(store, ["http://feed.test/rss"], fetch=lambda u: b"")
    d.topic_worker.start()
    d.fsm.enqueue(Event(Ev.IDLE_TIMEOUT))
    d.fsm.pump()
    assert d.fsm.ctx.state.value == "rest" and d.topic_worker.running is False
