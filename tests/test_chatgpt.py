"""呼びかけで ChatGPT の音声会話へ引き渡す (設計書 6.11)。Chrome・CDP は模擬する。"""
import asyncio
import json

import pytest
from mocks import FakeClock
from mocks_browser import MockBrowser

from idle_talk import chatgpt_voice as cv
from idle_talk.chatgpt_voice import ChatgptVoice, IdleWatch, chrome_args, wake_match
from idle_talk.clients import AsrClient
from idle_talk.config import Chatgpt, load_chatgpt
from idle_talk.director import HANDOFF_KIND, HANDOFF_TAIL_S, IdleTalkDirector
from idle_talk.fsm import State
from idle_talk.store import Store
from idle_talk.vad import EnergyVad

PHRASES = ("チャッピー",)


# ---- 呼びかけの一致 ----
@pytest.mark.parametrize("text", ["チャッピー", "ちゃっぴー", "チャッピー。", "ﾁｬｯﾋﾟｰ", "チャッピーさん", "ねえチャッピー",
                                  "チャッピー！"])
def test_wake_match(text):
    assert wake_match(text, PHRASES)


@pytest.mark.parametrize("text", ["はチャッピーで大丈夫だよ", "チャッピーさんがいます", "迷っちゃって", "ちゃった",
                                  "笑っちゃった", "チャピ", ""])
def test_wake_no_match(text):
    assert not wake_match(text, PHRASES)


def test_wake_other_phrase():
    assert wake_match("ねえパソコン", ("ねえパソコン",)) and not wake_match("パソコン", ("ねえパソコン",))


# ---- 設定 ----
def test_load_chatgpt(tmp_path):
    f = tmp_path / "t.toml"
    f.write_text('[chatgpt]\non = true\nphrases = ["チャッピー", "ちゃっぴ"]\nport = 9444\n', encoding="utf-8")
    c = load_chatgpt(f)
    assert c.on and c.phrases == ("チャッピー", "ちゃっぴ") and c.port == 9444 and c.url == "https://chatgpt.com/"
    assert load_chatgpt(tmp_path / "none.toml") == Chatgpt()
    for bad in ("on = 1", "phrases = []", 'phrases = ["x"]', "port = 80", 'url = "http://chatgpt.com/"',
                "idle_timeout_s = 1", "x = 1"):
        f.write_text(f"[chatgpt]\n{bad}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_chatgpt(f)


def test_chrome_args_use_own_profile_and_local_port(tmp_path):
    a = chrome_args("chrome.exe", tmp_path / "p", 9333, "https://chatgpt.com/")
    assert f"--user-data-dir={tmp_path / 'p'}" in a and "--remote-debugging-port=9333" in a
    assert "--app=https://chatgpt.com/" in a and "--disable-backgrounding-occluded-windows" in a


# ---- 無操作の締め切り ----
def test_idle_watch():
    w = IdleWatch(0.0, 20, 30)
    assert not w.expired(19.9) and w.expired(20)
    w.activity(10, voice=True)  # 声を聞いた後は idle_s
    assert not w.expired(39.9) and w.expired(40)
    w.activity(11)  # 締め切りは縮めない
    assert w.deadline == 41


# ---- ChatgptVoice (ページは模擬) ----
class FakePage:
    def __init__(self, snaps):
        self.snaps = list(snaps)  # 順に返す。最後の 1 つは繰り返す
        self.clicks: list[tuple[float, float]] = []
        self.navigated: list[str] = []

    async def probe(self):
        snap = self.snaps[0] if len(self.snaps) == 1 else self.snaps.pop(0)
        if callable(snap):
            snap = snap(self)
        return dict(snap)

    async def click(self, x, y):
        self.clicks.append((x, y))

    async def navigate(self, url):
        self.navigated.append(url)

    async def front(self):
        pass


class FakeHost:
    def __init__(self, page=None, fail=False):
        self._p, self.fail, self.forgot = page, fail, 0

    async def page(self):
        if self.fail:
            raise cv.CdpError("no chrome")
        return self._p

    def forget(self):
        self.forgot += 1


def voice(page=None, fail=False, **cfg):
    clock = FakeClock()

    async def sleep(s):
        clock.advance(s)
        await asyncio.sleep(0)
    v = ChatgptVoice(Chatgpt(on=True, **cfg), FakeHost(page, fail), clock=clock, sleep=sleep)
    return v, clock


READY = {"kind": "Ready", "start": {"x": 10, "y": 20}, "busy": False, "revision": "0:0"}
ACTIVE = {"kind": "Active", "end": {"x": 30, "y": 40}, "busy": False, "revision": "0:0"}


def test_start_clicks_voice_button():
    page = FakePage([READY, READY, ACTIVE])
    v, _ = voice(page)
    assert asyncio.run(v.start()) == cv.STARTED and page.clicks == [(10, 20)]


@pytest.mark.parametrize("snap,want", [({"kind": "Login"}, cv.LOGIN), ({"kind": "NeedsAction"}, cv.NEEDS_ACTION),
                                       ({"kind": "Unsupported"}, cv.UNSUPPORTED), (ACTIVE, cv.ALREADY),
                                       ({"kind": "Loading"}, cv.TIMEOUT)])
def test_start_results(snap, want):
    page = FakePage([snap])
    v, _ = voice(page)
    assert asyncio.run(v.start()) == want and page.clicks == []


def test_start_without_browser():
    v, _ = voice(fail=True)
    assert asyncio.run(v.start()) == cv.NO_BROWSER and v.host.forgot == 1


def test_session_ends_idle_and_presses_end():
    def snap(p):  # 終了ボタンを押すと開始ボタンに戻る
        return READY if p.clicks else ACTIVE
    page = FakePage([snap])
    v, clock = voice(page, first_timeout_s=20)
    assert asyncio.run(v.session()) == cv.END_IDLE
    assert 20 <= clock.t < 30 and page.clicks == [(30, 40)] and page.navigated == []


def test_session_voice_extends_deadline():
    page = FakePage([lambda p: READY if p.clicks else ACTIVE])
    v, clock = voice(page, first_timeout_s=20, idle_timeout_s=30)

    async def run():
        task = asyncio.ensure_future(v.session())
        while clock.t < 15:
            await asyncio.sleep(0)
        for _ in range(cv.VOICED_FRAMES):  # 15 秒目に声: 締め切りは 45 秒
            v.hear(0.5, 0.1)
        return await task
    assert asyncio.run(run()) == cv.END_IDLE and 45 <= clock.t < 50


def test_session_short_noise_does_not_extend():
    page = FakePage([lambda p: READY if p.clicks else ACTIVE])
    v, clock = voice(page, first_timeout_s=20)

    async def run():
        task = asyncio.ensure_future(v.session())
        while clock.t < 15:
            await asyncio.sleep(0)
        for _ in range(cv.VOICED_FRAMES - 1):
            v.hear(0.5, 0.1)
        v.hear(0.0, 0.1)
        return await task
    assert asyncio.run(run()) == cv.END_IDLE and clock.t < 25


def test_session_assistant_update_extends():
    revs = iter(range(1000))

    def snap(p):
        if p.clicks:
            return READY
        r = next(revs)
        return dict(ACTIVE, revision=f"1:{r if r < 60 else 60}")  # 30 秒 (0.5 秒 x 60) 応答が更新される
    page = FakePage([snap])
    v, clock = voice(page, first_timeout_s=20)
    assert asyncio.run(v.session()) == cv.END_IDLE and clock.t >= 50


def test_session_closed_by_user():
    page = FakePage([ACTIVE] * 4 + [READY])
    v, _ = voice(page)
    assert asyncio.run(v.session()) == cv.END_CLOSED and page.clicks == []


def test_session_max():
    page = FakePage([lambda p: READY if p.clicks else dict(ACTIVE, busy=True)])
    v, clock = voice(page, max_s=60)
    assert asyncio.run(v.session()) == cv.END_MAX and 60 <= clock.t < 62


def test_stop_reloads_when_end_button_missing():
    def snap(p):
        return READY if p.navigated else {"kind": "Active", "end": None}
    page = FakePage([snap])
    v, _ = voice(page)
    assert asyncio.run(v.stop()) and page.navigated == ["https://chatgpt.com/"]


# ---- Director ----
class FakeGpt:
    """ChatgptVoice の代わり。session は end が立つまで待つ。"""

    def __init__(self, result=cv.STARTED):
        self.result, self.started, self.heard = result, 0, 0
        self.end = asyncio.Event()
        self.reason = cv.END_IDLE

    def matches(self, text):
        return wake_match(text, PHRASES)

    def hear(self, level, threshold):
        self.heard += 1

    async def start(self):
        self.started += 1
        return self.result

    async def session(self):
        await self.end.wait()
        return self.reason


def make(tmp_path, texts, replies=None, result=cv.STARTED):
    clock = FakeClock()
    store = Store(tmp_path / "t.db", db_backup=False)
    seen: list[list[dict]] = []
    replies = list(replies or [])

    def llm(messages):
        seen.append(messages)
        yield replies.pop(0) if replies else "うん。"

    def send(req, timeout):
        return 200, {"utterance_id": req["utterance_id"], "text": texts.pop(0),
                     "no_speech_prob": 0.1, "compression_ratio": 1.0}
    d = IdleTalkDirector(store, llm, lambda text: b"RIFF....WAVE" + b"\x10\x00" * 10, lambda: None, EnergyVad(),
                         AsrClient(send, clock=clock), clock=clock, now_ms=lambda: 1000)
    d.chatgpt = FakeGpt(result)
    b = MockBrowser()
    b.attach(d.core)
    d.clock_, d.seen_ = clock, seen
    return d, b, store


async def until(pred, timeout=5.0):
    end = asyncio.get_running_loop().time() + timeout
    while not pred():
        assert asyncio.get_running_loop().time() < end, "timeout"
        await asyncio.sleep(0.01)


def handoff_events(store):
    return [json.loads(r[0]) for r in store.query("SELECT payload_json FROM events WHERE kind=?", (HANDOFF_KIND,))]


def test_wake_hands_off_and_stops_listening(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["チャッピー"])
        b.speak()
        await until(lambda: d.handoff_on)
        assert d.fsm.ctx.state is State.REST and not d.fsm.ctx.muted
        assert d.seen_ == [] and b.of("handoff")[-1]["active"]  # LLM は呼ばない
        await until(lambda: d.chatgpt.started == 1)
        n = d.ingress._n
        b.speak()  # 会話中の音は VAD にも ASR にも入れず、音量だけ見る
        assert d.ingress._n == n and d.chatgpt.heard == 27
        b.attach(d.core)  # 画面を開き直しても戻らない
        d.core.on_text(b, json.dumps({"t": "ui", "mute": False}))
        assert d.fsm.ctx.state is State.REST and b.of("handoff")[-1]["active"]
        assert d.spontaneous_tick() is None
        d.chatgpt.end.set()
        await until(lambda: not d.handoff_on)
        assert d.fsm.ctx.state is State.IDLE and b.of("handoff")[-1] == {"t": "handoff", "active": False, "reason": "idle"}
        b.speak()  # 戻った直後の音も捨てる (ChatGPT の最後の声)
        assert d.ingress._n == n
        d.clock_.advance(HANDOFF_TAIL_S)
        b.speak()
        assert d.ingress._n > n
        assert [e["phase"] for e in handoff_events(store)] == ["start", "end"]
        assert store.query("SELECT COUNT(*) FROM turns")[0][0] == 0  # 呼びかけは会話の履歴に残さない
    asyncio.run(run())


def test_wake_failure_returns_to_idle(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["チャッピー"], result=cv.LOGIN)
        b.speak()
        await until(lambda: b.of("handoff") and not b.of("handoff")[-1]["active"])
        assert not d.handoff_on and d.fsm.ctx.state is State.IDLE
        assert b.of("handoff")[-1]["reason"] == "login" and d.chatgpt.started == 1
    asyncio.run(run())


def test_wake_while_ai_playing_cancels(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["こんにちは", "チャッピー"], ["やあ、今日はいい天気だね。"])
        b.auto_playback = False
        b.speak()
        await until(lambda: d.fsm.ctx.state is State.AI_PLAYING)
        ep = d.epoch.current
        b.speak()
        await until(lambda: d.handoff_on)
        assert d.fsm.ctx.state is State.REST and d.epoch.current == ep + 1
        assert b.of("cancel")[-1]["new_epoch"] == ep + 1
    asyncio.run(run())


def test_sentence_with_name_is_normal_talk(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["はチャッピーで大丈夫だよ"])
        b.speak()
        await until(lambda: len(d.seen_) == 1)
        assert not d.handoff_on and d.chatgpt.started == 0
    asyncio.run(run())


def test_no_wake_in_transcribe_mode(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["チャッピー"])
        d.core.on_text(b, json.dumps({"t": "ui", "transcribe": True}))
        b.speak()
        await until(lambda: b.of("transcript"))
        assert not d.handoff_on and d.chatgpt.started == 0
    asyncio.run(run())


def test_without_chatgpt_wake_is_normal_talk(tmp_path):
    async def run():
        d, b, store = make(tmp_path, ["チャッピー"])
        d.chatgpt = None
        b.speak()
        await until(lambda: len(d.seen_) == 1)
        assert not d.handoff_on
    asyncio.run(run())

