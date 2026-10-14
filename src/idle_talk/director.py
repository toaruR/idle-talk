"""Director 結線 (設計書 2 / 4.3 / 4.4)。状態機械 + WS サーバ + 入出力 + ストアを単一イベントループで繋ぐ。

fsm.Director を継承した StateDirector が遷移ごとに state メッセージを 1 件送出する。
通常応答の再生は ResponsePipeline.play_utterance、キャンセルは ResponsePipeline.cancel を唯一の入口とし、
履歴確定は _finalize_utterance に一本化する (不変条件 3 / 4)。
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import socket
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from dataclasses import replace

from .backchannel import Backchannel, event_payload
from .config import Config
from .echo_chain import CANCEL_KIND
from .epoch import Epoch
from .errors import ERR_OUT_OF_RANGE
from .fsm import Action, Ctx, Ev, Event, State, transition
from .fsm import Director as FsmDirector
from .history import PlaybackHistory, history_window, talk_log
from .ingress import Ingress
from .interrupt import judge
from .memory import MIN_UTTERANCE_CHARS, extract_messages, memory_prompt, parse_extracted, remember
from .metrics import MetricsRecorder
from .pipeline import ResponsePipeline
from .protocol import pack_down_chunk
from .server import WsCore
from .spontaneous import Spontaneous
from .store import Store
from .topics import mark_used, select_cards
from .vad import Vad
from .voice_id import (ASK_MAX, ASK_PROMPT, ASK_QUESTION, ASKED_PROMPT, MIN_VOICE_MS, UNNAMED_PROMPT, VoiceBook,
                       name_messages, parse_name, speaker_prompt)
from .web_search import link_items

log = logging.getLogger("idle_talk.director")

SYSTEM_PROMPT = ("あなたは気さくな日本語の雑談相手です。話し言葉で 2〜4 文ほど自然に返答してください。"
                 "相づちだけで終わらせず、自分の感想や小さな体験談を添え、ときどき質問や関連する新しい話題を自分から振ってください。"
                 "箇条書きや記号は使わないでください。")
# 自発発話で振る話題の種。毎回ランダムに 1 つ渡して、同じ話題の繰り返しを避ける。
SPONT_TOPICS = ("最近の天気や季節", "食べ物や料理", "休日の過ごし方", "最近気になっていること", "子どもの頃の思い出",
                "好きな音楽や映画", "旅行してみたい場所", "ちょっとした豆知識", "動物や生き物", "今日あったこと",
                "趣味や習い事", "仕事や勉強の息抜き", "もしもの話", "身近な道具や家電")
RADIO_PROMPT = ("(利用者は聞いているだけで返事はしません。ラジオのパーソナリティとして独り語りを続けてください。"
                "直前の話の続きでも、例えば{topic}のような新しい話題でも構いません。"
                "自分の感想やちょっとした体験談を交えて 3〜5 文で話し、質問や呼びかけで返事を求めないでください)")
# 話題カード (RSS) があれば SPONT_TOPICS より優先して 1 枚使う。概要は長すぎるとプロンプトを圧迫するので切り詰める
CARD_FACTS_MAX = 200
# カードで話すときに渡す直近の履歴ターン数。長いと独り語りの情景や言い回し (「雨の音」等) を引きずって繰り返す
CARD_TURNS = 2
CARD_RULES = "前の話の情景や言い回し (天気や音など) は繰り返さず、ニュースの内容を中心に話し、概要に無い事実は付け足さないでください。"
RADIO_CARD_PROMPT = ("(利用者は聞いているだけで返事はしません。ラジオのパーソナリティとして{topic}を紹介してください。"
                     + CARD_RULES + "自分の感想を交えて 3〜5 文で話し、質問や呼びかけで返事を求めないでください)")
CARD_PROMPT = ("(利用者は黙っています。{topic}を話題に話しかけてください。"
               + CARD_RULES + "自分の感想を一言添えて 2〜3 文で話し、最後に軽く質問してください)")
FRAME_BYTES = 640  # 20ms @ 16kHz s16le
# ブラウザ (web/player.js MAX_LOOKAHEAD) が再生中チャンクの後ろに保持する数。超えて送ると dropped で捨てられる
DOWN_LOOKAHEAD = 1
MEM_BATCH = 4  # 記憶抽出に 1 回で渡す利用者発話の上限
ECHO_SUPPRESSED_KIND = "echo_suppressed"  # AI 再生中に拾った自声の認識結果を捨てた 1 回 = 1 行


def _alnum(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def wav_to_pcm(wav: bytes) -> bytes:
    """WAV の data チャンク本体を返す。data チャンクが無ければそのまま返す。"""
    if wav[:4] == b"RIFF" and wav[8:12] == b"WAVE":
        i = 12
        while i + 8 <= len(wav):
            cid, size = wav[i:i + 4], int.from_bytes(wav[i + 4:i + 8], "little")
            if cid == b"data":
                return wav[i + 8:i + 8 + size]
            i += 8 + size + (size & 1)
    return wav


class StateDirector(FsmDirector):
    """遷移ごとに on_state(ctx) を呼ぶ。ctx.epoch は共有 Epoch に追従させる。"""

    def __init__(self, cfg: Config | None = None, epoch: Epoch | None = None,
                 on_actions: Callable[[tuple[Action, ...]], None] | None = None,
                 on_state: Callable[[Ctx], None] | None = None) -> None:
        super().__init__(cfg, None, on_actions)
        self.shared_epoch = epoch
        self.on_state = on_state

    def pump(self) -> list[Action]:
        out: list[Action] = []
        while self._q:
            if self.shared_epoch is not None and self.ctx.epoch != self.shared_epoch.current:
                self.ctx = replace(self.ctx, epoch=self.shared_epoch.current)  # 再接続等による bump に追従
            before = self.ctx.state
            self.ctx, acts = transition(self.ctx, self._q.popleft(), self.cfg)
            if self.ctx.state is not before and self.on_state:
                self.on_state(self.ctx)
            if acts:
                out.extend(acts)
                if self._on_actions:
                    self._on_actions(acts)
        return out


class IdleTalkDirector:
    def __init__(self, store: Store, llm_stream: Callable[[list[dict]], object],
                 synthesize: Callable[[str], bytes], cancel_tts: Callable[[], None],
                 vad: Vad, asr=None, cfg: Config | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 now_ms: Callable[[], int] = lambda: int(time.time() * 1000),
                 metrics: MetricsRecorder | None = None, app_commit: str = "unknown",
                 gpu1_load: Callable[[], bool] = lambda: False) -> None:
        self.cfg = cfg or Config()
        self.metrics = metrics  # M1 計測 (設計書 11.1)。None なら記録しない
        self.app_commit = app_commit or "unknown"
        self._gpu1_load = gpu1_load
        self._m_turn = 0
        self._m_marked: set[str] = set()
        self._m_started_at = now_ms()
        self._m_closed = False
        self.store = store
        self.epoch = Epoch()
        self.now_ms = now_ms
        self.clock = clock
        self._ic_start_ms = 0  # INTERRUPT_CHECK 突入時刻 (clock 基準 ms)
        self.path_calls: list[tuple[str, str]] = []  # (route, 関数名): 不変条件 3 の記録
        self.core = WsCore(self.cfg, self.epoch, clock, on_audio=self._on_up_audio,
                           on_playback=self._on_playback, on_ui=self._on_ui, on_hello=self._on_hello)
        self.fsm = StateDirector(self.cfg, self.epoch, on_actions=self._run_actions, on_state=self._send_state)
        self.history = PlaybackHistory(store, self.cfg)
        self._llm = llm_stream
        self._synthesize = synthesize
        self._bc_wavs: dict[str, bytes] = {}
        self._bc_active: str | None = None
        self._bc_utts: set[int] = set()
        self._down_q: deque[tuple[int, int, int, bytes]] = deque()  # 送出待ちの下りチャンク (ep, utt, chunk, pcm)
        self._down_sent: set[tuple[int, int, int]] = set()  # 送出済みで finished/dropped 未報告
        self._ai_text = ""  # 直近の AI 発話 (相槌除く) の送出済み本文。自声の認識結果を捨てる判定に使う
        self._speech_in_ai = False  # 現在の利用者発話が AI 再生中 (自発発話を含む) に始まったか
        self._spont_hold = False  # 自発発話の再生中に音を拾い、取消を認識結果まで保留している (duck 中)
        self.backchannel: Backchannel | None = None
        self._bc_silence_ms = 0
        self._bc_speaking = False
        self.pipeline = ResponsePipeline(llm_stream, self._synth_or_clip, cancel_tts, self._on_chunk,
                                         self._emit, self.epoch)
        # ASR は単一ワーカーで順に実行し、WS の受信処理 (イベントループ) を止めない
        self._asr_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")
        self.ingress = Ingress(self.fsm, vad, asr, on_transcript=self._on_transcript,
                               on_error=lambda c, m: self._emit({"t": "error", "code": c, "message": m}),
                               clock=clock, on_speech_end=lambda: self._m_mark("eot_wait"),
                               offload=self._offload_asr)
        self._text = ""
        self._task: asyncio.Task | None = None
        self._cancels: set[asyncio.Task] = set()
        self._gen_done = False
        self._cur: dict | None = None
        self.spontaneous = Spontaneous(self.cfg, record=self._bc_record)  # 既定オフ (設計書 6.1 / 8.1)
        self._spont_task: asyncio.Task | None = None
        self._last_op = clock()  # 最後の利用者操作時刻 (IDLE_TIMEOUT 判定)
        self._idle_since: float | None = clock()  # IDLE になった時刻 (自発発話の沈黙計測)。IDLE 以外は None
        self._mem_pending: list[tuple[int, str]] = []  # 記憶抽出待ちの利用者発話 (utterance_id, text)
        self._mem_task: asyncio.Task | None = None
        self._mem_stop = threading.Event()  # 取消時に抽出スレッドの LLM 受信を打ち切る (llama-server を応答に譲る)
        self.set_speaker: Callable[[int], None] | None = None  # 画面で選んだ話者 ID を TTS へ (main が結線)
        self.topic_worker = None  # TopicWorker。状態変化を伝えて休止中は取得を止める (設計書 7)
        self.web_search: Callable[[str], list[dict]] | None = None  # 利用者への返答で調べものをする (main が結線)
        # 話者識別 (設計書 6.4)。enable_voice_id で有効にする
        self.voices: VoiceBook | None = None
        self._voice: tuple[int, object] | None = None  # 認識結果待ちの声の特徴 (utterance_id, 埋め込み)
        self._speaker: int | None = None  # 今の話し相手 (speakers.id)。None = 知らない声 / 未判定
        self._asks: Counter[int] = Counter()  # 起動してから名前を尋ねた回数 (話者別)
        self._ask_sid: int | None = None  # 名前を尋ねた相手。その人の次の発話から名前を抜き出す
        self._ask_plan: int | None = None  # 名前を尋ねる応答を生成中の相手。最後まで再生されたら _ask_sid にする
        self._name_pending: list[str] = []  # 名前の抜き出し待ちの発話

    # ---- M1 計測 (設計書 11.1): 1 ターン 5 段階を events に kind 別で記録 ----
    def _m_mark(self, stage: str) -> None:
        if self.metrics is None:
            return
        if stage == "eot_wait":
            if "eot_wait" in self._m_marked:  # 前ターンが未完 (中断・無効発話): 破棄して新ターン
                self._m_turn += 1
                self._m_marked = set()
        elif "eot_wait" not in self._m_marked or stage in self._m_marked:
            return
        self._m_marked.add(stage)
        self.metrics.mark(self._m_turn, stage, self.now_ms(), self.epoch.current, bool(self._gpu1_load()))
        if stage == "play_start":  # ターン完了
            self._m_turn += 1
            self._m_marked = set()

    def _m_abandon(self) -> None:
        """相槌判定の発話はターンとして数えない (G1 から除外)。"""
        if "eot_wait" in self._m_marked:
            self._m_turn += 1
            self._m_marked = set()

    def close(self) -> None:
        """run_meta 行を書いてから store を閉じる。sample_count は run_meta 以外の events 行数。"""
        if self.metrics is not None and not self._m_closed:
            self._m_closed = True
            n = self.store.query("SELECT COUNT(*) FROM events WHERE kind!='run_meta'")[0][0]
            ended = max(self.now_ms(), self._m_started_at)
            meta = {"host": socket.gethostname() or "unknown", "started_at": self._m_started_at,
                    "ended_at": ended, "app_commit": self.app_commit, "sample_count": n}
            self.store.add_event(ended, self.epoch.current, "run_meta", json.dumps(meta))
        self._asr_pool.shutdown(wait=False, cancel_futures=True)
        self.store.close()

    # ---- 相槌 (設計書 6.1): 事前合成クリップも通常応答と同じ play_utterance / cancel 経路を通る ----
    def _synth_or_clip(self, text: str) -> bytes:
        if text == self._bc_active and text in self._bc_wavs:
            return self._bc_wavs[text]
        self._m_mark("llm_first_clause")  # LLM が最初の句を返し TTS へ渡す時点
        return self._synthesize(text)

    def enable_backchannel(self, clips: list[tuple[str, bytes]]) -> Backchannel:
        self._bc_wavs = dict(clips)
        self.backchannel = Backchannel(clips, self._bc_play, lambda: self._cancel_path("backchannel"),
                                       self._bc_record, self.cfg, self.clock)
        return self.backchannel

    def enable_voice_id(self, embed: Callable[[bytes], object], threshold: float) -> VoiceBook:
        self.voices = VoiceBook(self.store, threshold)
        self.ingress.embed, self.ingress.on_voice, self.ingress.voice_min_ms = embed, self._on_voice, MIN_VOICE_MS
        return self.voices

    def _bc_play(self, text: str, wav: bytes) -> None:
        self._bc_active = text
        self.path_calls.append(("backchannel", "play_utterance"))
        t = asyncio.get_running_loop().create_task(self.pipeline.play_utterance(iter([text])))
        self._cancels.add(t)
        t.add_done_callback(self._cancels.discard)

    def _bc_record(self, kind: str, payload: dict) -> None:
        self.store.add_event(self.now_ms(), self.epoch.current, kind, event_payload(payload))

    def backchannel_tick(self, silence_ms: float) -> bool:
        """無音継続 (ms) を通知する。発話中 (非 AI 再生) のみ有効。"""
        bc = self.backchannel
        if bc is None or self.fsm.ctx.state is State.AI_PLAYING:
            return False
        return bc.on_silence(silence_ms)

    def _bc_track(self, frames: int) -> None:
        speaking = self.ingress._buf is not None
        if speaking:
            if not self._bc_speaking:
                self.backchannel.on_speech_start()
            self._bc_speaking, self._bc_silence_ms = True, 0
        elif self._bc_speaking:
            self._bc_silence_ms += frames * 20
            self.backchannel_tick(self._bc_silence_ms)

    # ---- 送信 ----
    def _emit(self, msg: dict) -> None:
        if self.core.conn is not None:
            self.core.conn.send_text(json.dumps(msg, ensure_ascii=False))

    def _send_down(self, ep: int, utt: int, chunk: int, pcm: bytes) -> None:
        self._down_q.append((ep, utt, chunk, pcm))
        self._flush_down()

    def _flush_down(self) -> None:
        """再生中 1 + 先読み DOWN_LOOKAHEAD を超えないよう、再生終了の報告を待って次を送る。旧世代は捨てる。"""
        cur = self.epoch.current
        self._down_sent = {k for k in self._down_sent if k[0] == cur}  # 旧世代の報告はサーバで捨てられ届かない
        while self._down_q and len(self._down_sent) <= DOWN_LOOKAHEAD:
            ep, utt, chunk, pcm = self._down_q.popleft()
            if ep != cur or self.core.conn is None:
                continue
            self._down_sent.add((ep, utt, chunk))  # 送出前に登録 (送出中に再生終了の報告が来ても取りこぼさない)
            self.core.conn.send_bytes(pack_down_chunk(ep, utt, chunk, pcm))

    def _on_hello(self) -> None:
        ctx = self.fsm.ctx
        self._down_q.clear()  # 新しい接続: 前の接続向けの送出待ち・未報告は無効
        self._down_sent.clear()
        if self.spontaneous.pending:  # 前の接続で再生されずに残った自発発話は完了通知が来ないので取消す
            self._cancel_spontaneous()
        self.spontaneous.on_user_speech()  # 新しい聞き手: 無反応による待ち時間の延長を戻す
        if ctx.state is State.IDLE:  # 沈黙は接続時から測る (接続前の時間で即座に話しかけない)
            self._idle_since = self.clock()
        self._emit({"t": "state", "state": ctx.state.value, "muted": ctx.muted})

    def _send_state(self, ctx: Ctx) -> None:
        if ctx.state is not State.IDLE:
            self._idle_since = None
        elif self._idle_since is None:
            self._idle_since = self.clock()
        self._emit({"t": "state", "state": ctx.state.value, "muted": ctx.muted})
        if self.topic_worker is not None:
            self.topic_worker.on_state(ctx.state.value)
        if ctx.state is State.INTERRUPT_CHECK:
            # 認識結果が出るまで duck のまま待つ。語が無ければ再生を続け (行14 / 15)、語があれば判定する (行10 / 11)
            self._ic_start_ms = int(self.clock() * 1000)

    def request_rest(self) -> None:
        """LLM_RETRY 超過など復旧不能な障害で休止へ移る (設計書 12.4「超過で休止 + warn」)。"""
        self.fsm.enqueue(Event(Ev.FAILURES))
        self._pump()

    def _judge_interrupt(self, text: str) -> str:
        """INTERRUPT_CHECK 中の ASR 結果を判定して JUDGED を送る (設計書 5.1)。"""
        t_ms = int(self.clock() * 1000)
        verdict = judge(text, self._ic_start_ms, t_ms, self.cfg)
        self.fsm.enqueue(Event(Ev.JUDGED, verdict=verdict, elapsed_ms=t_ms - self._ic_start_ms))
        if verdict != "backchannel":  # 発話は既に終了済み: USER_SPEAKING から終話確認へ進める
            self.fsm.enqueue(Event(Ev.VAD_END, seg_ms=self.cfg.MIN_SEG))
        return verdict

    def _pump(self) -> None:
        self.fsm.pump()

    # ---- 入力 ----
    def _offload_asr(self, work: Callable[[], object], done: Callable[[object], None]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # ループ外 (同期テスト等): その場で実行
            done(work())
            return
        fut = self._asr_pool.submit(work)
        fut.add_done_callback(lambda f: loop.call_soon_threadsafe(self._asr_finished, f, done))

    def _asr_finished(self, fut: Future, done: Callable[[object], None]) -> None:
        if fut.cancelled():
            return
        done(fut.result())
        self._pump()

    def _on_up_audio(self, seq: int, pcm: bytes) -> None:
        if not self.core.hello_done:
            return
        cur = self.epoch.current  # 再生中 = 現行世代の送出済みチャンクが再生終了の報告待ち (VAD の自動校正が使う)
        self.ingress.vad.playing = any(k[0] == cur for k in self._down_sent)
        for i in range(0, len(pcm), FRAME_BYTES):
            before, was_speaking = self.fsm.ctx.state, self.ingress._buf is not None
            self.ingress.feed(pcm[i:i + FRAME_BYTES])
            if not was_speaking and self.ingress._buf is not None:
                spont_playing = self.spontaneous.pending and self._cur is not None
                self._speech_in_ai = before in (State.AI_PLAYING, State.INTERRUPT_CHECK) or spont_playing
            if self.backchannel is not None:
                self._bc_track(1)

    def _on_ui(self, settings: dict) -> None:
        try:
            self.spontaneous.apply_settings(settings)
        except ValueError as e:
            self._emit({"t": "error", "code": ERR_OUT_OF_RANGE, "message": str(e)[:200]})
        if "speaker" in settings and self.set_speaker is not None:
            try:
                self.set_speaker(settings["speaker"])
            except ValueError as e:
                self._emit({"t": "error", "code": ERR_OUT_OF_RANGE, "message": str(e)[:200]})
        self._last_op = self.clock()
        muted = bool(settings.get("mute"))
        if muted and not self.fsm.ctx.muted:
            self.fsm.enqueue(Event(Ev.MUTE))
        elif not muted and (self.fsm.ctx.muted or self.fsm.ctx.state is State.REST):
            self.fsm.enqueue(Event(Ev.WAKE))  # 無操作・連続失敗による休止もミュートなしの操作で解除する
        self._pump()

    def _on_transcript(self, uid: int, text: str) -> bool:
        """認識結果を受ける。語が無い・自声なら False (Ingress が捨てて DISCARDED を送る)。"""
        in_ai, self._speech_in_ai = self._speech_in_ai, False
        if not _alnum(text):
            log.info("transcript without words discarded: utterance %d %r", uid, text)
            return False
        if in_ai and self._is_echo(uid, text):
            return False
        self._text = text
        self._identify(uid)
        self._m_mark("asr_done")
        self._release_spont_hold(speech=True)
        if self.fsm.ctx.state is State.INTERRUPT_CHECK and self._judge_interrupt(text) == "backchannel":
            self._m_abandon()
            return True  # 相槌は履歴に積まない
        if self.fsm.ctx.state is State.USER_SPEAKING and self.ingress._buf is None:
            # T_confirm 超過で割り込み扱いになった後に届いた認識結果: 発話は終わっているので終話確認へ進める
            self.fsm.enqueue(Event(Ev.VAD_END, seg_ms=self.cfg.MIN_SEG))
        talk_log.info("USER: %s", text)
        self.store.add_turn(self.now_ms(), "user", text, 0, None, uid)  # user 行は completed=0 (ai の再生完了のみ 1)
        if self._ask_sid is not None and self._speaker == self._ask_sid:  # 名前を尋ねた相手の返事
            self._name_pending = self._name_pending[-1:] + [text]
        if len(text) >= MIN_UTTERANCE_CHARS:  # 古い分は捨て、直近 MEM_BATCH 発話だけ抽出に回す
            self._mem_pending = self._mem_pending[-(MEM_BATCH - 1):] + [(uid, text)]
        return True

    # ---- 話者識別 (設計書 6.4) ----
    def _on_voice(self, uid: int, emb: object) -> None:
        self._voice = (uid, emb)

    def _identify(self, uid: int) -> None:
        """自声・語なしを除いた発話の声で話し相手を決める。声の特徴が無い (短い発話) なら直前の相手のまま。"""
        v, self._voice = self._voice, None
        if self.voices is None or v is None or v[0] != uid:
            return
        sid = self.voices.identify(v[1], self.now_ms())
        if sid != self._speaker:
            log.info("speaker changed: %s -> %s (%s)", self._speaker, sid, self.voices.name(sid) or "名前未登録")
        self._speaker = sid

    def _speaker_section(self, ask: bool) -> str:
        """今の話し相手の節。名前の分からない登録済みの相手には、ask なら ASK_MAX 回まで名前を尋ねさせる。"""
        self._ask_plan = None
        if self.voices is None or self._speaker is None:
            return ""
        name = self.voices.name(self._speaker)
        if not name and self._ask_sid == self._speaker and self._name_pending:  # 尋ねた直後の返事 (抜き出しは再生後)
            return ASKED_PROMPT
        if name:
            return speaker_prompt(name)
        if not ask or self._asks[self._speaker] >= ASK_MAX:
            return UNNAMED_PROMPT
        self._ask_plan = self._speaker  # 尋ねたことにするのは再生し終えてから (割り込みで聞こえていない返事を名前にしない)
        return ASK_PROMPT

    def _asked(self, heard: bool) -> None:
        """名前を尋ねる応答が最後まで再生されたら (heard)、その人の次の発話から名前を抜き出す。"""
        sid, self._ask_plan = self._ask_plan, None
        if sid is None or not heard:
            return
        self._asks[sid] += 1
        self._ask_sid, self._name_pending = sid, []

    async def _learn_name(self, utterances: list[str], stop: threading.Event) -> None:
        sid = self._ask_sid
        try:
            text = await asyncio.to_thread(self._collect, name_messages(utterances), stop)
        except Exception as e:  # noqa: BLE001  LLM の失敗で会話を止めない (次に尋ねたときに再挑戦)
            log.warning("name extraction failed: %s", e)
            return
        if text is None or asyncio.current_task() is not self._mem_task or sid != self._ask_sid:
            return
        self._name_pending = []
        name, self._ask_sid = parse_name(text), None  # 名乗らなかったら、次の応答で (ASK_MAX まで) 改めて尋ねる
        if name is None:
            return
        kept = self.voices.set_name(sid, name, self.now_ms())
        if self._speaker == sid:
            self._speaker = kept
        log.info("speaker %d named %r%s", sid, name, f" (merged into {kept})" if kept != sid else "")

    def _is_echo(self, uid: int, text: str) -> bool:
        """AI 再生中に始まった発話の認識結果が直近 AI 発話の本文に含まれる = スピーカーから回り込んだ自声 (§6.2)。

        句読点・空白を除いた部分文字列一致のみ (閾値なし)。言い換えや部分的な誤認識は検出できない。"""
        if _alnum(text) not in _alnum(self._ai_text):
            return False
        log.info("echo discarded: utterance %d %r", uid, text)
        if self.metrics is not None:
            self.store.add_event(self.now_ms(), self.epoch.current, ECHO_SUPPRESSED_KIND,
                                 json.dumps({"utterance_id": uid, "text": text}, ensure_ascii=False))
        return True

    def _on_playback(self, msg: dict) -> None:
        self.history.on_playback(msg)
        if self.metrics is not None and msg["utterance_id"] not in self._bc_utts:  # 相槌クリップは turns に載らないので G3/履歴照合から除外
            self.store.add_event(self.now_ms(), self.epoch.current, "playback", json.dumps(
                {k: msg[k] for k in ("epoch", "utterance_id", "chunk_id", "state", "position_ms")}))
        if msg["state"] != "started":
            self._down_sent.discard((msg["epoch"], msg["utterance_id"], msg["chunk_id"]))
            self._flush_down()
        c = self._cur
        if c and msg["state"] == "started" and msg["epoch"] == c["epoch"] and msg["utterance_id"] == c["utt"]                 and msg["chunk_id"] == 0:
            self._m_mark("play_start")
        if c and msg["epoch"] == c["epoch"] and msg["utterance_id"] == c["utt"] and msg["state"] != "started":
            # dropped も終了扱い: 待ち続けると再生完了が来ず、応答・自発発話が終わらない
            c["finished"].add(msg["chunk_id"])
            self._check_done()

    # ---- 出力 (pipeline.on_audio) ----
    def _on_chunk(self, ep: int, utt: int, chunk: int, text: str, wav: bytes) -> None:
        if not self.epoch.is_current(ep):  # 旧世代チャンクは送出しない (G3)
            self.pipeline._discard(ep, text)
            return
        if self._bc_active is not None and text == self._bc_active:  # 相槌: 音声のみ送出、FSM・履歴には触れない
            self._bc_active = None
            self._bc_utts.add(utt)
            self._send_down(ep, utt, chunk, wav_to_pcm(wav))
            return
        if self._cur is None or self._cur["utt"] != utt:
            self._cur = {"epoch": ep, "utt": utt, "chunks": set(), "finished": set()}
        self._cur["chunks"].add(chunk)
        self._ai_text = text if chunk == 0 else self._ai_text + text
        if chunk == 0:
            self._m_mark("tts_first")
        self.history.register_chunk(ep, utt, chunk, text)
        self._emit({"t": "text", "epoch": ep, "utterance_id": utt, "text": text})
        self._send_down(ep, utt, chunk, wav_to_pcm(wav))
        if chunk == 0:
            self.fsm.enqueue(Event(Ev.FIRST_CLAUSE, epoch=ep))
            self._pump()

    def _played_out(self, c: dict | None) -> bool:
        return bool(c and self._gen_done and c["chunks"] and c["finished"] >= c["chunks"])

    def _check_done(self) -> None:
        c = self._cur
        if self.spontaneous.pending and self._gen_done and self.fsm.ctx.state is State.IDLE:
            if self._spont_hold:  # 利用者の発話か雑音かの判定待ち: 完了は保留解除で確定する
                return
            if c is None or self._played_out(c):  # 自発発話の再生完了
                self._finalize_utterance("spontaneous", interrupted=False)
                self.spontaneous.done()
                self._idle_since = self.clock()  # 自発発話は IDLE のまま再生するので、沈黙の計測をここで始め直す
            return
        if self._played_out(c):
            self.fsm.enqueue(Event(Ev.PLAYBACK_DONE, epoch=c["epoch"]))
            self._pump()

    # ---- アクション実行 ----
    def _run_actions(self, acts: tuple[Action, ...]) -> None:
        for a in acts:
            h = getattr(self, "_act_" + a.name, None)
            if h:
                h(a)

    def _act_call_llm(self, a: Action) -> None:
        if self.backchannel is not None:  # T_silence 到達: 未完の相槌を取消 (T-BC)
            self.backchannel.resolve(self.cfg.T_silence)
            self._bc_speaking, self._bc_silence_ms, self._bc_active = False, 0, None
        self._gen_done = False
        self._task = asyncio.get_running_loop().create_task(self._respond())

    def _act_cancel_generation(self, a: Action) -> None:
        if self._task and not self._task.done():
            self._task.cancel()

    def _act_cancel(self, a: Action) -> None:
        self._cancel_path("interrupt")
        if self.metrics is None or self.fsm.ctx.state is State.REST:  # 休止遷移の取消は割り込みではない
            return
        self.store.add_event(self.now_ms(), self.epoch.current, CANCEL_KIND,
                             json.dumps({"epoch": self.epoch.current}))  # 確定した割り込み取消 1 回 = 1 行 (M5)

    def _act_abort_background(self, a: Action) -> None:
        """利用者発話開始: 準備中/再生中の自発発話は duck して認識結果まで取消を保留する (雑音・自声で止めない)。"""
        self._last_op = self.clock()
        if self.spontaneous.pending:
            if not self._spont_hold:
                self._spont_hold = True
                self._act_duck(a)
        else:
            self.spontaneous.on_user_speech()
        self._cancel_memory()

    def _act_discard_segment(self, a: Action) -> None:
        self._release_spont_hold(speech=False)

    def _release_spont_hold(self, speech: bool) -> None:
        """保留中の自発発話を、語が認識されたら取消し (反応ありとして待ち時間を戻す)、雑音なら再生を戻す。"""
        if not self._spont_hold:
            return
        self._spont_hold = False
        if speech:
            self._cancel_spontaneous()
            self.spontaneous.on_user_speech()
        else:
            self._emit({"t": "resume", "epoch": self.epoch.current})
            self._check_done()  # 保留中に再生し終えていれば完了扱いにする

    def _act_stop_background(self, a: Action) -> None:
        if self.spontaneous.pending:
            self._cancel_spontaneous()
        self._cancel_memory()

    def _cancel_spontaneous(self) -> None:
        self._spont_hold = False
        if self._spont_task and not self._spont_task.done():
            self._spont_task.cancel()
        self.spontaneous.cancel()
        self._cancel_path("spontaneous")

    def _act_commit_history(self, a: Action) -> None:
        self._last_op = self.clock()
        self._asked(heard=True)
        self._finalize_utterance("response", interrupted=False)
        self._start_memory()

    def _act_duck(self, a: Action) -> None:
        self._emit({"t": "duck", "epoch": self.epoch.current, "gain": self.cfg.DUCK_GAIN})

    def _act_resume(self, a: Action) -> None:
        self._emit({"t": "resume", "epoch": self.epoch.current})
        if self._played_out(self._cur):  # 判定中 (INTERRUPT_CHECK) に再生し終えた: その PLAYBACK_DONE は捨てられている
            self.fsm.enqueue(Event(Ev.PLAYBACK_DONE, epoch=self._cur["epoch"]))

    # ---- 自発発話 / 無操作休止 (設計書 4.2 / 6.1) ----
    def spontaneous_tick(self, now: float | None = None) -> str | None:
        """定期呼び出し。'rest' (IDLE_TIMEOUT 遷移) / 'spontaneous' (発火) / None を返す。"""
        now = self.clock() if now is None else now
        idle = self.fsm.ctx.state is State.IDLE
        if idle and not self.spontaneous.radio and now - self._last_op >= self.cfg.IDLE_TIMEOUT:  # ラジオは聞き流し前提
            self.fsm.enqueue(Event(Ev.IDLE_TIMEOUT))
            self._pump()
            return "rest"
        listening = self.core.conn is not None and self.core.hello_done  # 聞き手がいないと再生完了が来ない
        # 記憶抽出と LLM を取り合わない。ラジオは間を空けない (llama-server は複数スロットで並行処理する)
        if not self.spontaneous.radio and self._mem_task is not None and not self._mem_task.done():
            return None
        if listening and self.spontaneous.should_fire(now, self._idle_since if idle else None):
            self.spontaneous.fire(now)
            self._gen_done = False
            self._spont_task = asyncio.get_running_loop().create_task(self._spontaneous_respond())
            return "spontaneous"
        return None

    async def _spontaneous_respond(self) -> None:
        topic, card = self._pick_topic()
        if card is not None and (card["source_url"] or "").startswith(("http://", "https://")):
            # カードで話すニュースは聞かれる前に画面の URL 欄へ出し、「URL を教えて」にも答えられるよう覚えておく
            self.pipeline.links = link_items([{"title": card["hook"], "url": card["source_url"]}])
            self._emit({"t": "links", "items": self.pipeline.links})
        messages = self._chat_messages(CARD_TURNS if card else None)
        if card:
            prompt = (RADIO_CARD_PROMPT if self.spontaneous.radio else CARD_PROMPT).format(topic=topic)
        elif self.spontaneous.radio:
            prompt = RADIO_PROMPT.format(topic=topic)
        else:
            prompt = ("(利用者は黙っています。これまでの会話と違う話題を自分から振ってください。"
                      f"例えば{topic}について、自分の話を一言添えて 2〜3 文で話しかけ、最後に軽く質問してください)")
        messages.append({"role": "user", "content": prompt})
        self.path_calls.append(("spontaneous", "play_utterance"))  # 通常応答と同じ play_utterance を通る
        me = asyncio.current_task()
        await self.pipeline.respond(messages)
        if self._spont_task is not me or not self.spontaneous.pending:
            return
        self._gen_done = True
        self._check_done()

    def _pick_topic(self) -> tuple[str, dict | None]:
        """(話題, 使ったカード)。未使用・未期限切れの話題カードがあれば使用済みにしてニュースを返す。無ければ SPONT_TOPICS から選ぶ。"""
        cards = select_cards(self.store, self.now_ms(), limit=2)
        if len(cards) < 2 and self.topic_worker is not None:  # これで在庫切れ: 次の発話までにニュースを仕入れる
            self.topic_worker.request_refill()
        if not cards:
            return f"「{random.choice(SPONT_TOPICS)}」", None
        card = cards[0]
        mark_used(self.store, card["id"], self.now_ms())
        facts = re.sub(r"<[^>]+>", "", card["facts"] or "").strip()[:CARD_FACTS_MAX]
        return f"最近見かけたニュース「{card['hook']}」" + (f"(概要: {facts})" if facts else ""), card

    # ---- 記憶 (設計書 3.4): 応答の再生完了後に、溜まった利用者発話から抜き出して保存する ----
    def _chat_messages(self, turns: int | None = None, ask: bool = False) -> list[dict]:
        """system + 履歴。turns を指定すると直近 turns 件だけ渡す。ask なら名前の分からない相手に名前を尋ねさせる。"""
        history = [{"role": "assistant" if t["role"] == "ai" else "user", "content": t["text"]}
                   for t in history_window(self.store, self.cfg) if t["text"]]
        system = SYSTEM_PROMPT + memory_prompt(self.store) + self._speaker_section(ask)
        return [{"role": "system", "content": system}] + (
            history[-turns:] if turns else history)

    def _start_memory(self) -> None:
        """応答と LLM を取り合わないよう、再生完了 (IDLE) 後にだけ走らせる。利用者発話で取消し、次回まとめて再試行する。"""
        if not (self._mem_pending or self._name_pending) or (self._mem_task is not None and not self._mem_task.done()):
            return
        self._mem_stop = threading.Event()
        self._mem_task = asyncio.get_running_loop().create_task(
            self._background(list(self._mem_pending), list(self._name_pending), self._mem_stop))

    async def _background(self, batch: list[tuple[int, str]], names: list[str], stop: threading.Event) -> None:
        """名前の抜き出し (尋ねた直後なので先に) と記憶の抽出を順に行う。"""
        if names:
            await self._learn_name(names, stop)
        if batch and not stop.is_set():
            await self._extract_memory(batch, stop)

    def _cancel_memory(self) -> None:
        self._mem_stop.set()
        if self._mem_task is not None and not self._mem_task.done():
            self._mem_task.cancel()

    def _collect(self, messages: list[dict], stop: threading.Event) -> str | None:
        out: list[str] = []
        for tok in self._llm(messages):
            if stop.is_set():
                return None
            out.append(tok)
        return "".join(out)

    async def _extract_memory(self, batch: list[tuple[int, str]], stop: threading.Event) -> None:
        try:
            text = await asyncio.to_thread(self._collect, extract_messages([t for _, t in batch]), stop)
        except Exception as e:  # noqa: BLE001  LLM の失敗で会話を止めない (未処理分は次回に回す)
            log.warning("memory extraction failed: %s", e)
            return
        if text is None or asyncio.current_task() is not self._mem_task:
            return
        done = {uid for uid, _ in batch}
        self._mem_pending = [p for p in self._mem_pending if p[0] not in done]
        items = parse_extracted(text)
        if items:
            remember(self.store, items, batch[-1][0], self.now_ms())
            log.info("memory saved: %s", " / ".join(items))

    # ---- 唯一の再生/キャンセル/確定経路 (不変条件 3) ----
    async def _respond(self) -> None:
        messages = self._chat_messages(ask=True)
        self.path_calls.append(("response", "play_utterance"))
        me = asyncio.current_task()
        ep = self.epoch.current
        try:
            await self.pipeline.respond(messages, search=self.web_search,
                                        tail=ASK_QUESTION if self._ask_plan is not None else "")
        except asyncio.CancelledError:  # 準備中の再発話で生成が取り消された: 旧世代の結果は破棄扱い (E15)
            self.pipeline._discard(ep, "<cancelled generation>")
            raise
        if self._task is not me:  # キャンセル後に走り終えた古い応答は状態に触れない
            return
        self._gen_done = True
        if self._cur is None and self.fsm.ctx.state is State.PREPARING:  # 音声ゼロ (TTS 全失敗)
            self.fsm.enqueue(Event(Ev.FIRST_CLAUSE, epoch=self.epoch.current))
            self.fsm.enqueue(Event(Ev.PLAYBACK_DONE, epoch=self.epoch.current))
            self._pump()
        else:
            self._check_done()

    def _cancel_path(self, route: str) -> None:
        """pipeline.cancel は await を含まないので eager 実行して epoch を即時に確定させる。"""
        self.path_calls.append((route, "cancel"))
        if route == "interrupt":
            self._asked(heard=False)
        t = asyncio.Task(self.pipeline.cancel(), loop=asyncio.get_running_loop(), eager_start=True)
        self._cancels.add(t)
        t.add_done_callback(self._cancels.discard)
        self._finalize_utterance(route, interrupted=True)

    def _finalize_utterance(self, route: str, interrupted: bool) -> None:
        self.path_calls.append((route, "_finalize_utterance"))
        c, self._cur = self._cur, None
        if c is None:
            return
        self.history.commit(c["epoch"], c["utt"], self.now_ms(), interrupted=interrupted)
