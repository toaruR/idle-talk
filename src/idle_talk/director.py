"""Director 結線 (設計書 2 / 4.3 / 4.4)。状態機械 + WS サーバ + 入出力 + ストアを単一イベントループで繋ぐ。

fsm.Director を継承した StateDirector が遷移ごとに state メッセージを 1 件送出する。
通常応答の再生は ResponsePipeline.play_utterance、キャンセルは ResponsePipeline.cancel を唯一の入口とし、
履歴確定は _finalize_utterance に一本化する (不変条件 3 / 4)。
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from collections.abc import Callable
from dataclasses import replace

from .backchannel import Backchannel, event_payload
from .config import Config
from .echo_chain import CANCEL_KIND, ECHO_CHAIN_KIND
from .epoch import Epoch
from .errors import ERR_OUT_OF_RANGE
from .fsm import Action, Ctx, Ev, Event, State, transition
from .fsm import Director as FsmDirector
from .history import PlaybackHistory, history_window
from .ingress import Ingress
from .interrupt import judge
from .metrics import MetricsRecorder
from .pipeline import ResponsePipeline
from .protocol import pack_down_chunk
from .server import WsCore
from .spontaneous import Spontaneous
from .store import Store
from .vad import Vad

log = logging.getLogger("idle_talk.director")

SYSTEM_PROMPT = "あなたは気さくな日本語の雑談相手です。短い文で自然に返答してください。"
FRAME_BYTES = 640  # 20ms @ 16kHz s16le


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
                           on_playback=self._on_playback, on_ui=self._on_ui)
        self.fsm = StateDirector(self.cfg, self.epoch, on_actions=self._run_actions, on_state=self._send_state)
        self.history = PlaybackHistory(store, self.cfg)
        self._synthesize = synthesize
        self._bc_wavs: dict[str, bytes] = {}
        self._bc_active: str | None = None
        self._bc_utts: set[int] = set()
        self._ai_text = ""  # 直近の AI 発話 (相槌除く) の送出済み本文。自声起因の応答連鎖の判定に使う
        self._speech_in_ai = False  # 現在の利用者発話が AI 再生中に始まったか
        self.backchannel: Backchannel | None = None
        self._bc_silence_ms = 0
        self._bc_speaking = False
        self.pipeline = ResponsePipeline(llm_stream, self._synth_or_clip, cancel_tts, self._on_chunk,
                                         self._emit, self.epoch)
        self.ingress = Ingress(self.fsm, vad, asr, on_transcript=self._on_transcript,
                               on_error=lambda c, m: self._emit({"t": "error", "code": c, "message": m}),
                               clock=clock, on_speech_end=lambda: self._m_mark("eot_wait"))
        self._text = ""
        self._task: asyncio.Task | None = None
        self._cancels: set[asyncio.Task] = set()
        self._gen_done = False
        self._cur: dict | None = None
        self.spontaneous = Spontaneous(self.cfg, record=self._bc_record)  # 既定オフ (設計書 6.1 / 8.1)
        self._spont_task: asyncio.Task | None = None
        self._last_op = clock()  # 最後の利用者操作時刻 (IDLE_TIMEOUT 判定)

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

    def _send_state(self, ctx: Ctx) -> None:
        self._emit({"t": "state", "state": ctx.state.value, "muted": ctx.muted})
        if ctx.state is State.INTERRUPT_CHECK:
            self._ic_start_ms = int(self.clock() * 1000)
            try:  # T_confirm 内に語が得られなければ判断不能として cancel (行11)
                asyncio.get_running_loop().call_later(self.cfg.T_confirm / 1000, self.check_confirm_timeout)
            except RuntimeError:
                pass

    def check_confirm_timeout(self) -> bool:
        """INTERRUPT_CHECK が T_confirm を超えて未判定なら JUDGED(unknown) を送る。送ったら True。"""
        if self.fsm.ctx.state is not State.INTERRUPT_CHECK:
            return False
        elapsed = int(self.clock() * 1000) - self._ic_start_ms
        if elapsed < self.cfg.T_confirm:
            return False
        self.fsm.enqueue(Event(Ev.JUDGED, verdict="unknown", elapsed_ms=elapsed))
        self._pump()
        return True

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
    def _on_up_audio(self, seq: int, pcm: bytes) -> None:
        if not self.core.hello_done:
            return
        for i in range(0, len(pcm), FRAME_BYTES):
            before, was_speaking = self.fsm.ctx.state, self.ingress._buf is not None
            self.ingress.feed(pcm[i:i + FRAME_BYTES])
            if not was_speaking and self.ingress._buf is not None:
                self._speech_in_ai = before is State.AI_PLAYING
            if self.backchannel is not None:
                self._bc_track(1)

    def _on_ui(self, settings: dict) -> None:
        try:
            self.spontaneous.apply_settings(settings)
        except ValueError as e:
            self._emit({"t": "error", "code": ERR_OUT_OF_RANGE, "message": str(e)[:200]})
        self._last_op = self.clock()
        muted = bool(settings.get("mute"))
        if muted and not self.fsm.ctx.muted:
            self.fsm.enqueue(Event(Ev.MUTE))
        elif not muted and self.fsm.ctx.muted:
            self.fsm.enqueue(Event(Ev.WAKE))
        self._pump()

    def _on_transcript(self, uid: int, text: str) -> None:
        self._text = text
        self._record_echo_chain(uid, text)
        self._m_mark("asr_done")
        if self.fsm.ctx.state is State.INTERRUPT_CHECK and self._judge_interrupt(text) == "backchannel":
            self._m_abandon()
            return  # 相槌は履歴に積まない
        self.store.add_turn(self.now_ms(), "user", text, 0, None, uid)  # user 行は completed=0 (ai の再生完了のみ 1)

    def _record_echo_chain(self, uid: int, text: str) -> None:
        """AI 再生中に始まった発話の認識結果が直近 AI 発話の本文に含まれる = 自声の認識 (§6.2 / §12.3 E10)。

        設計書に検出則の定数は無いため、句読点・空白を除いた部分文字列一致のみ (閾値なし)。1 件 = 1 連鎖。
        """
        in_ai, self._speech_in_ai = self._speech_in_ai, False
        if self.metrics is None:
            return
        norm = "".join(ch for ch in text if ch.isalnum())
        if in_ai and norm and norm in "".join(ch for ch in self._ai_text if ch.isalnum()):
            self.store.add_event(self.now_ms(), self.epoch.current, ECHO_CHAIN_KIND,
                                 json.dumps({"utterance_id": uid, "text": text}, ensure_ascii=False))

    def _on_playback(self, msg: dict) -> None:
        self.history.on_playback(msg)
        if self.metrics is not None and msg["utterance_id"] not in self._bc_utts:  # 相槌クリップは turns に載らないので G3/履歴照合から除外
            self.store.add_event(self.now_ms(), self.epoch.current, "playback", json.dumps(
                {k: msg[k] for k in ("epoch", "utterance_id", "chunk_id", "state", "position_ms")}))
        c = self._cur
        if c and msg["state"] == "started" and msg["epoch"] == c["epoch"] and msg["utterance_id"] == c["utt"]                 and msg["chunk_id"] == 0:
            self._m_mark("play_start")
        if c and msg["epoch"] == c["epoch"] and msg["utterance_id"] == c["utt"] and msg["state"] == "finished":
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
            if self.core.conn is not None:
                self.core.conn.send_bytes(pack_down_chunk(ep, utt, chunk, wav_to_pcm(wav)))
            return
        if self._cur is None or self._cur["utt"] != utt:
            self._cur = {"epoch": ep, "utt": utt, "chunks": set(), "finished": set()}
        self._cur["chunks"].add(chunk)
        self._ai_text = text if chunk == 0 else self._ai_text + text
        if chunk == 0:
            self._m_mark("tts_first")
        self.history.register_chunk(ep, utt, chunk, text)
        self._emit({"t": "text", "epoch": ep, "utterance_id": utt, "text": text})
        if self.core.conn is not None:
            self.core.conn.send_bytes(pack_down_chunk(ep, utt, chunk, wav_to_pcm(wav)))
        if chunk == 0:
            self.fsm.enqueue(Event(Ev.FIRST_CLAUSE, epoch=ep))
            self._pump()

    def _check_done(self) -> None:
        c = self._cur
        if self.spontaneous.pending and self._gen_done and self.fsm.ctx.state is State.IDLE:
            if c is None or (c["chunks"] and c["finished"] >= c["chunks"]):  # 自発発話の再生完了
                self._finalize_utterance("spontaneous", interrupted=False)
                self.spontaneous.done()
            return
        if c and self._gen_done and c["chunks"] and c["finished"] >= c["chunks"]:
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
        """利用者発話開始: 準備中/再生中の自発発話を取消す。反応があったので待ち時間は基本に戻る。"""
        self._last_op = self.clock()
        if self.spontaneous.pending:
            self._cancel_spontaneous()
        self.spontaneous.on_user_speech()

    def _act_stop_background(self, a: Action) -> None:
        if self.spontaneous.pending:
            self._cancel_spontaneous()

    def _cancel_spontaneous(self) -> None:
        if self._spont_task and not self._spont_task.done():
            self._spont_task.cancel()
        self.spontaneous.cancel()
        self._cancel_path("spontaneous")

    def _act_commit_history(self, a: Action) -> None:
        self._last_op = self.clock()
        self._finalize_utterance("response", interrupted=False)

    def _act_duck(self, a: Action) -> None:
        self._emit({"t": "duck", "epoch": self.epoch.current, "gain": self.cfg.DUCK_GAIN})

    def _act_resume(self, a: Action) -> None:
        self._emit({"t": "resume", "epoch": self.epoch.current})

    # ---- 自発発話 / 無操作休止 (設計書 4.2 / 6.1) ----
    def spontaneous_tick(self, now: float | None = None) -> str | None:
        """定期呼び出し。'rest' (IDLE_TIMEOUT 遷移) / 'spontaneous' (発火) / None を返す。"""
        now = self.clock() if now is None else now
        idle = self.fsm.ctx.state is State.IDLE
        if idle and now - self._last_op >= self.cfg.IDLE_TIMEOUT:
            self.fsm.enqueue(Event(Ev.IDLE_TIMEOUT))
            self._pump()
            return "rest"
        if self.spontaneous.should_fire(now, idle):
            self.spontaneous.fire(now)
            self._gen_done = False
            self._spont_task = asyncio.get_running_loop().create_task(self._spontaneous_respond())
            return "spontaneous"
        return None

    async def _spontaneous_respond(self) -> None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": "assistant" if t["role"] == "ai" else "user", "content": t["text"]}
            for t in history_window(self.store, self.cfg) if t["text"]]
        messages.append({"role": "user", "content": "(利用者は黙っています。短く自然に話しかけてください)"})
        self.path_calls.append(("spontaneous", "play_utterance"))  # 通常応答と同じ play_utterance を通る
        me = asyncio.current_task()
        await self.pipeline.respond(messages)
        if self._spont_task is not me or not self.spontaneous.pending:
            return
        self._gen_done = True
        self._check_done()

    # ---- 唯一の再生/キャンセル/確定経路 (不変条件 3) ----
    async def _respond(self) -> None:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": "assistant" if t["role"] == "ai" else "user", "content": t["text"]}
            for t in history_window(self.store, self.cfg) if t["text"]]
        self.path_calls.append(("response", "play_utterance"))
        me = asyncio.current_task()
        ep = self.epoch.current
        try:
            await self.pipeline.respond(messages)
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
