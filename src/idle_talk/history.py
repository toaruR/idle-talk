"""再生済み履歴の確定 (設計書 5.2, 不変条件 4)。"""
from __future__ import annotations

import logging

from .config import Config
from .store import Store

log = logging.getLogger("idle_talk.history")
talk_log = logging.getLogger("idle_talk.talk")  # 発話文言の記録 (main が data/talk.log に出す)
TALK_LOG_FILE = "talk.log"


class PlaybackHistory:
    def __init__(self, store: Store, cfg: Config | None = None) -> None:
        self.store = store
        self.cfg = cfg or Config()
        self._text: dict[tuple[int, int], dict[int, str]] = {}
        self._state: dict[tuple[int, int], dict[int, str]] = {}
        self._pos: dict[tuple[int, int], dict[int, int]] = {}

    def register_chunk(self, epoch: int, utterance_id: int, chunk_id: int, text: str) -> None:
        self._text.setdefault((epoch, utterance_id), {})[chunk_id] = text

    def on_playback(self, msg: dict) -> None:
        key = (msg["epoch"], msg["utterance_id"])
        cid = msg["chunk_id"]
        self._state.setdefault(key, {})[cid] = msg["state"]
        self._pos.setdefault(key, {})[cid] = msg["position_ms"]

    def commit(self, epoch: int, utterance_id: int, ts: int, interrupted: bool = False) -> int | None:
        """finished チャンクの本文のみ確定。interrupted なら completed=0 + stop_position_ms。"""
        key = (epoch, utterance_id)
        texts = self._text.pop(key, {})
        states = self._state.pop(key, {})
        poss = self._pos.pop(key, {})
        body = "".join(texts.get(c, "") for c in sorted(states) if states[c] == "finished")
        if interrupted:
            unfinished = [c for c in states if states[c] != "finished"]
            stop = poss[max(unfinished)] if unfinished else (poss[max(poss)] if poss else 0)
            if body:
                talk_log.info("AI(中断): %s", body)
            return self.store.add_turn(ts, "ai", body, 0, stop, utterance_id)
        if not body:
            return None
        talk_log.info("AI: %s", body)
        return self.store.add_turn(ts, "ai", body, 1, None, utterance_id)


def history_window(store: Store, cfg: Config | None = None) -> list[dict]:
    """直近 N_TURNS ターンを古い順で返す。切り詰め時は log を 1 件残す。"""
    cfg = cfg or Config()
    rows = store.query(
        "SELECT id, ts, role, text, completed, stop_position_ms, utterance_id "
        "FROM turns WHERE role != 'transcript' ORDER BY id DESC LIMIT ?", (cfg.N_TURNS + 1,))  # 文字起こしは会話ではない
    if len(rows) > cfg.N_TURNS:
        log.info("history truncated to last %d turns", cfg.N_TURNS)
        rows = rows[: cfg.N_TURNS]
    return [dict(r) for r in reversed(rows)]
