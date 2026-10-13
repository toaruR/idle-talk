"""話題カード (設計書 3.4) と質問率制御 RATIO_Q (§8.1: 3 応答に 1 回以下)。"""
from __future__ import annotations

import json
from collections import deque
from typing import Any

from .store import Store

RATIO_Q_WINDOW = 3  # 3 応答に 1 回以下


def add_card(store: Store, hook: str | None, facts: Any = None, source_url: str | None = None,
             published_at: int | None = None, expires_at: int | None = None,
             angle: str | None = None) -> int:
    if not hook or not str(hook).strip():
        raise ValueError("hook is required")
    facts_s = facts if (facts is None or isinstance(facts, str)) else json.dumps(facts, ensure_ascii=False)
    return store.execute(
        "INSERT INTO topic_cards(hook, facts, source_url, published_at, expires_at, angle) VALUES (?,?,?,?,?,?)",
        (hook, facts_s, source_url, published_at, expires_at, angle),
    )


def has_card(store: Store, source_url: str | None, hook: str) -> bool:
    """同じ記事のカードが保存済みか (URL があれば URL、無ければ hook で照合)。再起動をまたいだ重複保存を防ぐ。"""
    if source_url:
        return bool(store.query("SELECT 1 FROM topic_cards WHERE source_url=? LIMIT 1", (source_url,)))
    return bool(store.query("SELECT 1 FROM topic_cards WHERE hook=? LIMIT 1", (hook,)))


def select_cards(store: Store, now_ms: int, limit: int = 1) -> list:
    """未使用かつ未期限切れのカード (expires_at NULL は無期限)。"""
    return store.query(
        "SELECT * FROM topic_cards WHERE used_at IS NULL AND (expires_at IS NULL OR expires_at >= ?) "
        "ORDER BY COALESCE(published_at, 0) DESC, id ASC LIMIT ?",
        (now_ms, limit),
    )


def mark_used(store: Store, card_id: int, now_ms: int) -> None:
    store.execute("UPDATE topic_cards SET used_at=? WHERE id=?", (now_ms, card_id))


def set_reaction(store: Store, card_id: int, reaction: str) -> None:
    store.execute("UPDATE topic_cards SET user_reaction=? WHERE id=?", (reaction, card_id))


class QuestionRatio:
    """直近 RATIO_Q_WINDOW 応答中の質問数を 1 以下に保つ。"""

    def __init__(self, window: int = RATIO_Q_WINDOW) -> None:
        self.window = window
        self._recent: deque[bool] = deque(maxlen=window - 1)

    def can_ask(self) -> bool:
        return not any(self._recent)

    def record(self, was_question: bool) -> None:
        self._recent.append(bool(was_question))

    def decide(self, wants_question: bool) -> bool:
        """質問したい場合に許可されれば True。応答を 1 件記録する。"""
        q = bool(wants_question) and self.can_ask()
        self.record(q)
        return q
