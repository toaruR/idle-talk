"""記憶 (設計書 3.4 / §7)。根拠発話なしの保存は拒否、deletable=1 のみ個別削除可。

会話の利用者発話から、本人が明言した好み・習慣・続いている話題だけを LLM で抜き出して少量保存し、
応答のシステムプロンプトに添える (最終アーキテクチャ §5「記憶」)。ASR の誤認識で確定させないよう、
短い発話は対象外・1 発話あたりの件数と 1 件の長さに上限を設け、利用者が ctl から確認・削除できる。
"""
from __future__ import annotations

import re

from .store import Store

MAX_MEMORIES = 30  # 保存上限。超えたら古い deletable 行から消す
MAX_PER_UTTERANCE = 3
MAX_CHARS = 60  # 1 件の長さ上限 (超えたら捨てる: 要約に失敗した長文はプロンプトを圧迫する)
MIN_UTTERANCE_CHARS = 6  # これより短い発話 (相づち・誤認識の断片) からは抜き出さない
SUBJECT = "利用者"  # 抽出結果はこの語で始まる行だけ採る
_BULLET = re.compile(r"^(?:[-・*●]|\d+[.)、．])\s*")

EXTRACT_PROMPT = ("あなたは会話記録から利用者についての事実を抜き出す係です。"
                  "利用者の発言のうち、本人が自分について明言した好み・習慣・予定・続いている出来事だけを、"
                  "「利用者は〜」で始まる短い 1 文にして 1 行に 1 つずつ書いてください (最大 3 行)。"
                  "推測・一般論・質問・AI の発言の内容は書かないでください。該当が無ければ「なし」とだけ書いてください。")


class MemoryError_(Exception):
    pass


def add_memory(store: Store, content: str, evidence_utterance_id: int | None,
               updated_at: int, deletable: int = 1) -> int:
    if evidence_utterance_id is None:
        raise ValueError("evidence_utterance_id is required")
    return store.execute(
        "INSERT INTO memories(content, evidence_utterance_id, updated_at, deletable) VALUES (?,?,?,?)",
        (content, evidence_utterance_id, updated_at, 1 if deletable else 0),
    )


def list_memories(store: Store) -> list:
    return store.query("SELECT * FROM memories ORDER BY id")


def delete_memory(store: Store, memory_id: int) -> None:
    """deletable=1 のみ削除。deletable=0 や不在は例外。"""
    rows = store.query("SELECT deletable FROM memories WHERE id=?", (memory_id,))
    if not rows:
        raise KeyError(memory_id)
    if not rows[0][0]:
        raise PermissionError(f"memory {memory_id} is not deletable")
    store.execute("DELETE FROM memories WHERE id=? AND deletable=1", (memory_id,))


def delete_all_memories(store: Store) -> int:
    """§7 利用者操作の全記憶削除 API。削除件数を返す。"""
    n = store.count("memories")
    store.execute("DELETE FROM memories")
    return n


# ---- 抽出・保存・プロンプト ----
def extract_messages(utterances: list[str]) -> list[dict]:
    """抽出用の LLM 入力。utterances は利用者発話 (古い順)。"""
    body = "\n".join(f"利用者: {u}" for u in utterances)
    return [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": body}]


def parse_extracted(text: str) -> list[str]:
    """LLM 出力を記憶候補に整える。箇条記号を外し、「利用者」で始まらない行・長すぎる行を捨てる。"""
    out: list[str] = []
    for line in text.splitlines():
        s = _BULLET.sub("", line.strip()).strip()
        if not s.startswith(SUBJECT) or len(s) > MAX_CHARS or s in out:  # 指示の形式外 (なし・雑談) は捨てる
            continue
        out.append(s)
    return out[:MAX_PER_UTTERANCE]


def _norm(s: str) -> str:
    return "".join(ch for ch in s if ch.isalnum())


def remember(store: Store, items: list[str], evidence_utterance_id: int, now_ms: int) -> list[int]:
    """候補を保存する。同じ内容 (句読点・空白を除いて一致) は更新時刻だけ進める。上限超過分は古い順に消す。"""
    existing = {_norm(r["content"]): r["id"] for r in list_memories(store)}
    ids: list[int] = []
    for item in items:
        key = _norm(item)
        if not key:
            continue
        if key in existing:
            store.execute("UPDATE memories SET updated_at=?, evidence_utterance_id=? WHERE id=?",
                          (now_ms, evidence_utterance_id, existing[key]))
            continue
        existing[key] = add_memory(store, item, evidence_utterance_id, now_ms)
        ids.append(existing[key])
    over = store.count("memories") - MAX_MEMORIES
    if over > 0:
        store.execute("DELETE FROM memories WHERE id IN (SELECT id FROM memories WHERE deletable=1 "
                      "ORDER BY updated_at, id LIMIT ?)", (over,))
    return ids


def memory_prompt(store: Store) -> str:
    """システムプロンプトに添える記憶の節。記憶が無ければ空文字。"""
    rows = store.query("SELECT content FROM memories ORDER BY updated_at DESC, id DESC LIMIT ?", (MAX_MEMORIES,))
    if not rows:
        return ""
    lines = "\n".join(f"- {r['content']}" for r in rows)
    return ("\n\n以下は以前の会話で利用者が自分で話した内容です。話題に自然に関係するときだけ触れ、無理に持ち出さないでください。"
            "ここに無いことを覚えているふりはしないでください。\n" + lines)
