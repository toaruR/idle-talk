"""会話ログ (data/idle_talk.db の turns) から「利用者の発話 → AI の返答」の組を取り出す。

出力: data/dialect/pairs.jsonl  1 行 {"id", "user", "ai", "prev_id"}
prev_id は同じ会話の直前の組 (返答の前の前が AI の返答で、その前が利用者の発話) の id。無ければ null。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "dialect" / "pairs.jsonl"
AI_MIN, AI_MAX = 30, 220  # 返答の長さ (字)
USER_MIN = 3  # 「キュッ」等の物音の聞き違いを除く
GAP_MS = 120_000  # 発話から返答までがこれ以上空いたら組にしない
SPONT_USER = "(利用者は黙っています。何か話題を見つけて話しかけてください)"
SPONT_SHARE = 0.4  # 自発発話の割合の上限
NOISE = re.compile(r"^[ぁ-ゖァ-ヺー・。、！？!?\s]{1,4}$")  # 短いかなだけの発話


def norm(s: str) -> str:
    return re.sub(r"[\s。、！？!?…「」]", "", s)


def extract(db: Path, limit: int, seed: int) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("select id, ts, role, text, completed from turns "
                       "where role in ('user', 'ai') order by id").fetchall()
    pairs, spont, seen = [], [], set()
    last_pair_by_index = {}
    for i in range(1, len(rows)):
        aid, ats, arole, atext, done = rows[i]
        uid, uts, urole, utext, _ = rows[i - 1]
        if arole != "ai" or not done or not atext:
            continue
        if urole != "user":  # 自発発話 (話題は保存されていないので、学習では汎用の指示を付ける)
            t = atext.strip()
            if AI_MIN <= len(t) <= AI_MAX and "http" not in t and norm(t)[:40] not in seen:
                seen.add(norm(t)[:40])
                spont.append({"id": aid, "user": SPONT_USER, "ai": t, "prev_id": None, "kind": "spont"})
            continue
        if not utext:
            continue
        utext, atext = utext.strip(), atext.strip()
        if not AI_MIN <= len(atext) <= AI_MAX or len(norm(utext)) < USER_MIN or NOISE.match(utext):
            continue
        if ats - uts > GAP_MS or "http" in atext:
            continue
        key = norm(atext)[:40]
        if key in seen:
            continue
        seen.add(key)
        prev = last_pair_by_index.get(i - 2)
        pair = {"id": aid, "user": utext, "ai": atext, "prev_id": prev, "kind": "reply"}
        last_pair_by_index[i] = aid
        pairs.append(pair)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    rng.shuffle(spont)
    n_spont = min(len(spont), max(0, limit - len(pairs)), int(limit * SPONT_SHARE))
    picked = pairs[:limit - n_spont] + spont[:n_spont]
    keep = {p["id"] for p in picked}
    for p in picked:  # 直前の組が選ばれていなければ履歴に使えない
        if p["prev_id"] not in keep:
            p["prev_id"] = None
    picked.sort(key=lambda p: p["id"])
    return picked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "idle_talk.db")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    pairs = extract(a.db, a.limit, a.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_spont = sum(p["kind"] == "spont" for p in pairs)
    n_hist = sum(p["prev_id"] is not None for p in pairs)
    print(f"{len(pairs)} pairs -> {OUT} (reply {len(pairs) - n_spont}, spont {n_spont}, with history {n_hist})")


if __name__ == "__main__":
    main()
