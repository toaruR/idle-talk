"""第 3 版の学習データを作る。system は推論 (director._chat_messages + pipeline の検索の指示) と同じ組み立て。

  system = ペルソナ + (質問しない注記) + 長さの目安 + 検索の指示 (今日の日付つき)

元: v3/converted.jsonl (talk.log を書き直したもの) と、第 2 版の converted.jsonl のうち
質問で終わらず短い返答 (村山弁の量を保つため)。第 2 版は system に検索の指示・長さの目安が無く、推論と食い違っていた。

出力: data/dialect/v3/train.jsonl / valid.jsonl  1 行 {"messages": [...]}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from idle_talk.director import NO_QUESTION_NOTE  # noqa: E402
from idle_talk.persona import PersonaBook  # noqa: E402
from idle_talk.talk_length import CHAT_LONG_NOTE, chat_note  # noqa: E402
from idle_talk.topics import ends_with_question  # noqa: E402
from idle_talk.web_search import with_search_rule  # noqa: E402

DIR = ROOT / "data" / "dialect"
OUT = DIR / "v3"
SEARCH_SHARE = 0.15  # 「検索:」の例の割合の上限 (多いと雑談でも調べたがる)
V2_SHARE = 0.4       # 第 2 版から足す例の割合の上限


def sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[。！？!?])", t) if s.strip()])


def system(book: PersonaBook, pid: str, user: str, may_ask: bool) -> str:
    book.select(pid)
    return book.system_prompt() + ("" if may_ask else NO_QUESTION_NOTE) + chat_note(user)


def example(book: PersonaBook, pid: str, context: list[dict], user: str, reply: str, may_ask: bool,
            today: dt.date) -> dict:
    msgs = [{"role": "system", "content": system(book, pid, user, may_ask)}] + context + [
        {"role": "user", "content": user}]
    return {"messages": with_search_rule(msgs, today) + [{"role": "assistant", "content": reply}]}


def card_example(book: PersonaBook, r: dict) -> dict:
    """話題カードの自発発話。推論 (director._spontaneous_respond) は長さの目安も検索の指示も付けない。"""
    book.select(r["persona"])
    system = book.system_prompt() + ("" if r["may_ask"] else NO_QUESTION_NOTE)
    return {"messages": [{"role": "system", "content": system}] + r["context"] + [
        {"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["reply"]}]}


def from_v3(book: PersonaBook) -> tuple[list[dict], list[dict], list[dict]]:
    """(雑談, 検索, 話題カード)。各例に場面の group (ペルソナの複製は元と同じ) を付け、train / valid をまたがせない。"""
    talk, search, card = [], [], []
    for name in ("converted.jsonl", "converted_extra.jsonl"):
        path = OUT / name
        if not path.exists():
            continue
        for l in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            reply = r.get("reply")
            if r.get("drop") or not reply:
                continue
            if r.get("kind") == "card":
                ex, bucket = card_example(book, r), card
            else:
                ex = example(book, r["persona"], r["context"], r["user"], reply, r["may_ask"],
                             dt.date.fromtimestamp(r["ts"]))
                bucket = search if reply.startswith("検索:") else talk
            bucket.append({**ex, "group": r.get("copy_of", r["id"])})
    return talk, search, card


def from_v2(book: PersonaBook, rng: random.Random) -> list[dict]:
    """第 2 版から、質問で終わらず長さの目安に収まる返答だけ (履歴なし、日付は第 2 版の期間から)。"""
    path = DIR / "converted.jsonl"
    out = []
    for l in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        d = r.get("dialect")
        if r.get("drop") or not d or r.get("kind") == "spont" or ends_with_question(d) or "リキ" in d:
            continue
        limit = 3 if chat_note(r["user"]) == CHAT_LONG_NOTE else 2
        if sentences(d) > limit or re.search(r"星|宇宙|200歳|博多|関西", d):
            continue
        today = dt.date(2026, 10, 14) + dt.timedelta(days=rng.randrange(40))
        out.append(example(book, "yamagata", [], r["user"], d, rng.random() < 0.5, today))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid", type=float, default=0.05)
    ap.add_argument("--no-v2", action="store_true")
    a = ap.parse_args()
    rng = random.Random(5)
    book = PersonaBook(ROOT / "personas.toml")
    talk, search, card = from_v3(book)
    rng.shuffle(search)
    v2 = [] if a.no_v2 else [{**ex, "group": f"v2-{i}"} for i, ex in enumerate(from_v2(book, rng))]
    rng.shuffle(v2)
    v2 = v2[:int((len(talk) + len(card)) * V2_SHARE / (1 - V2_SHARE))]
    n_search = min(len(search), int((len(talk) + len(card) + len(v2)) * SEARCH_SHARE / (1 - SEARCH_SHARE)))
    data = talk + card + v2 + search[:n_search]
    groups = sorted({d["group"] for d in data}, key=str)
    rng.shuffle(groups)
    valid_groups = set(groups[:max(1, int(len(groups) * a.valid))])
    rng.shuffle(data)
    parts = {"valid": [d for d in data if d["group"] in valid_groups],
             "train": [d for d in data if d["group"] not in valid_groups]}
    for name, part in parts.items():
        (OUT / f"{name}.jsonl").write_text(
            "".join(json.dumps({"messages": d["messages"]}, ensure_ascii=False) + "\n" for d in part),
            encoding="utf-8")
    answers = [d["messages"][-1]["content"] for d in data]
    q = sum(ends_with_question(t) for t in answers)
    print(f"v3 talk {len(talk)}, card {len(card)}, v2 {len(v2)}, search {n_search}/{len(search)}")
    print(f"q-end {q}/{len(answers)}, sentences {dict(sorted({k: sum(sentences(t) == k for t in answers) for k in range(1, 6)}.items()))}")
    print(f"train {len(parts['train'])}, valid {len(parts['valid'])} -> {OUT}")


if __name__ == "__main__":
    main()
