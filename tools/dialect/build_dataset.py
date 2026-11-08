"""converted.jsonl から学習データ (chat 形式) を作る。system は personas.toml の yamagata と同じ組み立て。

出力: data/dialect/train.jsonl / valid.jsonl  1 行 {"messages": [{"role", "content"}, ...]}
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from idle_talk.persona import PersonaBook  # noqa: E402

DIR = ROOT / "data" / "dialect"
# 元の返答は他のペルソナ (博多弁の宇宙人など) の設定を含む。山形の人物と食い違うので学習に入れない
OTHER_PERSONA = re.compile(r"星|宇宙|地球人|200歳|二百歳|宇宙人|UFO|博多|関西|大阪")
# 第 2 版 (2026-11-08)：利用者の名前を覚えて system に無いときも呼んだので「あんた」に置き換える
USER_NAMES = ("リキ",)
NAME_RE = re.compile("(?:" + "|".join(USER_NAMES) + ")(?:さん|ちゃん|くん|君)?")
MAX_SENTENCES = 4  # 第 1 版は 5〜6 文の返答が出た


def sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[。！？!?])", t) if s.strip()])


def v2_filter(rows: list[dict]) -> list[dict]:
    """話題の無い自発発話・長い返答・名前そのものの話を外し、名前を「あんた」にする。"""
    out, why = [], {"spont": 0, "long": 0, "name_talk": 0}
    for r in rows:
        if r.get("kind") == "spont":  # 話題 (台詞・ニュース) が保存されておらず、何も無い所に一言を返す癖がつく
            why["spont"] += 1
        elif sentences(r["dialect"]) > MAX_SENTENCES:
            why["long"] += 1
        elif NAME_RE.search(r["user"]) or (NAME_RE.search(r["dialect"]) and "名前" in r["dialect"]):
            why["name_talk"] += 1  # 名乗り・名前の話は置き換えると意味が通らない
        else:
            out.append({**r, "dialect": NAME_RE.sub("あんた", r["dialect"])})
    print(f"v2 dropped: {why}")
    return out


def system_prompt(pid: str) -> str:
    book = PersonaBook(ROOT / "personas.toml")
    book.select(pid)
    return book.system_prompt()


def build(rows: list[dict], system: str) -> list[dict]:
    by_id = {r["id"]: r for r in rows}
    out = []
    for r in rows:
        msgs = [{"role": "system", "content": system}]
        prev = by_id.get(r.get("prev_id"))
        if prev:  # 直前の組も村山弁にしてあれば履歴として入れる (履歴の口調にも合わせて話すように)
            msgs += [{"role": "user", "content": prev["user"]}, {"role": "assistant", "content": prev["dialect"]}]
        msgs += [{"role": "user", "content": r["user"]}, {"role": "assistant", "content": r["dialect"]}]
        out.append({"messages": msgs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", default="yamagata")
    ap.add_argument("--valid", type=float, default=0.05)
    a = ap.parse_args()
    rows = [json.loads(l) for l in (DIR / "converted.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if not r.get("drop") and r.get("dialect")]
    n_all = len(rows)
    rows = [r for r in rows if not OTHER_PERSONA.search(r["dialect"])]
    print(f"other persona dropped: {n_all - len(rows)}")
    rows = v2_filter(rows)
    data = build(rows, system_prompt(a.persona))
    random.Random(3).shuffle(data)
    n_valid = max(1, int(len(data) * a.valid))
    for name, part in (("valid", data[:n_valid]), ("train", data[n_valid:])):
        (DIR / f"{name}.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in part),
                                           encoding="utf-8")
    print(f"train {len(data) - n_valid}, valid {n_valid} -> {DIR}")


if __name__ == "__main__":
    main()
