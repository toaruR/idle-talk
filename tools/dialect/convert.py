"""pairs.jsonl の AI の返答を、Claude (claude -p) で村山弁に書き換える。再開可能 (済んだ id は飛ばす)。

出力: data/dialect/converted.jsonl  1 行 {"id", "user", "ai", "dialect", "prev_id"}
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "dialect"
RULES = Path(__file__).with_name("murayama_rules.md")
BAD = re.compile(r"やん|ほんま|へん[。！？]|けん[、。]|っちゃん|ばい[。！]|じゃけ|うち[はがもの]")

PROMPT = """あなたは山形県山形市出身の方言話者です。ツールやコマンドは使わず、テキストだけで答えてください。
次の規則に従い、各「返答」を村山弁に書き換えてください。利用者の発話は文脈で、書き換えません。

{rules}

話し手は山形市生まれ・在住の人間の女性です。元の返答にある別の人物設定 (宇宙人・「わたしの星」・200歳・博多や関西の出身など) は、
山形での暮らし・地元の話に自然に置き換えてください (例：「わたしの星には無い」→「山形ではあんまり見ねぇ」)。それ以外の意味は変えません。

出力は JSON 配列だけ (前置き・コードブロック不要): [{{"id": 数値, "dialect": "書き換えた返答"}}, ...]
入力:
{items}
"""


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ask(prompt: str, model: str, timeout: int) -> str:
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text"],
                       input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300] or f"exit {r.returncode}")
    return r.stdout


def parse(out: str) -> dict[int, str]:
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        raise ValueError("no json array")
    return {int(o["id"]): str(o["dialect"]).strip() for o in json.loads(m.group(0))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="今回書き換える最大件数 (0 で全部)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--redo", action="store_true",
                    help="別の人物設定 (OTHER_PERSONA) が残った行を書き換え直す")
    a = ap.parse_args()
    pairs = load(DIR / "pairs.jsonl")
    out_path = DIR / "converted.jsonl"
    if a.redo:
        from build_dataset import OTHER_PERSONA
        rows = load(out_path)
        keep = [r for r in rows if not OTHER_PERSONA.search(r["dialect"])]
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8")
        print(f"redo {len(rows) - len(keep)} rows", flush=True)
    done = {c["id"] for c in load(out_path)}
    todo = [p for p in pairs if p["id"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    rules = RULES.read_text(encoding="utf-8")
    print(f"done {len(done)}, todo {len(todo)}", flush=True)
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        items = "\n".join(json.dumps({"id": p["id"], "利用者": p["user"], "返答": p["ai"]}, ensure_ascii=False)
                          for p in chunk)
        try:
            got = parse(ask(PROMPT.format(rules=rules, items=items), a.model, a.timeout))
        except (RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            print(f"batch {s}: failed: {e}", file=sys.stderr, flush=True)
            continue
        with out_path.open("a", encoding="utf-8") as f:
            for p in chunk:
                d = got.get(p["id"])
                if not d:
                    continue
                if BAD.search(d):
                    print(f"  {p['id']}: 他の方言が残った: {d}", file=sys.stderr)
                f.write(json.dumps({**p, "dialect": d}, ensure_ascii=False) + "\n")
        print(f"batch {s}: {len(got)}/{len(chunk)}", flush=True)


if __name__ == "__main__":
    main()
