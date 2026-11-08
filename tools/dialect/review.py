"""書き換えた返答を目で確かめて直す。

  python tools/dialect/review.py export [--n 200]  -> data/dialect/review.tsv (id, ok, 元, 村山弁)
  (Excel 等で開き、村山弁の列を直す。使わない行は ok を x にする)
  python tools/dialect/review.py apply [--tsv F]   -> converted.jsonl に反映 (x は "drop": true)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

DIR = Path(__file__).resolve().parents[2] / "data" / "dialect"
CONV, TSV = DIR / "converted.jsonl", DIR / "review.tsv"


def load() -> list[dict]:
    return [json.loads(l) for l in CONV.read_text(encoding="utf-8").splitlines() if l.strip()]


def export(n: int, tsv: Path) -> None:
    rows = load()
    pick = random.Random(1).sample(rows, min(n, len(rows)))
    with tsv.open("w", encoding="utf-8-sig", newline="") as f:  # BOM 付きで Excel でも化けない
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "ok", "user", "original", "dialect"])
        for r in sorted(pick, key=lambda r: r["id"]):
            w.writerow([r["id"], "", r["user"], r["ai"], r["dialect"]])
    print(f"{len(pick)} rows -> {tsv}")


def apply(tsv: Path) -> None:
    with tsv.open(encoding="utf-8-sig", newline="") as f:
        edits = {int(r["id"]): r for r in csv.DictReader(f, delimiter="\t")}
    rows, changed, dropped = load(), 0, 0
    for r in rows:
        e = edits.get(r["id"])
        if not e:
            continue
        r["reviewed"] = True
        if e["ok"].strip().lower() == "x":
            r["drop"], dropped = True, dropped + 1
        elif e["dialect"].strip() and e["dialect"].strip() != r["dialect"]:
            r["dialect"], changed = e["dialect"].strip(), changed + 1
    CONV.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"reviewed {len(edits)}, changed {changed}, dropped {dropped}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "apply"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tsv", type=Path, default=TSV)
    a = ap.parse_args()
    export(a.n, a.tsv) if a.cmd == "export" else apply(a.tsv)


if __name__ == "__main__":
    main()
