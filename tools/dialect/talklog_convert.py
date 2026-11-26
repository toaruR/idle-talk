"""v3/pairs.jsonl の AI の返答を、Claude (claude -p) で「その場面で返すべき返答」に書き直す。再開可能 (済んだ id は飛ばす)。

元の返答 (Qwen・Gemma が当時のペルソナで答えたもの) の癖を直す:
  - 質問で締めすぎる → may_ask が偽なら質問・問いかけで終えない。真でも質問は自然なときだけ
  - 長すぎる        → 通常 1〜2 文、long (説明・理由を求められた) なら 3 文ほどまで
  - 調べないと答えられない質問に作り話で答える → 「検索: 検索語」の 1 行
  - 物音・聞き違いの発話に話を広げる → 短く聞き返すか流す
  - 人物設定の混ざり (宇宙人・博多弁など) → 指定したペルソナの口調・設定に揃える (村山弁は murayama_rules.md)

出力: data/dialect/v3/converted.jsonl  1 行 {...pairs の行, "reply": str} (使えない場面は "drop": true)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "data" / "dialect" / "v3"
RULES = Path(__file__).with_name("murayama_rules.md")
BAD = re.compile(r"やん|ほんま|へん[。！？]|けん[、。]|っちゃん|ばい[。！]|じゃけ|うち[はがもの]")

PROMPT = """あなたは日本語の音声雑談アプリの学習データを作る編集者です。ツールやコマンドは使わず、テキストだけで答えてください。
各場面について、AI が返すべき返答を 1 つ書いてください。「元の返答」は参考で、内容が良ければ生かし、悪ければ作り直します。

## 話し手 (persona ごと)
{personas}

## 口調 (村山弁の規則。どちらの persona も従う)
{rules}
上の規則の「文の数・質問で終わるかを変えない」は今回は無視し、下の決まりを優先してください。

## 決まり
- 返答は声で読み上げる。箇条書き・記号・括弧・絵文字・URL を使わない。
- 文の数：long が false なら 1〜2 文で短くテンポよく。long が true なら 3 文ほどまで。
- may_ask が false なら、質問・問いかけ (「？」「教えて」「聞かせて」) で終えない。感想・共感・連想で言い切る。
  may_ask が true でも、質問で締めるのは本当に聞きたいときだけ (全体の 3 割ほど)。毎回質問にしない。
- 利用者の発話は音声認識の結果で、聞き違い・途切れ・物音 (「キュッ」「はい」だけ等) がある。
  意味の取れない発話には話を広げず、短く聞き返すか軽く受け流す。履歴から意味が分かるならそれに答える。
- 今日は場面の date の日 (曜日も date のとおり)。株価・為替・天気・最新のニュース・人物や物事の詳しい事実など、調べないと正確に答えられないことを聞かれたとき、
  または「もっと詳しく」等と直前の話の続きを求められたときは、返答の代わりに 1 行で「検索: 検索語」とだけ書く。
  検索語は会話の文脈を補った具体的な語 (例: 検索: 日経平均株価 今日)。日付・曜日はそのまま答えてよい。元の返答の作り話の数字は使わない。
- 履歴の AI 発話は別の人物・口調のことがある。口調は persona に従い、履歴に合わせない。
- 利用者の名前は書かない (呼ぶなら「あんた」)。
- URL・リンク・出典を求められた場面、画面や名前の登録などアプリの開発の話で答えが決められない場面は reply を null にする。
- kind が card の場面では利用者は黙っていて、AI から「話題」のニュースで話しかける。見出しの要点を短く伝え、自分の感想を一言添えて 1〜2 文。
  概要に無い事実を足さない。履歴の情景や言い回し (雨の音など) を繰り返さない。検索はしない。話題が暗い・分かりにくいときも短く触れるだけにする。
- 場面として使えない (発話も履歴も意味が取れない、別のモードの会話など) ときは reply を null にする。

出力は JSON 配列だけ (前置き・コードブロック不要): [{{"id": 数値, "reply": "返答" または null}}, ...]
場面:
{items}
"""


def persona_text() -> str:
    book = tomllib.loads((ROOT / "personas.toml").read_text(encoding="utf-8"))
    out = []
    for pid in ("yamagata", "uzu"):
        p = book[pid]
        extra = " 宇宙人の設定 (わたしの星・200歳) は使わず、山形での暮らしの話にする。" if pid == "yamagata" else ""
        out.append(f"- {pid}：名前「{p['name']}」。{p['prompt'].strip()}{extra}")
    return "\n".join(out)


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def ask(prompt: str, model: str, timeout: int) -> str:
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "text"],
                       input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if r.returncode != 0:
        # settings.json の WSL のパスについての警告が毎回 stderr の頭に出て、本当の理由を隠す
        err = "\n".join(l for l in r.stderr.splitlines() if "network path" not in l).strip()
        raise RuntimeError(f"exit {r.returncode}: {(err or r.stdout.strip())[:300]}")
    return r.stdout


def parse(out: str) -> dict[int, str | None]:
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        raise ValueError("no json array")
    return {int(o["id"]): (str(o["reply"]).strip() if o.get("reply") else None) for o in json.loads(m.group(0))}


WEEKDAYS = "月火水木金土日"


def _date(ts: int) -> str:
    d = dt.date.fromtimestamp(ts)
    return f"{d.year}年{d.month}月{d.day}日 ({WEEKDAYS[d.weekday()]}曜日)"


def item(p: dict) -> str:
    hist = [("利用者" if h["role"] == "user" else "AI") + ": " + h["content"] for h in p["context"]]
    o = {"id": p["id"], "kind": p.get("kind", "reply"), "persona": p["persona"], "date": _date(p["ts"]),
         "long": p["long"], "may_ask": p["may_ask"], "履歴": hist}
    o.update({"話題": p["topic"]} if p.get("kind") == "card" else {"利用者": p["user"]})
    return json.dumps({**o, "元の返答": p["ai"]}, ensure_ascii=False)


def prepare_extra(rows: list[dict], out_path: Path) -> list[dict]:
    """extra.jsonl の行のうち書き直しが要るもの。ペルソナの複製は、元の場面の書き直し結果を「元の返答」にする
    (使えない場面は捨て、「検索:」はペルソナに関係ないのでそのまま書く)。"""
    conv = {c["id"]: c for c in load(DIR / "converted.jsonl")}
    todo, direct = [], []
    for p in rows:
        src = conv.get(p.get("copy_of")) if "copy_of" in p else None
        if "copy_of" not in p:
            todo.append(p)
        elif src is None or src.get("drop"):
            continue
        elif src["reply"].startswith("検索:"):
            direct.append({**p, "reply": src["reply"]})
        else:
            todo.append({**p, "ai": src["reply"]})
    if direct:
        with out_path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in direct)
    return todo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="今回書き直す最大件数 (0 で全部)")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--extra", action="store_true", help="extra.jsonl -> converted_extra.jsonl (pairs の書き直しが済んでから)")
    a = ap.parse_args()
    out_path = DIR / ("converted_extra.jsonl" if a.extra else "converted.jsonl")
    done = {c["id"] for c in load(out_path)}
    pairs = [p for p in load(DIR / ("extra.jsonl" if a.extra else "pairs.jsonl")) if p["id"] not in done]
    todo = prepare_extra(pairs, out_path) if a.extra else pairs
    if a.limit:
        todo = todo[:a.limit]
    head = dict(personas=persona_text(), rules=RULES.read_text(encoding="utf-8"))
    print(f"done {len(done)}, todo {len(todo)}", flush=True)
    for s in range(0, len(todo), a.batch):
        chunk = todo[s:s + a.batch]
        try:
            got = parse(ask(PROMPT.format(**head, items="\n".join(item(p) for p in chunk)), a.model, a.timeout))
        except (RuntimeError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            print(f"batch {s}: failed: {e}", file=sys.stderr, flush=True)
            continue
        with out_path.open("a", encoding="utf-8") as f:
            for p in chunk:
                if p["id"] not in got:
                    continue
                reply = got[p["id"]]
                if reply and BAD.search(reply):
                    print(f"  {p['id']}: 他の方言が残った: {reply}", file=sys.stderr)
                row = {**p, "reply": reply} if reply else {**p, "drop": True}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"batch {s}: {len(got)}/{len(chunk)}", flush=True)


if __name__ == "__main__":
    main()
