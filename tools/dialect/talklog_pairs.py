"""data/talk.log から「履歴 + 利用者の発話 → AI の返答」を取り出す (第 3 版の学習データの元)。

第 2 版 (turns から) は返答の 4 割が質問で終わり 9 割が 3〜4 文で、LoRA が質問癖と長さを強めた。
第 3 版は返答を書き直す (talklog_convert.py) ので、ここでは場面 (履歴・発話) と、推論と同じ system の材料を決める。

出力: data/dialect/v3/pairs.jsonl  1 行
  {"id", "ts", "persona", "context": [{"role", "content"}], "user", "ai", "long": bool, "may_ask": bool}
  long     … talk_length.chat_note が長め (3 文ほど) を選ぶ発話か
  may_ask  … 直近の AI 発話に質問終わりが無く、質問で締めてもよいか (director._question_note と同じ判定)
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from idle_talk.director import CARD_FACTS_MAX, CARD_PROMPT, CARD_TURNS  # noqa: E402
from idle_talk.talk_length import CHAT_LONG_NOTE, chat_note  # noqa: E402
from idle_talk.topics import QuestionRatio  # noqa: E402
from idle_talk.voice_id import ASK_QUESTION  # noqa: E402
from idle_talk.web_search import X_TREND  # noqa: E402

OUT = ROOT / "data" / "dialect" / "v3" / "pairs.jsonl"
EXTRA = OUT.with_name("extra.jsonl")  # 追加分 (話題カードの自発発話・ペルソナを入れ替えた複製)。pairs.jsonl の id は変えない
CARD_WINDOW_S = 120   # カードを使ってから自発発話が talk.log に書かれる (再生完了) までの上限
COPY_ID = 1_000_000   # ペルソナを入れ替えた複製の id = 元の id + COPY_ID
LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) ([A-Z?]+)(\([^)]*\))?:\s?(.*)$")
SESSION_GAP_S = 600   # これ以上空いたら別の会話 (履歴を切る)
REPLY_GAP_S = 120     # 発話から返答までがこれ以上空いたら組にしない
CONTEXT_TURNS = 4     # 履歴に入れる直前のターン数 (推論は N_TURNS=8。system + 検索の指示で約 700 トークンあり max_len に収める)
AI_MIN, AI_MAX = 8, 260
USER_MIN = 2
# 他のモードの行。ドラマ・ブラウザ補正なしの会話は system が普段と違うので使わない (次の会話の切れ目まで)
OTHER_MODE = {"DRAMA", "PC"}
# 文字起こし (AI は話さない) と ChatGPT への引き渡しの行。履歴だけ切る (この後の AI の返答はモードを抜けた後)
RESET_ONLY = {"TEXT", "TRANSCRIPT", "SPEAKER", "HANDOFF", "HANDOFF?"}
# 返答に混ざる定型文・検索などの別経路。書き直せないので組にしない
SKIP_AI = re.compile(r"調べてみる|もう少し待って|画面に表示する|読んでみる|ChatGPT で要約|見ているページが分からな|"
                     r"なんだっけ…|ぼーっとしてだ|って感じだべ。$|ほだな感じだず。$|http|検索[:：]|リンク[:：]")
PERSONAS = (("yamagata", 0.7), ("uzu", 0.3))  # personas.toml で lora = true のペルソナ


def parse(path: Path) -> list[dict]:
    rows: list[dict] = []
    for raw in path.read_bytes().decode("utf-8", errors="replace").splitlines():
        m = LINE.match(raw)
        if m:
            ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
            rows.append({"ts": ts, "role": m.group(2), "cut": m.group(3) == "(中断)", "text": m.group(4).strip()})
        elif rows and raw.strip():  # 複数行の発話の続き
            rows[-1]["text"] += "\n" + raw.strip()
    return rows


def norm(s: str) -> str:
    return re.sub(r"[\s。、！？!?…「」]", "", s)


def load_cards(db: Path) -> list[tuple[float, str]]:
    """使用済みの話題カード (使った時刻 秒, director._pick_topic と同じ話題の文)。古い順。"""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = []
    for hook, facts, angle, used in con.execute(
            "select hook, facts, angle, used_at from topic_cards where used_at is not null order by used_at"):
        facts = re.sub(r"<[^>]+>", "", facts or "").strip()[:CARD_FACTS_MAX]
        label = "X で話題になっている「{}」" if angle == X_TREND else "最近見かけたニュース「{}」"
        out.append((used / 1000, label.format(hook) + (f"(概要: {facts})" if facts else "")))
    return out


def card_for(cards: list[tuple[float, str]], ts: float) -> str | None:
    """ts の直前 CARD_WINDOW_S 秒以内に使ったカードの話題 (一番近いもの)。"""
    i = bisect.bisect_right(cards, (ts, "￿")) - 1
    if i >= 0 and ts - cards[i][0] <= CARD_WINDOW_S:
        return cards[i][1]
    return None


def extract(rows: list[dict], cards: list[tuple[float, str]] | None = None) -> list[dict]:
    """cards を渡すと、利用者の発話への返答ではなく、話題カードで話しかけた自発発話を取り出す (kind "card")。"""
    out, seen, used = [], set(), set()
    ctx: list[dict] = []   # 今の会話の履歴 (推論の history と同じく中断した AI 発話も入る)
    tainted = False        # 他のモードの行を見た会話
    last_ts = 0.0
    for i, r in enumerate(rows):
        if r["ts"] - last_ts > SESSION_GAP_S:
            ctx, tainted = [], False
        last_ts = r["ts"]
        if r["role"] in OTHER_MODE:
            ctx, tainted = [], True
            continue
        if r["role"] in RESET_ONLY:
            ctx = []
            continue
        if r["role"] not in ("USER", "AI") or not r["text"]:
            continue
        if r["role"] == "USER":
            ctx.append({"role": "user", "content": r["text"], "ts": r["ts"]})
            continue
        text = r["text"]
        prev = ctx[-1] if ctx else None
        if cards is not None:
            reply = prev is not None and prev["role"] == "user" and r["ts"] - prev["ts"] <= REPLY_GAP_S
            topic = None if (tainted or r["cut"] or reply) else card_for(cards, r["ts"])
            if topic and topic not in used and AI_MIN <= len(text) <= AI_MAX and not SKIP_AI.search(text):
                used.add(topic)
                hist = ctx[-CARD_TURNS:]
                out.append({
                    "id": i, "ts": int(r["ts"]), "kind": "card", "topic": topic,
                    "context": [{"role": h["role"], "content": h["content"]} for h in hist],
                    "user": CARD_PROMPT.format(topic=topic), "ai": text, "long": False,
                    "may_ask": QuestionRatio.from_texts([h["content"] for h in hist if h["role"] == "assistant"]).can_ask(),
                })
            ctx.append({"role": "assistant", "content": text, "ts": r["ts"]})
            continue
        target = (not tainted and not r["cut"] and prev is not None and prev["role"] == "user"
                  and r["ts"] - prev["ts"] <= REPLY_GAP_S)
        if target:
            body = text.replace(ASK_QUESTION, "").strip()
            user = prev["content"]
            key = norm(user)[:30] + "|" + norm(body)[:30]
            if (AI_MIN <= len(body) <= AI_MAX and len(norm(user)) >= USER_MIN and not SKIP_AI.search(body)
                    and key not in seen):
                seen.add(key)
                hist = ctx[:-1][-CONTEXT_TURNS:]
                ai_hist = [h["content"] for h in hist if h["role"] == "assistant"]
                out.append({
                    "id": i, "ts": int(r["ts"]),
                    "context": [{"role": h["role"], "content": h["content"]} for h in hist],
                    "user": user, "ai": body,
                    "long": chat_note(user) == CHAT_LONG_NOTE,
                    "may_ask": QuestionRatio.from_texts(ai_hist).can_ask(),
                })
        ctx.append({"role": "assistant", "content": text, "ts": r["ts"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=ROOT / "data" / "talk.log")
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--extra", action="store_true",
                    help="extra.jsonl を作る: 話題カードの自発発話 (--cards 件まで) + pairs.jsonl のペルソナを入れ替えた複製")
    ap.add_argument("--cards", type=int, default=700)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "idle_talk.db")
    a = ap.parse_args()
    converted = OUT.with_name("converted_extra.jsonl" if a.extra else "converted.jsonl")
    if converted.exists() and not a.force:
        # talk.log が伸びると選び方とペルソナの割り振りが変わり、書き直し済みの行と食い違う
        sys.exit(f"{converted} がある。作り直すなら変換済みデータを退避して --force")
    if a.extra:
        return extra(a)
    pairs = extract(parse(a.log))
    rng = random.Random(a.seed)
    rng.shuffle(pairs)
    picked = sorted(pairs[:a.limit], key=lambda p: p["id"])
    names, weights = zip(*PERSONAS)
    for p in picked:
        p["persona"] = rng.choices(names, weights)[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked), encoding="utf-8")
    q = sum(bool(re.search(r"[？?]\s*$", p["ai"])) for p in picked)
    print(f"candidates {len(pairs)}, picked {len(picked)} -> {OUT}")
    print(f"  long {sum(p['long'] for p in picked)}, may_ask {sum(p['may_ask'] for p in picked)}, "
          f"original q-end {q}, with context {sum(bool(p['context']) for p in picked)}")


def extra(a: argparse.Namespace) -> None:
    rng = random.Random(a.seed + 1)
    names, weights = zip(*PERSONAS)
    cards = extract(parse(a.log), load_cards(a.db))
    rng.shuffle(cards)
    cards = sorted(cards[:a.cards], key=lambda p: p["id"])
    for p in cards:
        p["persona"] = rng.choices(names, weights)[0]
    other = {"yamagata": "uzu", "uzu": "yamagata"}
    base = [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]
    copies = [{**p, "id": p["id"] + COPY_ID, "persona": other[p["persona"]], "copy_of": p["id"]} for p in base]
    EXTRA.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in cards + copies), encoding="utf-8")
    print(f"cards {len(cards)}, persona copies {len(copies)} -> {EXTRA}")


if __name__ == "__main__":
    main()
