"""話題カード (設計書 3.4) と質問率制御 RATIO_Q (§8.1: 3 応答に 1 回以下)。"""
from __future__ import annotations

import json
import re
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


# 雑談の話題はポジティブな記事を先に使う。暗い記事は他が尽きたときだけ (捨てはしない)
POSITIVE_WORDS = ("受賞", "優勝", "金メダル", "快挙", "記録更新", "新記録", "誕生", "開業", "開幕", "オープン",
                  "発売", "公開", "発表会", "発見", "成功", "人気", "話題", "初の", "復活", "見頃", "満開", "紅葉",
                  "祭", "イベント", "グルメ", "スイーツ", "新作", "新商品", "リニューアル", "楽しめ", "笑顔",
                  "かわいい", "癒", "感動", "絶景", "上昇", "最高値", "好調", "回復", "便利", "無料")
NEGATIVE_WORDS = ("死亡", "死去", "死者", "亡くな", "殺", "遺体", "事故", "事件", "逮捕", "容疑", "起訴", "判決",
                  "被害", "詐欺", "強盗", "窃盗", "暴行", "火災", "火事", "地震", "津波", "豪雨", "台風", "災害",
                  "避難", "停電", "戦争", "攻撃", "空爆", "ミサイル", "テロ", "紛争", "負傷", "けが", "重体",
                  "感染", "不祥事", "謝罪", "辞任", "倒産", "破綻", "下落", "急落", "値上げ", "減少", "懸念", "批判",
                  "違反", "不正", "流出", "行方不明", "中止", "延期")


def card_tone(hook: str | None, facts: str | None = None) -> int:
    """見出し・概要のポジティブ度 (正の語の数 − 負の語の数×2)。暗い語は 1 つでも重く見る。"""
    text = f"{hook or ''} {facts or ''}"
    return sum(w in text for w in POSITIVE_WORDS) - 2 * sum(w in text for w in NEGATIVE_WORDS)


def select_cards(store: Store, now_ms: int, limit: int = 1) -> list:
    """未使用かつ未期限切れのカード (expires_at NULL は無期限)。ポジティブ度 (card_tone) の高い順、同じなら新しい順。
    自発発話はこの上位 PICK_MAX 件から LLM に一番明るいものを選ばせる (director._pick_topic)。"""
    rows = store.query(
        "SELECT * FROM topic_cards WHERE used_at IS NULL AND (expires_at IS NULL OR expires_at >= ?) "
        "ORDER BY COALESCE(published_at, 0) DESC, id ASC",
        (now_ms,),
    )
    rows.sort(key=lambda r: -card_tone(r["hook"], r["facts"]))  # 安定ソートなので同点は新しい順のまま
    return rows[:limit]


PICK_MAX = 8  # LLM に明るさを比べさせる候補の上限 (見出し 8 件で約 300 トークン。多いと 4B は端を選びがち)
PICK_PROMPT = ("次のニュースの見出しから、雑談で話して一番明るい気分になれるものを選び、番号だけ答えて。"
               "どれも暗ければ一番ましなものを選んで。\n{items}")


def pick_messages(cards: list) -> list[dict]:
    """候補の見出しから一番明るいものを番号で答えさせる LLM の要求。"""
    items = "\n".join(f"{i}. {c['hook']}" for i, c in enumerate(cards, 1))
    return [{"role": "user", "content": PICK_PROMPT.format(items=items)}]


def parse_pick(text: str | None, n: int) -> int | None:
    """LLM の答えの最初の数字 (1 始まり) を 0 始まりの添字にする。範囲外・数字なしは None。"""
    m = re.search(r"\d+", (text or "").translate(str.maketrans("０１２３４５６７８９", "0123456789")))
    if m is None or not 1 <= int(m.group()) <= n:
        return None
    return int(m.group()) - 1


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

    @classmethod
    def from_texts(cls, ai_texts: list[str], window: int = RATIO_Q_WINDOW) -> "QuestionRatio":
        """再生済みの AI 発話 (古い順) から作る。割り込みで途中までしか聞こえなかった発話は質問まで届いていない。"""
        r = cls(window)
        for t in ai_texts:
            r.record(ends_with_question(t))
        return r


_TRAIL = "」』)）\"' 　\n"
_REQUESTS = ("教えて", "聞かせて")  # 疑問符なしでも返事を求める締め (「趣味を聞かせてほしい」等、2026-10-19 の調査)


def ends_with_question(text: str) -> bool:
    """発話が質問・返事の要求で終わっているか。"""
    s = (text or "").rstrip(_TRAIL)
    if s.endswith(("?", "？")):
        return True
    s = s.rstrip("。！!…ー〜♪ね よな").rstrip(_TRAIL)
    return any(w in s[-12:] for w in _REQUESTS)
