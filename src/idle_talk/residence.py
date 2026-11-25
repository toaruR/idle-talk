"""話者ごとの居住地と、聞き過ぎないための質問記録。"""
from __future__ import annotations

import json
import random
import re
from collections.abc import Callable

from .store import Store

ASKED_KIND = "residence_asked"
ASK_GAP_MS = 24 * 60 * 60 * 1000
REPLY_WINDOW_MS = 5 * 60 * 1000
QUESTIONS = (
    "そういえば、普段はどの辺りに住んでるの？",
    "そういえば、住んでいるのはどの市や町の辺り？",
    "普段はどの辺りで暮らしてるの？",
)
ASK_NOTE = "\n返答の後で住んでいる地域を軽く尋ねるので、この返答の本文は質問で終えないでください。詳しい住所は尋ねません。"
_RESIDENCE = re.compile(
    r"^利用者は(?:現在[、は]?|今は)?"
    r"(?P<place>[一-龯ぁ-ゖァ-ヺー々ヶケA-Za-z ・]{2,32}?)"
    r"(?:に(?:住んでいる|住んでいます|暮らしている|暮らしています|住む)|在住(?:です)?)\s*[。．.]?$"
)
_RESIDENCE_LABEL = re.compile(r"^利用者の(?:居住地|住まい)は(?P<place>[一-龯ぁ-ゖァ-ヺー々ヶケA-Za-z ・]{2,32}?)です[。．.]?$")


def residence_place(content: str) -> str | None:
    match = _RESIDENCE.fullmatch(content) or _RESIDENCE_LABEL.fullmatch(content)
    if match:
        place = match["place"].strip()
        if place not in ("不明", "未定") and not any(word in place for word in (
                "以前", "昔", "出身", "旅行", "予定", "ではなく", "じゃなく", "友人が", "友達が", "家族が",
                "妻が", "夫が", "子供が", "子どもが", "両親が", "父が", "母が", "兄が", "姉が", "弟が", "妹が")):
            return place
    return None


def residence(store: Store, persona: str, sid: int | None) -> str | None:
    """現在の話者の記憶だけを使う。出身地・旅行先・過去や否定の居住情報は使わない。"""
    if sid is None:
        return None
    rows = store.query(
        "SELECT content FROM memories WHERE speaker_id=? AND (persona IS ? OR persona IS NULL) "
        "ORDER BY updated_at DESC, id DESC", (sid, persona))
    for row in rows:
        content = row["content"] or ""
        place = residence_place(content)
        if place:
            return place
        if any(word in content for word in (
                "住んでいない", "住んでいません", "居住地は不明", "住まいは不明", "居住地は未定", "引っ越", "転居")):
            return None
    return None


class ResidenceQuestions:
    def __init__(self, rand: Callable[[], float] = random.random) -> None:
        self.rand = rand

    def choose(self, store: Store, persona: str, sid: int, now_ms: int) -> str:
        if residence(store, persona, sid) is not None or self.rand() >= 0.1:
            return ""
        rows = store.query(
            "SELECT MAX(ts) FROM events WHERE kind=? AND json_extract(payload_json, '$.speaker_id')=?",
            (ASKED_KIND, sid))
        last = rows[0][0]
        if last is not None and now_ms - last < ASK_GAP_MS:
            return ""
        return random.choice(QUESTIONS)

    def heard(self, store: Store, sid: int, now_ms: int, epoch: int) -> None:
        store.add_event(now_ms, epoch, ASKED_KIND, json.dumps({"speaker_id": sid}))
