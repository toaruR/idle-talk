"""返答の長さの目安。場面ごとに文の数を変える (2026-11-14)。

声は読み飛ばせないので、雑談は短くテンポよく返し、頼まれて答える (検索・見ているページ・「なんで」) ときだけ長くする。
共通の style_rules には文の数を書かず、通常応答の system の末尾にこの目安を 1 つ足す。
"""
from __future__ import annotations

import re

# 通常応答 (雑談)。利用者の発話の長さに合わせる
CHAT_SHORT_NOTE = "\n\n今回は 1〜2 文で短く、テンポよく返してください。"
CHAT_LONG_NOTE = "\n\n今回は 3 文ほどまでで、聞かれたことにしっかり答えてください。"
# 頼まれて調べた・読んだ内容を答えるとき (検索・見ているページ)
ANSWER_SENTENCES = "3〜5 文"
# これ以上の字数の発話には長めに返す (句読点・空白を除く。1 文 20〜30 字の 2 文ほど)
LONG_USER_CHARS = 40
# 説明・理由・意見を求める語。短い発話でも長めに返す
ASK_WORDS = re.compile(r"なんで|なぜ|どうして|どうやって|どういう|教えて|詳しく|くわしく|説明|どう思う|おすすめ")
_NOTES = (CHAT_SHORT_NOTE, CHAT_LONG_NOTE)


def chat_note(user_text: str) -> str:
    """利用者の最後の発話から、通常応答の長さの目安を返す。"""
    chars = sum(ch.isalnum() for ch in user_text)
    return CHAT_LONG_NOTE if chars >= LONG_USER_CHARS or ASK_WORDS.search(user_text) else CHAT_SHORT_NOTE


def strip_chat_note(system: list[dict]) -> list[dict]:
    """[system] から通常応答の長さの目安を除く (検索・ページの答えは別の文の数で答えさせる)。"""
    out = []
    for m in system:
        c = m["content"]
        if isinstance(c, str):
            for note in _NOTES:
                c = c.replace(note, "")
        out.append({**m, "content": c})
    return out
