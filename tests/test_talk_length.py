"""返答の長さの目安 (talk_length) と、検索・ページの答えでの差し替え。"""
from idle_talk.browser_page import page_messages
from idle_talk.talk_length import (ANSWER_SENTENCES, CHAT_LONG_NOTE, CHAT_SHORT_NOTE, LONG_USER_CHARS,
                                   chat_note, strip_chat_note)
from idle_talk.web_search import answer_messages


def test_short_chat_gets_short_note():
    assert chat_note("今日は寒いね") == CHAT_SHORT_NOTE
    assert chat_note("") == CHAT_SHORT_NOTE


def test_long_utterance_gets_long_note():
    assert chat_note("あ" * LONG_USER_CHARS) == CHAT_LONG_NOTE
    assert chat_note("あ" * (LONG_USER_CHARS - 1)) == CHAT_SHORT_NOTE


def test_punctuation_not_counted():
    assert chat_note("、。 " * LONG_USER_CHARS) == CHAT_SHORT_NOTE


def test_ask_words_get_long_note():
    for t in ("なんで？", "どうして", "詳しく教えて", "これどう思う"):
        assert chat_note(t) == CHAT_LONG_NOTE, t


def test_strip_chat_note_removes_only_note():
    sys = [{"role": "system", "content": "人物設定" + CHAT_SHORT_NOTE}]
    assert strip_chat_note(sys) == [{"role": "system", "content": "人物設定"}]
    assert sys[0]["content"].endswith(CHAT_SHORT_NOTE)  # 元は変えない
    assert strip_chat_note([]) == []


def _msgs():
    return [{"role": "system", "content": "人物設定" + CHAT_SHORT_NOTE},
            {"role": "user", "content": "天気は？"}]


def test_search_answer_uses_answer_length():
    out = answer_messages(_msgs(), "天気", [{"title": "t", "url": "https://e/", "snippet": "晴れ"}])
    assert CHAT_SHORT_NOTE not in out[0]["content"]
    assert ANSWER_SENTENCES in out[-1]["content"]


def test_page_answer_uses_answer_length():
    out = page_messages(_msgs(), {"title": "記事", "text": "本文"}, 4)
    assert CHAT_SHORT_NOTE not in out[0]["content"]
    assert ANSWER_SENTENCES in out[-1]["content"]
    assert "抜粋に無いことは推測せず" in out[-1]["content"]
