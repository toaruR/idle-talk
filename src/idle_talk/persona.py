"""ペルソナ (設計書 6.6)。personas.toml の人物設定と共通ルールを読み、system プロンプトに使う。

ファイルは更新時刻が変わったら読み直す (保存すれば再起動なしで次の応答から効く)。
不正な内容は警告して前の内容を使い続ける。選択肢の増減 (画面の一覧) は再起動後に反映される。
"""
from __future__ import annotations

import logging
import random
import tomllib
from pathlib import Path
from typing import NamedTuple

log = logging.getLogger("idle_talk.persona")

PERSONAS_FILE = "personas.toml"
ID_MAX = 64  # ui.persona の最大長 (protocol の MESSAGE_SCHEMA と一致させる)


class Persona(NamedTuple):
    id: str
    name: str  # AI 自身の名前 (system に「あなたの名前は〜」と添える)。空なら添えない
    label: str  # 画面の選択肢に出す表示名 (省略時は name、それも無ければ id)
    prompt: str  # 人物設定 (性格・口調)。声は画面の「声」で別に選ぶ
    fallback_start: tuple[str, ...] = ()  # LLM が失敗したときの一言 (何も話す前)。空なら共通のもの
    fallback_end: tuple[str, ...] = ()  # 同 (途中まで話した後)。空なら共通のもの
    misheard: tuple[str, ...] = ()  # 音声認識が name の代わりに書きがちな語 (「うちゅ」→「宇宙」)


DEFAULT = Persona("default", "", "気さくな雑談相手", "あなたは気さくな日本語の雑談相手です。")
NAME_PROMPT = ("あなたの名前は「{name}」です。名前を聞かれたら「{name}」と名乗り、"
               "「{name}」と呼ばれたら自分のことだと分かって答えてください。\n")
# 利用者の発話は音声認識の結果なので、珍しい名前はよくある語に化ける (ASR は学習でよく見た語を選ぶ)
MISHEARD_PROMPT = ("利用者の発話は音声認識の結果で、あなたの名前「{name}」が{words}と書かれることがあります。"
                   "呼びかけや名前として使われていたら、あなたのことだと考えて答えてください。"
                   "聞き違いを指摘したり、言い直させたりはしないでください。\n")
# LLM の時間切れ・接続断で読む一言 (ランダムに 1 つ)。personas.toml のトップレベル・各ペルソナで上書きできる
FALLBACK_START = ("あれ、なんだっけ…ごめん、もう一回言って？",
                  "うーん、頭が真っ白になっちゃった。もう一度お願い。",
                  "えっと…ごめん、ちょっとぼーっとしてた。もう一回いい？")
FALLBACK_END = ("…まあ、そんな感じかな。", "…って感じ。", "うん、そんなところ。")
FALLBACK_KEYS = ("fallback_start", "fallback_end")
FALLBACK_MAX = 20  # 1 つの一覧に書ける数
FALLBACK_CHARS = 100  # 1 つの一言の最大長


def _phrases(value: object, where: str) -> tuple[str, ...]:
    """フィラーの一覧 (空でない文字列の配列) を検査して tuple にする。不正なら ValueError。"""
    if (not isinstance(value, list) or not 1 <= len(value) <= FALLBACK_MAX
            or not all(isinstance(s, str) and 1 <= len(s.strip()) <= FALLBACK_CHARS for s in value)):
        raise ValueError(f"{where} must be 1..{FALLBACK_MAX} non-empty strings (<= {FALLBACK_CHARS} chars)")
    return tuple(s.strip() for s in value)


def parse_fallbacks(raw: dict, where: str = "") -> dict[str, tuple[str, ...]]:
    """表の fallback_start / fallback_end -> {キー: 一覧}。書かれていないキーは含めない。"""
    return {k: _phrases(raw[k], where + k) for k in FALLBACK_KEYS if k in raw}


def parse_personas(raw: dict) -> tuple[list[Persona], str]:
    """personas.toml の中身 -> (ペルソナ一覧 (ファイル順), 既定の id)。不正なら ValueError。"""
    raw = dict(raw)
    default = raw.pop("default", None)
    if not isinstance(raw.pop("style_rules", ""), str):
        raise ValueError("style_rules must be a string")
    parse_fallbacks(raw)  # トップレベルの共通フィラーは検査だけ (PersonaBook が読む)
    for k in FALLBACK_KEYS:
        raw.pop(k, None)
    out: list[Persona] = []
    for pid, t in raw.items():
        if not isinstance(t, dict):
            raise ValueError(f"persona {pid!r} must be a table")
        if not 1 <= len(pid) <= ID_MAX:
            raise ValueError(f"persona id must be 1..{ID_MAX} chars: {pid!r}")
        unknown = set(t) - {"name", "label", "prompt", "misheard", *FALLBACK_KEYS}
        if unknown:
            raise ValueError(f"unknown keys in [{pid}]: {sorted(unknown)}")
        prompt = t.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"[{pid}].prompt must be a non-empty string")
        for key in ("name", "label"):
            if key in t and (not isinstance(t[key], str) or not t[key].strip()):
                raise ValueError(f"[{pid}].{key} must be a non-empty string")
        name = t.get("name", "").strip()
        fb = parse_fallbacks(t, f"[{pid}].")
        misheard = _phrases(t["misheard"], f"[{pid}].misheard") if "misheard" in t else ()
        if misheard and not name:
            raise ValueError(f"[{pid}].misheard needs name")
        out.append(Persona(pid, name, t.get("label", "").strip() or name or pid, prompt.strip(),
                           fb.get("fallback_start", ()), fb.get("fallback_end", ()), misheard))
    if not out:
        raise ValueError("no persona defined")
    if default is None:
        default = out[0].id
    if default not in {p.id for p in out}:
        raise ValueError(f"default persona not defined: {default!r}")
    return out, default


class PersonaBook:
    """ペルソナ一覧と今選んでいるもの。path が無い・読めないときは DEFAULT だけで動く。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.personas: list[Persona] = [DEFAULT]
        self.default_id = DEFAULT.id
        self.style_rules = ""
        self.fallbacks: dict[str, tuple[str, ...]] = {}  # トップレベルの共通フィラー
        self._mtime: float | None = None
        if self.path is not None and self.path.is_file():
            self._mtime = self._stat()
            with self.path.open("rb") as f:  # 起動時の不正は ValueError / TOMLDecodeError で止める
                raw = tomllib.load(f)
                self.personas, self.default_id = parse_personas(raw)
                self.style_rules = raw.get("style_rules", "").strip()
                self.fallbacks = parse_fallbacks(raw)
        self.current_id = self.default_id

    def _stat(self) -> float | None:
        try:
            return self.path.stat().st_mtime if self.path is not None else None
        except OSError:
            return None

    def reload_if_changed(self) -> bool:
        mtime = self._stat()
        if self.path is None or mtime is None or mtime == self._mtime:
            return False
        self._mtime = mtime
        try:
            with self.path.open("rb") as f:
                raw = tomllib.load(f)
                personas, default_id = parse_personas(raw)
                style_rules = raw.get("style_rules", "").strip()
                fallbacks = parse_fallbacks(raw)
        except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
            log.warning("personas not reloaded (keeping previous): %s", e)
            return False
        self.personas, self.default_id = personas, default_id
        self.style_rules = style_rules
        self.fallbacks = fallbacks
        if self.current_id not in {p.id for p in personas}:  # 選んでいたものが消えた
            self.current_id = default_id
        log.info("personas reloaded: %s", [p.id for p in personas])
        return True

    def get(self, pid: str) -> Persona | None:
        return next((p for p in self.personas if p.id == pid), None)

    def select(self, pid: str) -> None:
        """画面で選んだペルソナにする。一覧に無い id は ValueError (5006)。次の応答から効く。"""
        if self.get(pid) is None:
            raise ValueError(f"unknown persona: {pid}")
        if pid != self.current_id:
            self.current_id = pid
            log.info("persona set from ui: %s", pid)

    @property
    def current(self) -> Persona:
        self.reload_if_changed()
        return self.get(self.current_id) or self.personas[0]

    def system_prompt(self) -> str:
        p = self.current  # 更新時刻の確認と再読み込みを先に行う
        prompt = (NAME_PROMPT.format(name=p.name) if p.name else "")
        if p.misheard:
            prompt += MISHEARD_PROMPT.format(name=p.name, words="・".join(f"「{w}」" for w in p.misheard))
        prompt += p.prompt
        return prompt + ("\n" + self.style_rules if self.style_rules else "")

    def fallback(self, spoke: bool, rng: random.Random | None = None) -> str:
        """LLM が失敗したときに読む一言。spoke は途中まで話していたか。ペルソナ > 共通 > 組み込みの順。"""
        p = self.current
        key = "fallback_end" if spoke else "fallback_start"
        own = p.fallback_end if spoke else p.fallback_start
        pool = own or self.fallbacks.get(key) or (FALLBACK_END if spoke else FALLBACK_START)
        return (rng or random).choice(pool)

    def choices(self) -> list[dict]:
        """画面の選択肢 (/config.json の personas)。"""
        return [{"id": p.id, "name": p.label} for p in self.personas]
