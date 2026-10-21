"""ペルソナ (設計書 6.6)。personas.toml の人物設定と共通ルールを読み、system プロンプトに使う。

ファイルは更新時刻が変わったら読み直す (保存すれば再起動なしで次の応答から効く)。
不正な内容は警告して前の内容を使い続ける。選択肢の増減 (画面の一覧) は再起動後に反映される。
"""
from __future__ import annotations

import logging
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


DEFAULT = Persona("default", "", "気さくな雑談相手", "あなたは気さくな日本語の雑談相手です。")
NAME_PROMPT = ("あなたの名前は「{name}」です。名前を聞かれたら「{name}」と名乗り、"
               "「{name}」と呼ばれたら自分のことだと分かって答えてください。\n")


def parse_personas(raw: dict) -> tuple[list[Persona], str]:
    """personas.toml の中身 -> (ペルソナ一覧 (ファイル順), 既定の id)。不正なら ValueError。"""
    raw = dict(raw)
    default = raw.pop("default", None)
    if not isinstance(raw.pop("style_rules", ""), str):
        raise ValueError("style_rules must be a string")
    out: list[Persona] = []
    for pid, t in raw.items():
        if not isinstance(t, dict):
            raise ValueError(f"persona {pid!r} must be a table")
        if not 1 <= len(pid) <= ID_MAX:
            raise ValueError(f"persona id must be 1..{ID_MAX} chars: {pid!r}")
        unknown = set(t) - {"name", "label", "prompt"}
        if unknown:
            raise ValueError(f"unknown keys in [{pid}]: {sorted(unknown)}")
        prompt = t.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"[{pid}].prompt must be a non-empty string")
        for key in ("name", "label"):
            if key in t and (not isinstance(t[key], str) or not t[key].strip()):
                raise ValueError(f"[{pid}].{key} must be a non-empty string")
        name = t.get("name", "").strip()
        out.append(Persona(pid, name, t.get("label", "").strip() or name or pid, prompt.strip()))
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
        self._mtime: float | None = None
        if self.path is not None and self.path.is_file():
            self._mtime = self._stat()
            with self.path.open("rb") as f:  # 起動時の不正は ValueError / TOMLDecodeError で止める
                raw = tomllib.load(f)
                self.personas, self.default_id = parse_personas(raw)
                self.style_rules = raw.get("style_rules", "").strip()
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
        except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
            log.warning("personas not reloaded (keeping previous): %s", e)
            return False
        self.personas, self.default_id = personas, default_id
        self.style_rules = style_rules
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
        prompt = (NAME_PROMPT.format(name=p.name) if p.name else "") + p.prompt
        return prompt + ("\n" + self.style_rules if self.style_rules else "")

    def choices(self) -> list[dict]:
        """画面の選択肢 (/config.json の personas)。"""
        return [{"id": p.id, "name": p.label} for p in self.personas]
