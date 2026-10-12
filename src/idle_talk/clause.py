"""LLM トークン列の節分割 (設計書 3.3 / 4.3)。句点・読点等で最初の合成可能な節が確定した時点で返す。"""
from __future__ import annotations

CLAUSE_DELIMS = "。、！？!?,，．\n"
_PUNCT_ONLY = set(CLAUSE_DELIMS) | set(" 　「」『』（）()・…ー〜~-")


def is_synthesizable(clause: str) -> bool:
    """区切り・記号のみの節は合成不能。"""
    return any(c not in _PUNCT_ONLY for c in clause)


class ClauseSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        """トークンを追加し、確定した合成可能な節 (区切り文字を含む) を順に返す。"""
        out: list[str] = []
        for ch in text:
            self._buf += ch
            if ch in CLAUSE_DELIMS and is_synthesizable(self._buf):
                out.append(self._buf.strip())
                self._buf = ""
        return out

    def flush(self) -> list[str]:
        """ストリーム終端の残り。合成可能なら 1 節として返す。"""
        rest, self._buf = self._buf.strip(), ""
        return [rest] if rest and is_synthesizable(rest) else []
