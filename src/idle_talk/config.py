"""Options (設計書 8.1 既定値表)。時間は特記なき限り ms。"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

SETTINGS_FILE = "idle_talk.toml"


class LlmRetry(NamedTuple):
    max: int = 3
    window_s: int = 600
    backoff_s: tuple[int, ...] = (1, 2, 4)


class StrongUtt(NamedTuple):
    volume_ratio: float = 2.0
    window_s: float = 1.0


class AsrHallu(NamedTuple):
    no_speech_prob: float = 0.6
    compression_ratio: float = 2.4


class HttpTimeout(NamedTuple):
    llm_first: int = 5
    llm_total: int = 20
    tts_sentence: int = 3
    asr_factor: int = 2  # asr = 区間長 x asr_factor + asr_offset_s 秒
    asr_offset_s: int = 2

    def asr(self, segment_s: float) -> float:
        return segment_s * self.asr_factor + self.asr_offset_s


class EParams(NamedTuple):
    silence_input_s: int = 60
    device_lost_s: int = 3
    tts_cpu_settle_s: int = 2
    tts_cpu_delta_pt: int = 5


def _hhmm_ok(s: str) -> bool:
    import re

    return bool(re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", s))


@dataclass(frozen=True)
class Config:
    T_silence: int = 400
    TTS_GPU_THRESH: int = 300
    PING_INTERVAL: int = 5
    LLM_RETRY: LlmRetry = field(default_factory=LlmRetry)
    IDLE_TIMEOUT: int = 1800
    STRONG_UTT: StrongUtt = field(default_factory=StrongUtt)
    OVERLAP_MAX: int = 300
    HTTP_TIMEOUT: HttpTimeout = field(default_factory=HttpTimeout)
    DUCK_GAIN: float = 0.2
    VAR_SUM: int = 750
    G2_SUM: int = 130
    TTS_RATIO_MAX: float = 0.7
    TEST_N: int = 20
    E_PARAMS: EParams = field(default_factory=EParams)
    DB_BACKUP: int = 1
    MIN_SEG: int = 200
    T_confirm: int = 400
    STOP_WORDS: tuple[str, ...] = ("待って", "違う", "ちょっと", "ストップ")
    N_TURNS: int = 8
    SIL_BACKCHANNEL: int = 300
    BC_COOLDOWN: int = 5
    RATIO_Q: tuple[int, int] = (1, 3)
    ASR_HALLU: AsrHallu = field(default_factory=AsrHallu)
    # 自発発話 (既定オフ)
    spontaneous_enabled: bool = False
    freq_per_hour: int = 3
    quiet_start: str = "22:00"
    quiet_end: str = "07:00"

    def __post_init__(self) -> None:
        if not 300 <= self.T_confirm <= 500:
            raise ValueError(f"T_confirm must be in [300, 500]: {self.T_confirm}")
        if not 250 <= self.SIL_BACKCHANNEL <= 350:
            raise ValueError(f"SIL_BACKCHANNEL must be in [250, 350]: {self.SIL_BACKCHANNEL}")
        if self.SIL_BACKCHANNEL >= self.T_silence:
            raise ValueError("SIL_BACKCHANNEL must be < T_silence")
        if not 0 <= self.freq_per_hour <= 12:
            raise ValueError(f"freq_per_hour must be in [0, 12]: {self.freq_per_hour}")
        if not (_hhmm_ok(self.quiet_start) and _hhmm_ok(self.quiet_end)):
            raise ValueError("quiet_start/quiet_end must be HH:MM")


class Ports(NamedTuple):
    """サーバ/接続先ポート。共通設定ファイル (idle_talk.toml の [ports]) で上書きする。"""
    ws: int = 8765
    asr: int = 8090
    llm: int = 8080
    tts: int = 10101


DEFAULT_SPEAKER = 888753760  # AivisSpeech の話者 (スタイル) ID


def _read_section(path: str | Path, section: str, keys: tuple[str, ...]) -> dict:
    """設定ファイルの [section] を読む。ファイルが無ければ空。未知キーは ValueError。"""
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("rb") as f:
        raw = tomllib.load(f).get(section, {})
    unknown = set(raw) - set(keys)
    if unknown:
        raise ValueError(f"unknown [{section}] keys in {p}: {sorted(unknown)}")
    return raw


def load_ports(path: str | Path = SETTINGS_FILE) -> Ports:
    """[ports] を読む。ファイルが無ければ既定値。未知キー・非整数は ValueError。"""
    raw = _read_section(path, "ports", Ports._fields)
    for k, v in raw.items():
        if type(v) is not int or not 0 <= v <= 65535:
            raise ValueError(f"[ports].{k} must be an int in [0, 65535]: {v!r}")
    return Ports(**raw)


def load_speaker(path: str | Path = SETTINGS_FILE) -> int:
    """[tts].speaker を読む。無ければ DEFAULT_SPEAKER。非整数・負数は ValueError。"""
    v = _read_section(path, "tts", ("speaker",)).get("speaker", DEFAULT_SPEAKER)
    if type(v) is not int or v < 0:
        raise ValueError(f"[tts].speaker must be a non-negative int: {v!r}")
    return v
