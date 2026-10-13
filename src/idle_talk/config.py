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


SPONT_INTERVAL_RANGE = (10, 3600)  # 自発発話の沈黙秒数の受付範囲 (protocol の ui.interval_s と同じ)


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
    spont_interval_s: int = 1200  # 沈黙がこの秒数続いたら話しかける (旧 freq_per_hour=3 相当)
    quiet_start: str = "22:00"
    quiet_end: str = "07:00"
    radio: bool = False  # ラジオモード: 返答を待たず RADIO_GAP_S の間隔で話し続ける

    def __post_init__(self) -> None:
        if not 300 <= self.T_confirm <= 500:
            raise ValueError(f"T_confirm must be in [300, 500]: {self.T_confirm}")
        if not 250 <= self.SIL_BACKCHANNEL <= 350:
            raise ValueError(f"SIL_BACKCHANNEL must be in [250, 350]: {self.SIL_BACKCHANNEL}")
        if self.SIL_BACKCHANNEL >= self.T_silence:
            raise ValueError("SIL_BACKCHANNEL must be < T_silence")
        lo, hi = SPONT_INTERVAL_RANGE
        if type(self.spont_interval_s) is not int or not lo <= self.spont_interval_s <= hi:
            raise ValueError(f"spont_interval_s must be an int in [{lo}, {hi}]: {self.spont_interval_s!r}")
        if not (_hhmm_ok(self.quiet_start) and _hhmm_ok(self.quiet_end)):
            raise ValueError("quiet_start/quiet_end must be HH:MM")


class Ports(NamedTuple):
    """サーバ/接続先ポート。共通設定ファイル (idle_talk.toml の [ports]) で上書きする。"""
    ws: int = 8765
    asr: int = 8090
    llm: int = 8080
    tts: int = 10101
    web: int = 8000  # web/ の静的配信 (main が起動)


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


# [tts] の声パラメータ: 設定キー -> (AivisSpeech AudioQuery のフィールド, 下限, 上限)。
# 範囲は AivisSpeech Engine 1.2.0 の openapi の説明に従う (無音長と句読点の倍率は上限の記載なし)。
VOICE_PARAMS: dict[str, tuple[str, float, float]] = {
    "speed": ("speedScale", 0.5, 2.0),              # 話速
    "intonation": ("intonationScale", 0.0, 2.0),    # スタイルの感情表現の強さ
    "tempo_dynamics": ("tempoDynamicsScale", 0.0, 2.0),  # テンポの緩急
    "pitch": ("pitchScale", -0.15, 0.15),           # 音高 (0 以外は音質が劣化する)
    "volume": ("volumeScale", 0.0, 2.0),            # 音量
    "pause": ("pauseLengthScale", 0.0, float("inf")),     # 句読点の無音の倍率
    "pre_silence": ("prePhonemeLength", 0.0, float("inf")),   # 音声の前の無音 (秒)
    "post_silence": ("postPhonemeLength", 0.0, float("inf")),  # 音声の後の無音 (秒)
}
_TTS_KEYS = ("speaker", *VOICE_PARAMS)


def load_speaker(path: str | Path = SETTINGS_FILE) -> int:
    """[tts].speaker を読む。無ければ DEFAULT_SPEAKER。非整数・負数は ValueError。"""
    v = _read_section(path, "tts", _TTS_KEYS).get("speaker", DEFAULT_SPEAKER)
    if type(v) is not int or v < 0:
        raise ValueError(f"[tts].speaker must be a non-negative int: {v!r}")
    return v


def load_voice(path: str | Path = SETTINGS_FILE) -> dict[str, float]:
    """[tts] の声パラメータを AudioQuery のフィールド名で返す (指定されたキーのみ)。

    数値でない・VOICE_PARAMS の範囲外は ValueError。"""
    raw = _read_section(path, "tts", _TTS_KEYS)
    out: dict[str, float] = {}
    for k, (field, lo, hi) in VOICE_PARAMS.items():
        if k not in raw:
            continue
        v = raw[k]
        if type(v) not in (int, float) or not lo <= v <= hi:
            raise ValueError(f"[tts].{k} must be a number in [{lo}, {hi}]: {v!r}")
        out[field] = float(v)
    return out


def load_spontaneous(path: str | Path = SETTINGS_FILE) -> dict:
    """[spontaneous] (on / interval_s / quiet_start / quiet_end / radio) を Config の引数名で返す (指定キーのみ)。

    型違い・未知キーは ValueError。範囲と HH:MM は Config 生成時に検証される。"""
    raw = _read_section(path, "spontaneous", ("on", "interval_s", "quiet_start", "quiet_end", "radio"))
    types = {"on": bool, "interval_s": int, "quiet_start": str, "quiet_end": str, "radio": bool}
    for k, v in raw.items():
        if type(v) is not types[k]:
            raise ValueError(f"[spontaneous].{k} must be {types[k].__name__}: {v!r}")
    names = {"on": "spontaneous_enabled", "interval_s": "spont_interval_s"}
    return {names.get(k, k): v for k, v in raw.items()}


class LlmLaunch(NamedTuple):
    """[llm]: Director が llama-server を起動・監督するための設定。model が空なら手動起動 (監督なし)。"""
    server: str = "llama-server"
    model: str = ""
    ngl: int = 99
    ctx: int = 4096


def load_llm_launch(path: str | Path = SETTINGS_FILE) -> LlmLaunch:
    """[llm] を読む。無ければ既定値 (監督なし)。型違い・未知キーは ValueError。"""
    raw = _read_section(path, "llm", LlmLaunch._fields)
    for k, v in raw.items():
        want = str if k in ("server", "model") else int
        if type(v) is not want or (want is int and v < 0):
            raise ValueError(f"[llm].{k} must be a {'string' if want is str else 'non-negative int'}: {v!r}")
    return LlmLaunch(**raw)


class Topics(NamedTuple):
    """[topics]: 話題カードを作る RSS / Atom フィード。feeds が空ならネットワークに出ない。"""
    feeds: tuple[str, ...] = ()
    interval_min: int = 30  # 取得間隔 (分)


def load_topics(path: str | Path = SETTINGS_FILE) -> Topics:
    """[topics] を読む。無ければ既定値 (取得しない)。型違い・未知キーは ValueError。"""
    raw = _read_section(path, "topics", Topics._fields)
    feeds = raw.get("feeds", [])
    if type(feeds) is not list or not all(type(u) is str and u.startswith(("http://", "https://")) for u in feeds):
        raise ValueError(f"[topics].feeds must be a list of http(s) URLs: {feeds!r}")
    interval = raw.get("interval_min", Topics._field_defaults["interval_min"])
    if type(interval) is not int or interval < 1:
        raise ValueError(f"[topics].interval_min must be a positive int: {interval!r}")
    return Topics(tuple(feeds), interval)


class VadSettings(NamedTuple):
    """[vad]: 発話検出 (vad.EnergyVad)。マイク音量を上げて物音で止まるなら threshold か onset_ms を上げる。"""
    threshold: float = 500.0  # 発話とみなすフレーム RMS (16bit PCM)
    onset_ms: int = 100       # この長さ音が続いたら発話開始 (短い物音を無視する)。20ms 単位に切り捨て


def load_vad(path: str | Path = SETTINGS_FILE) -> VadSettings:
    """[vad] を読む。無ければ既定値。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "vad", VadSettings._fields)
    t = raw.get("threshold", VadSettings._field_defaults["threshold"])
    if type(t) not in (int, float) or not 0 < t <= 32767:
        raise ValueError(f"[vad].threshold must be a number in (0, 32767]: {t!r}")
    onset = raw.get("onset_ms", VadSettings._field_defaults["onset_ms"])
    if type(onset) is not int or not 20 <= onset <= 1000:
        raise ValueError(f"[vad].onset_ms must be an int in [20, 1000]: {onset!r}")
    return VadSettings(float(t), onset)
