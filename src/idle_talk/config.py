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
    tts_sentence: int = 6  # ドラマ視聴中は台詞の認識と CPU を取り合い、長めの 1 文が 3 秒を超えた (2026-10-15)
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
    main_gpu: int = 0   # 載せる GPU の番号 (PCI 順)。-1 なら llama-server の自動配置 (複数 GPU に分割)
    mmproj: str = ""    # 画像エンコーダ (mmproj)。指定するとドラマモードで画面も見る
    image_max_tokens: int = 0  # 画像 1 枚のトークン上限。0 なら llama-server の既定


def load_llm_launch(path: str | Path = SETTINGS_FILE) -> LlmLaunch:
    """[llm] を読む。無ければ既定値 (監督なし)。型違い・未知キーは ValueError。"""
    raw = _read_section(path, "llm", LlmLaunch._fields)
    for k, v in raw.items():
        want = str if k in ("server", "model", "mmproj") else int
        low = -1 if k == "main_gpu" else 0
        if type(v) is not want or (want is int and v < low):
            raise ValueError(f"[llm].{k} must be a {'string' if want is str else 'non-negative int'}: {v!r}")
    return LlmLaunch(**raw)


class AivisLaunch(NamedTuple):
    """[aivis]: ctl start が AivisSpeech Engine を起動するための設定。engine が空なら手動起動。"""
    engine: str = ""  # AivisSpeech Engine の run.exe
    use_gpu: bool = False  # TTS は CPU (設計書 §9)


def load_aivis_launch(path: str | Path = SETTINGS_FILE) -> AivisLaunch:
    """[aivis] を読む。無ければ既定値 (手動起動)。型違い・未知キーは ValueError。"""
    raw = _read_section(path, "aivis", AivisLaunch._fields)
    for k, v in raw.items():
        want = str if k == "engine" else bool
        if type(v) is not want:
            raise ValueError(f"[aivis].{k} must be a {want.__name__}: {v!r}")
    return AivisLaunch(**raw)


class Topics(NamedTuple):
    """[topics]: 話題カードを作る RSS / Atom フィード。feeds が空ならネットワークに出ない。"""
    feeds: tuple[str, ...] = ()
    interval_min: int = 30  # 取得間隔 (分)
    x_trends: bool = False  # 取得のたびに X のトレンド (Yahoo! リアルタイム検索) もカードにする


def load_topics(path: str | Path = SETTINGS_FILE) -> Topics:
    """[topics] を読む。無ければ既定値 (取得しない)。型違い・未知キーは ValueError。"""
    raw = _read_section(path, "topics", Topics._fields)
    feeds = raw.get("feeds", [])
    if type(feeds) is not list or not all(type(u) is str and u.startswith(("http://", "https://")) for u in feeds):
        raise ValueError(f"[topics].feeds must be a list of http(s) URLs: {feeds!r}")
    interval = raw.get("interval_min", Topics._field_defaults["interval_min"])
    if type(interval) is not int or interval < 1:
        raise ValueError(f"[topics].interval_min must be a positive int: {interval!r}")
    x_trends = raw.get("x_trends", False)
    if type(x_trends) is not bool:
        raise ValueError(f"[topics].x_trends must be bool: {x_trends!r}")
    return Topics(tuple(feeds), interval, x_trends)


class Search(NamedTuple):
    """[search]: 利用者の質問に答えるための Web 検索。on でなければネットワークに出ない。"""
    on: bool = False
    engines: tuple[str, ...] = ("duckduckgo", "bing")  # 先頭から試し、失敗・0 件なら次へ
    read_page: bool = True  # 上位 1 件のページ本文を r.jina.ai で読んで添える
    realtime: bool = False  # X の投稿 (Yahoo! リアルタイム検索) も添える


SEARCH_ENGINES = ("duckduckgo", "bing")


def load_search(path: str | Path = SETTINGS_FILE) -> Search:
    """[search] を読む。無ければ既定値 (検索しない)。型違い・未知キー・未知のエンジンは ValueError。"""
    raw = _read_section(path, "search", Search._fields)
    for k in ("on", "read_page", "realtime"):
        if k in raw and type(raw[k]) is not bool:
            raise ValueError(f"[search].{k} must be bool: {raw[k]!r}")
    engines = raw.get("engines", list(Search._field_defaults["engines"]))
    if type(engines) is not list or not engines or not all(e in SEARCH_ENGINES for e in engines):
        raise ValueError(f"[search].engines must be a non-empty list of {list(SEARCH_ENGINES)}: {engines!r}")
    return Search(raw.get("on", False), tuple(engines), raw.get("read_page", True), raw.get("realtime", False))


class Asr(NamedTuple):
    """[asr]: 音声認識のエンジン。reazon は ReazonSpeech (sherpa-onnx, CPU)、whisper は faster-whisper small (GPU)。"""
    engine: str = "reazon"
    model_dir: str = "data/models/reazonspeech"  # reazon の encoder/decoder/joiner*.onnx と tokens.txt
    threads: int = 2  # reazon の CPU スレッド数 (AivisSpeech・話者識別と CPU を分け合う)
    # short_chars 字以下の認識結果は、AI の声から short_reply_s 秒以内 (返事・相槌) でなければ物音の聞き違いとして捨てる。
    # ReazonSpeech は食器・咳を「はい」「誰」「ハハ」と聞き違えるが、音では本物の返事と見分けられなかった (2026-10-15)
    short_chars: int = 2  # 0 で無効
    short_reply_s: int = 20  # 0 なら AI の声の最中も含めて常に捨てる


ASR_ENGINES = ("reazon", "whisper")


def load_asr(path: str | Path = SETTINGS_FILE) -> Asr:
    """[asr] を読む。無ければ既定値 (reazon)。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "asr", Asr._fields)
    if "engine" in raw and raw["engine"] not in ASR_ENGINES:
        raise ValueError(f"[asr].engine must be one of {list(ASR_ENGINES)}: {raw['engine']!r}")
    if "model_dir" in raw and (type(raw["model_dir"]) is not str or not raw["model_dir"]):
        raise ValueError(f"[asr].model_dir must be a non-empty string: {raw['model_dir']!r}")
    if "threads" in raw and (type(raw["threads"]) is not int or not 1 <= raw["threads"] <= 32):
        raise ValueError(f"[asr].threads must be an int in [1, 32]: {raw['threads']!r}")
    for k, hi in (("short_chars", 10), ("short_reply_s", 3600)):
        if k in raw and (type(raw[k]) is not int or not 0 <= raw[k] <= hi):
            raise ValueError(f"[asr].{k} must be an int in [0, {hi}]: {raw[k]!r}")
    return Asr(**raw)


class VoiceId(NamedTuple):
    """[voice_id]: 声で話し相手を見分け、初めての声なら名前を聞く (設計書 6.4)。声の特徴は外部に送らない。"""
    on: bool = False
    model: str = "data/models/voxceleb_resnet34_LM.onnx"  # WeSpeaker の話者埋め込みモデル (ONNX)
    threshold: float = 0.6  # 同じ人とみなすコサイン類似度。別人を同じ人と取り違えるなら上げる


def load_voice_id(path: str | Path = SETTINGS_FILE) -> VoiceId:
    """[voice_id] を読む。無ければ既定値 (使わない)。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "voice_id", VoiceId._fields)
    if "on" in raw and type(raw["on"]) is not bool:
        raise ValueError(f"[voice_id].on must be bool: {raw['on']!r}")
    if "model" in raw and (type(raw["model"]) is not str or not raw["model"]):
        raise ValueError(f"[voice_id].model must be a non-empty string: {raw['model']!r}")
    th = raw.get("threshold", VoiceId._field_defaults["threshold"])
    if type(th) not in (int, float) or not 0 < th < 1:
        raise ValueError(f"[voice_id].threshold must be a number in (0, 1): {th!r}")
    return VoiceId(raw.get("on", False), raw.get("model", VoiceId._field_defaults["model"]), float(th))


class VadSettings(NamedTuple):
    """[vad]: 発話検出 (vad.EnergyVad)。物音で発話扱いになりすぎるなら ratio (自動校正時) か onset_ms を上げる。"""
    threshold: float = 500.0     # 発話とみなすフレーム RMS (16bit PCM)。自動校正時は推定がそろうまでの値
    onset_ms: int = 100          # この長さ音が続いたら発話開始 (短い物音を無視する)。20ms 単位に切り捨て
    adaptive: bool = True        # 閾値を周囲の雑音と再生中の回り込みから自動で決める
    ratio: float = 3.0           # 自動校正の閾値 = 推定した雑音 (再生中は回り込み) x ratio
    min_threshold: float = 100.0  # 自動校正の閾値の下限


_VAD_RANGES = {"threshold": (0.0, 32767.0), "ratio": (1.0, 100.0), "min_threshold": (0.0, 32767.0)}


def load_vad(path: str | Path = SETTINGS_FILE) -> VadSettings:
    """[vad] を読む。無ければ既定値。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "vad", VadSettings._fields)
    out = dict(VadSettings._field_defaults)
    for k, v in raw.items():
        if k in _VAD_RANGES:
            lo, hi = _VAD_RANGES[k]
            if type(v) not in (int, float) or not lo < v <= hi:
                raise ValueError(f"[vad].{k} must be a number in ({lo:g}, {hi:g}]: {v!r}")
            v = float(v)
        elif k == "onset_ms":
            if type(v) is not int or not 20 <= v <= 1000:
                raise ValueError(f"[vad].onset_ms must be an int in [20, 1000]: {v!r}")
        elif type(v) is not bool:
            raise ValueError(f"[vad].{k} must be bool: {v!r}")
        out[k] = v
    return VadSettings(**out)


class Drama(NamedTuple):
    """[drama]: ドラマモード (共有したタブの音を聞き、場面の切れ目にたまに一言)。台詞はメモリにだけ置く。"""
    interval_s: int = 90    # 前の一言 (または開始) から最低これだけ空ける
    gap_ms: int = 2500      # ドラマの台詞が途切れてこれだけ経ったら場面の切れ目とみなす
    window_s: int = 180     # LLM に見せる台詞の範囲 (直近の秒数)
    min_chars: int = 60     # 前の一言から溜まった台詞がこれ未満なら黙る
    ctx_chars: int = 1200   # LLM に渡す台詞の上限字数 (古い方から捨てる)
    max_seg_s: int = 15     # 1 区間の上限。BGM 等で区間が終わらなくても切って認識に回す
    finale_gap_s: int = 60  # 台詞がこれだけ聞こえなければ見終わった (一時停止) とみなし、それまでの感想を話す
    finale_min_chars: int = 600  # 前の感想 (または開始) から聞いた台詞がこれ未満なら感想は言わない
    digest_chars: int = 800  # 台詞がこれだけ溜まったら裏であらすじメモに要約する (window_s を過ぎても話を覚えておく)
    log_prompt: bool = False  # 一言・感想・要約に LLM へ渡した台詞とメモを talk.log に残す (既定は保存しない)
    keep_snaps: int = 0     # 一言・感想に使った画面を data/snaps に新しい順でこの枚数だけ残す (0 なら保存しない)
    mic: bool = True  # ドラマモード中もマイクを聞く。False なら聞くだけ (話しかけられないが、拾った台詞を利用者の発話と取らない)


_DRAMA_RANGES = {"interval_s": (10, 3600), "gap_ms": (500, 30000), "window_s": (30, 1800),
                 "min_chars": (0, 2000), "ctx_chars": (100, 4000), "max_seg_s": (3, 60),
                 "finale_gap_s": (10, 1800), "finale_min_chars": (0, 20000), "digest_chars": (100, 2000), "keep_snaps": (0, 20)}


def load_drama(path: str | Path = SETTINGS_FILE) -> Drama:
    """[drama] を読む。無ければ既定値。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "drama", Drama._fields)
    for k, v in raw.items():
        if k in ("log_prompt", "mic"):
            if type(v) is not bool:
                raise ValueError(f"[drama].{k} must be bool: {v!r}")
            continue
        lo, hi = _DRAMA_RANGES[k]
        if type(v) is not int or not lo <= v <= hi:
            raise ValueError(f"[drama].{k} must be an int in [{lo}, {hi}]: {v!r}")
    return Drama(**raw)


class Screen(NamedTuple):
    """[screen]: 画面モード (共有した画面を interval_s ごとに見て、前に話したときから変わっていれば一言)。画面は保存しない。"""
    interval_s: int = 60  # 画面を見に行く間隔 (前の一言・見送りから)
    min_diff: int = 3     # 前に話した画面との差 (縮小した明るさの平均差、%) がこれ未満なら話さない
    follow_window: bool = False  # ブラウザの共有の代わりに前面のウィンドウを自動で撮る (Windows のみ)
    exclude: tuple[str, ...] = ()  # follow_window でタイトルにこの語を含むウィンドウは撮らない
    width: int = 1280     # follow_window で撮る画像の幅の上限 (高さは 16:9)。1280x720 で約 400 トークン (llama-server の 1 枚の上限 512 に収める)
    color_bits: int = 5   # follow_window の PNG の色数 (各色のビット数)。下げるほど小さくなる (8 で減らさない)


_SCREEN_RANGES = {"interval_s": (10, 3600), "min_diff": (0, 100), "width": (320, 1280), "color_bits": (4, 8)}


def load_screen(path: str | Path = SETTINGS_FILE) -> Screen:
    """[screen] を読む。無ければ既定値。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "screen", Screen._fields)
    for k, v in raw.items():
        if k == "follow_window":
            if type(v) is not bool:
                raise ValueError(f"[screen].follow_window must be bool: {v!r}")
        elif k == "exclude":
            if type(v) is not list or not all(type(s) is str and s for s in v):
                raise ValueError(f"[screen].exclude must be a list of non-empty strings: {v!r}")
            raw[k] = tuple(v)
        else:
            lo, hi = _SCREEN_RANGES[k]
            if type(v) is not int or not lo <= v <= hi:
                raise ValueError(f"[screen].{k} must be an int in [{lo}, {hi}]: {v!r}")
    return Screen(**raw)


class Pc(NamedTuple):
    """[pc]: ブラウザ補正なしモード (スピーカーで鳴る別の音声 AI と、マイクとスピーカーで話す)。AI の声は Director が鳴らす。"""
    monitor: str = ""  # AI の声を鳴らす出力機器の名前 (部分一致)。空なら Windows の既定の再生機器
    mic: str = ""  # マイクの入力機器の名前 (部分一致)。空なら Windows の既定の録音機器


def load_pc(path: str | Path = SETTINGS_FILE) -> Pc:
    """[pc] を読む。無ければ既定値。型違い・範囲外・未知キーは ValueError。"""
    raw = _read_section(path, "pc", Pc._fields)
    for k in ("monitor", "mic"):
        v = raw.get(k, "")
        if type(v) is not str or len(v) > 100:
            raise ValueError(f"[pc].{k} must be a string of up to 100 chars: {v!r}")
    return Pc(**raw)
