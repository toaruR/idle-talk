"""遅延予算・合否ゲート判定（純関数, 設計書 §1.2 / §8）。"""
from __future__ import annotations

from .config import Config

G1_P50_MS = 1200
G1_P95_MS = 2000
G2_P95_MS = 250
G2_BREAKDOWN_MS = (40, 60, 30)  # VAD 窓 / 送出 / ブラウザ
G3_MAX_OLD_AUDIO = 0
G4_MAX_FALSE_EOT = 2
G5_MAX_SLOPE_MIB_PER_H = 1.0
CTX_TOKENS = 4096
TOKENS_PER_TURN = 250
SYSTEM_TOKENS = 500
_DEFAULT = Config()


def g1_p50_pass(total_ms: float) -> bool:
    return total_ms <= G1_P50_MS


def g1_p50_margin(cfg: Config = _DEFAULT) -> int:
    return G1_P50_MS - (cfg.T_silence + cfg.VAR_SUM)


def g1_k_max(cfg: Config = _DEFAULT) -> float:
    return (G1_P95_MS - cfg.T_silence) / cfg.VAR_SUM


def g1_p95_pass(k: float, cfg: Config = _DEFAULT) -> bool:
    return cfg.T_silence + cfg.VAR_SUM * k <= G1_P95_MS


def g2_pass(ms: float) -> bool:
    return ms <= G2_P95_MS


def g2_sum() -> int:
    return sum(G2_BREAKDOWN_MS)


def confirm_cancel_ms(t_confirm: int, cfg: Config = _DEFAULT) -> int:
    return t_confirm + cfg.G2_SUM


def g3_pass(old_audio_count: int) -> bool:
    return old_audio_count <= G3_MAX_OLD_AUDIO


def g4_pass(false_eot_count: int) -> bool:
    return false_eot_count <= G4_MAX_FALSE_EOT


def g5_pass(delta_mib: float, hours: float) -> bool:
    return delta_mib / hours <= G5_MAX_SLOPE_MIB_PER_H


def strong_utt(avg_1s_volume: float, current_volume: float, ratio: float = 2.0) -> bool:
    return current_volume >= avg_1s_volume * ratio - 1e-9


def n_turns_tokens(turns: int) -> int:
    return turns * TOKENS_PER_TURN + SYSTEM_TOKENS


def n_turns_fits(tokens: int) -> bool:
    return tokens <= CTX_TOKENS


def tts_ratio_pass(audio_s: float, synth_s: float, cfg: Config = _DEFAULT) -> bool:
    return round(synth_s / audio_s, 10) < cfg.TTS_RATIO_MAX


def tts_gpu_compare(tts_ms: float, cfg: Config = _DEFAULT) -> bool:
    """True なら GPU1 DirectML と比較。ちょうど閾値は CPU 継続。"""
    return tts_ms > cfg.TTS_GPU_THRESH
