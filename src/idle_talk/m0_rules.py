"""M0 判定ルール (設計書 §9 既定値更新ルール / §1.4 資源上限 / §11.1)。

実機測定値を入力に、既定値を更新するかを決める純関数群。
"""
from __future__ import annotations

from .budget import tts_gpu_compare
from .config import Config

MIB = 1
GIB_MIB = 1024  # 1GiB = 1024MiB
LLM_VRAM_MAX_MIB = 9 * GIB_MIB  # §1.4 LLM は GPU0 で 9GiB 以内
ASR_VRAM_MAX_MIB = 3 * GIB_MIB  # §1.4 ASR 追加 3GiB 以内
RAM_MAX_MIB = 8 * GIB_MIB  # §1.4 追加 RAM 8GiB 以内
BLIND_MIN, BLIND_MAX = 30, 50  # §9 ブラインド比較の場面数

# M0 計測ハーネスが出力する測定値キー
MEASUREMENT_KEYS = frozenset({
    "llm_first_clause_ms", "tts_first_sentence_ms", "tts_ratio_p95",
    "vram_mib", "asr_vram_mib", "ram_mib", "overlap_p95",
})


def llm_choose(g1_met: bool, blind_scenes: int, nine_b_wins: bool) -> str:
    """9B 採用は G1 充足とブラインド比較(30〜50 場面)での 9B 勝ちが両方成立した場合のみ。"""
    if not BLIND_MIN <= blind_scenes <= BLIND_MAX:
        raise ValueError(f"blind_scenes must be {BLIND_MIN}..{BLIND_MAX}: {blind_scenes}")
    return "9B" if (g1_met and nine_b_wins) else "4B"


def asr_place(overlap_p95: float, cfg: Config | None = None) -> str:
    """重なり追加遅延 p95 が OVERLAP_MAX ちょうどは GPU0、超過で GPU1 推奨。"""
    cfg = cfg or Config()
    return "GPU0" if overlap_p95 <= cfg.OVERLAP_MAX else "GPU1"


def tts_place(tts_first_sentence_ms: float, tts_ratio_p95: float,
              cfg: Config | None = None) -> dict:
    """最初の1文が TTS_GPU_THRESH 超なら GPU1 DirectML 比較。ratio は p95 < TTS_RATIO_MAX で OK。"""
    cfg = cfg or Config()
    return {
        "place": "GPU1_DirectML_compare" if tts_gpu_compare(tts_first_sentence_ms, cfg) else "CPU",
        "ratio_ok": tts_ratio_ok(tts_ratio_p95, cfg),
    }


def tts_ratio_ok(tts_ratio_p95: float, cfg: Config | None = None) -> bool:
    cfg = cfg or Config()
    return round(tts_ratio_p95, 10) < cfg.TTS_RATIO_MAX


def no_think_check(output: str) -> bool:
    """実出力に思考(<think>)が出ていないこと。"""
    return "<think>" not in output


def resource_check(vram_mib: float, asr_vram_mib: float, ram_mib: float) -> dict:
    """§1.4 資源上限。ちょうど上限は OK、超過は NG。"""
    return {
        "llm_vram": vram_mib <= LLM_VRAM_MAX_MIB,
        "asr_vram": asr_vram_mib <= ASR_VRAM_MAX_MIB,
        "ram": ram_mib <= RAM_MAX_MIB,
    }


def eot_choice(fixed_early_cuts: int, smart_turn_early_cuts: int) -> str:
    """G4 の早切り件数が少ない方。同数は既定の FixedSilence。"""
    return "SmartTurn" if smart_turn_early_cuts < fixed_early_cuts else "FixedSilence"
