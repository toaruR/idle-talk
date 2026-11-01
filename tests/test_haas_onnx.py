"""idle_talk.haas_onnx の試験。実モデルは使わず、デコードの規則・語の復元・特徴量の形・GPU の選び方を検証する。

実モデルでの一致 (NeMo と同じ認識結果) は tools/asr_bench_ja.py --engine haas_onnx で確かめる (docs/asr-comparison-haas-ja.md)。
"""
from types import SimpleNamespace

import numpy as np

from idle_talk import haas_onnx
from idle_talk.haas_onnx import Features, HaasRecognizer, _providers

BLANK = 4


class FakePred:
    """joint の出力を台本どおりに返す。step で受けた語を記録する。"""

    def __init__(self, script):
        self.script = list(script)  # (label, duration index) を順に返す
        self.steps = []

    def project_encoder(self, enc):
        return enc

    def joint(self, f, g):
        label, dur = self.script.pop(0)
        out = np.full(BLANK + 1 + 3, -1.0, np.float32)
        out[label] = 1.0
        out[BLANK + 1 + dur] = 1.0
        return out

    def step(self, label, state):
        self.steps.append(label)
        return np.zeros(1), state


def recognizer(script, pieces=("a", "b", "c", "d"), max_symbols=10):
    r = HaasRecognizer.__new__(HaasRecognizer)
    r.pred, r.blank, r.durations, r.max_symbols = FakePred(script), BLANK, [0, 1, 2], max_symbols
    r.start, r.pieces = (np.zeros(1), None), list(pieces)
    return r


def test_tdt_greedy_follows_durations_and_blank_advances_at_least_one():
    # t=0: blank (dur 0 → 1)、t=1: 語0 dur0 (同じ時刻にとどまる)、t=1: 語1 dur2、t=3: blank dur1、t=4: 終わり (T=4)
    r = recognizer([(BLANK, 0), (0, 0), (1, 2), (BLANK, 1)])
    assert r.decode(np.zeros((4, 1))) == [0, 1] and r.pred.steps == [0, 1]


def test_tdt_greedy_forces_advance_after_max_symbols():
    r = recognizer([(2, 0), (2, 0), (3, 1)], max_symbols=2)
    assert r.decode(np.zeros((2, 1))) == [2, 2, 3]


def test_detokenize_joins_byte_fallback_and_word_boundaries():
    r = recognizer([], pieces=["\u2581こん", "<0xE3>", "<0x81>", "<0xAB>", "ちは", "\u2581ok"])
    assert r.detokenize([0, 1, 2, 3, 4, 5]) == "こんにちは ok"


def test_features_shape_padding_and_normalization():
    conf = {"n_fft": 512, "hop_length": 160, "preemph": 0.97, "log_zero_guard": 2 ** -24}
    rng = np.random.default_rng(0)
    f = Features(conf, np.hanning(400).astype(np.float32), rng.random((1, 128, 257)).astype(np.float32))
    mel, n = f(rng.standard_normal(16000).astype(np.float32))
    assert n == 100 and mel.shape == (128, 112)  # 101 フレーム → 16 の倍数に 0 埋め
    assert np.allclose(mel[:, :n].mean(1), 0, atol=1e-4) and np.all(mel[:, n:] == 0)


def test_providers_prefer_cuda_then_directml_then_cpu():
    assert _providers(1, ["CUDAExecutionProvider", "CPUExecutionProvider"])[0][1]["device_id"] == 1
    assert _providers(0, ["DmlExecutionProvider", "CPUExecutionProvider"])[0][0] == "DmlExecutionProvider"
    assert _providers(0, ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert _providers(-1, ["CUDAExecutionProvider", "CPUExecutionProvider"]) == ["CPUExecutionProvider"]


def test_kda_step_matches_reference_recurrence():
    """Predictor の KDA 1 ステップを、素直に書いた式 (fla の naive_recurrent_kda と同じ形) と比べる。"""
    H, K, D = 2, 4, 6
    rng = np.random.default_rng(1)
    p = "decoder.prediction.dec_rnn."
    w = {"decoder.prediction.embed.weight": rng.standard_normal((5, D)).astype(np.float32),
         f"{p}final_norm.weight": np.ones(D, np.float32), f"{p}final_norm.bias": np.zeros(D, np.float32),
         f"{p}norms.0.weight": np.ones(D, np.float32), f"{p}norms.0.bias": np.zeros(D, np.float32),
         "joint.pred.weight": np.eye(D, dtype=np.float32), "joint.pred.bias": np.zeros(D, np.float32),
         "joint.enc.weight": np.eye(D, dtype=np.float32), "joint.enc.bias": np.zeros(D, np.float32),
         "joint.joint_net.2.weight": np.eye(D, dtype=np.float32), "joint.joint_net.2.bias": np.zeros(D, np.float32)}
    L = f"{p}layers.0."
    shapes = {"A_log": (H,), "dt_bias": (H * K,), "q_proj.weight": (H * K, D), "k_proj.weight": (H * K, D),
              "v_proj.weight": (H * K, D), "q_conv1d.weight": (H * K, 1, 4), "k_conv1d.weight": (H * K, 1, 4),
              "v_conv1d.weight": (H * K, 1, 4), "f_proj.0.weight": (K, D), "f_proj.1.weight": (H * K, K),
              "b_proj.weight": (H, D), "g_proj.0.weight": (K, D), "g_proj.1.weight": (H * K, K),
              "g_proj.1.bias": (H * K,), "o_norm.weight": (K,), "o_proj.weight": (D, H * K)}
    for k, s in shapes.items():
        w[L + k] = (rng.standard_normal(s) * 0.5).astype(np.float32)
    conf = {"pred_layers": 1, "kda_num_heads": H, "kda_head_dim": K, "kda_conv_size": 4}
    pred = haas_onnx.Predictor(conf, w)
    state = pred.initial_state()
    S_ref = np.zeros((H, K, K))
    for label in (4, 1, 2):
        out, state = pred.step(label, state)
    # 参照: 同じ 3 ステップを素直に計算し直して最後の状態を比べる
    x_hist = {c: [] for c in "qkv"}
    for label in (4, 1, 2):
        x = w["decoder.prediction.embed.weight"][label]
        h = (x - x.mean()) / np.sqrt(x.var() + 1e-5)
        cur = {}
        for c in "qkv":
            x_hist[c].append(w[L + f"{c}_proj.weight"] @ h)
            hist = ([np.zeros(H * K)] * 4 + x_hist[c])[-4:]
            y = sum(w[L + f"{c}_conv1d.weight"][:, 0, j] * hist[j] for j in range(4))
            cur[c] = (y / (1 + np.exp(-y))).reshape(H, K)
        g = (w[L + "f_proj.1.weight"] @ (w[L + "f_proj.0.weight"] @ h) + w[L + "dt_bias"]).reshape(H, K)
        gk = -np.exp(w[L + "A_log"])[:, None] * np.log1p(np.exp(g))
        beta = 1 / (1 + np.exp(-(w[L + "b_proj.weight"] @ h)))
        q = cur["q"] / np.linalg.norm(cur["q"], axis=-1, keepdims=True) * K ** -0.5
        k = cur["k"] / np.linalg.norm(cur["k"], axis=-1, keepdims=True)
        for hd in range(H):
            S_ref[hd] = np.diag(np.exp(gk[hd])) @ S_ref[hd]
            S_ref[hd] += beta[hd] * np.outer(k[hd], cur["v"][hd] - S_ref[hd].T @ k[hd])
    np.testing.assert_allclose(state[0]["S"], S_ref, rtol=1e-3, atol=1e-4)
