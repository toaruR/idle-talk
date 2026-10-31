"""Haas-ja-v1.0 を onnxruntime + numpy で動かす ([asr].engine = "haas")。

モデルは tools/haas_export_onnx.py で書き出したもの (<haas_dir>/encoder.onnx, decoder.npz, tokens.txt, config.json)。
NeMo・PyTorch・Triton は使わない。重い FastConformer エンコーダ (609M) だけを onnxruntime で動かし
(GPU なら CUDA か DirectML)、軽い予測ネットワーク (KDA 2 層, 3.9M) と joint (3.0M)・TDT の貪欲デコードは numpy で CPU で回す。

NeMo の推論 (haas_asr の greedy_batch) と同じ計算をなぞる:
- 特徴量: プリエンファシス 0.97 → STFT (n_fft 512, 窓 400 の対称ハン窓, hop 160, center) → パワー → 128 メル → log(x + 2^-24)
  → 特徴ごとに平均 0・標準偏差 1 (不偏, +1e-5) → 長さ外を 0 → 16 フレームの倍数に 0 埋め
- TDT 貪欲: 各時刻で joint を取り、語と飛ばすフレーム数 (durations) を argmax で選ぶ。blank で飛ばす数が 0 なら 1。
  同じ時刻で max_symbols 個出したら 1 フレーム進める (NeMo の label looping と同じ)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

_BYTE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


def _layer_norm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.logaddexp(0.0, x)


class Features:
    """NeMo AudioToMelSpectrogramPreprocessor (推論時) と同じ log-mel 特徴量。"""

    def __init__(self, conf: dict, window: np.ndarray, fb: np.ndarray):
        self.n_fft, self.hop = conf["n_fft"], conf["hop_length"]
        self.preemph, self.guard = conf["preemph"], conf["log_zero_guard"]
        win = np.zeros(self.n_fft, np.float32)
        left = (self.n_fft - len(window)) // 2  # torch.stft は短い窓を n_fft の中央に置く
        win[left:left + len(window)] = window
        self.window = win
        self.fb = fb.reshape(fb.shape[-2], fb.shape[-1]).astype(np.float32)  # [n_mels, n_fft/2+1]

    def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        x = np.asarray(audio, np.float32)
        n = len(x)
        seq_len = n // self.hop  # (n + n_fft//2*2 - n_fft) // hop
        x = np.concatenate([x[:1], x[1:] - self.preemph * x[:-1]]).astype(np.float32)
        pad = self.n_fft // 2
        x = np.pad(x, (pad, pad))
        frames = 1 + (len(x) - self.n_fft) // self.hop
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(frames)[:, None]
        spec = np.fft.rfft(x[idx] * self.window, axis=-1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32).T  # [bins, frames]
        mel = np.log(self.fb @ power + self.guard)
        valid = mel[:, :seq_len]
        mean = valid.mean(1, keepdims=True)
        std = np.sqrt(((valid - mean) ** 2).sum(1, keepdims=True) / max(seq_len - 1, 1)) + 1e-5
        mel = (mel - mean) / std
        mel[:, seq_len:] = 0.0
        if mel.shape[1] % 16:
            mel = np.pad(mel, ((0, 0), (0, 16 - mel.shape[1] % 16)))
        return mel.astype(np.float32), seq_len


class Predictor:
    """予測ネットワーク (KDA 2 層) と joint を numpy で 1 ステップずつ回す (バッチ 1)。"""

    def __init__(self, conf: dict, w: dict):
        self.w = w
        self.layers = conf["pred_layers"]
        self.H, self.K = conf["kda_num_heads"], conf["kda_head_dim"]
        self.W = conf["kda_conv_size"]
        p = "decoder.prediction.dec_rnn."
        self.L = [{k.removeprefix(f"{p}layers.{i}."): v for k, v in w.items() if k.startswith(f"{p}layers.{i}.")}
                  for i in range(self.layers)]
        for i, L in enumerate(self.L):
            L["norm_w"], L["norm_b"] = w[f"{p}norms.{i}.weight"], w[f"{p}norms.{i}.bias"]
            for c in ("q", "k", "v"):
                L[f"{c}_conv"] = L[f"{c}_conv1d.weight"][:, 0, :]  # [D, W]
            L["neg_A"] = -np.exp(L["A_log"])[:, None]
            L["dt_bias2"] = L["dt_bias"].reshape(self.H, self.K)
        self.final_w, self.final_b = w[f"{p}final_norm.weight"], w[f"{p}final_norm.bias"]
        self.embed = w["decoder.prediction.embed.weight"]
        self.pred_w, self.pred_b = w["joint.pred.weight"], w["joint.pred.bias"]
        self.enc_w, self.enc_b = w["joint.enc.weight"], w["joint.enc.bias"]
        self.out_w, self.out_b = w["joint.joint_net.2.weight"], w["joint.joint_net.2.bias"]
        self.scale = self.K ** -0.5

    def initial_state(self):
        D = self.H * self.K
        return [{"q": np.zeros((D, self.W), np.float32), "k": np.zeros((D, self.W), np.float32),
                 "v": np.zeros((D, self.W), np.float32), "S": np.zeros((self.H, self.K, self.K), np.float32)}
                for _ in range(self.layers)]

    def _conv(self, L, c, x, st):
        cache = np.concatenate([st[c][:, 1:], x[:, None]], axis=1)
        return _silu((cache * L[f"{c}_conv"]).sum(1)), cache

    def step(self, label: int, state):
        """label (blank なら開始記号) を入れ、joint 用に射影した出力 [640] と新しい状態を返す。"""
        x = self.embed[label]
        new_state = []
        for L, st in zip(self.L, state):
            h = _layer_norm(x, L["norm_w"], L["norm_b"])
            q, cq = self._conv(L, "q", L["q_proj.weight"] @ h, st)
            k, ck = self._conv(L, "k", L["k_proj.weight"] @ h, st)
            v, cv = self._conv(L, "v", L["v_proj.weight"] @ h, st)
            g = (L["f_proj.1.weight"] @ (L["f_proj.0.weight"] @ h)).reshape(self.H, self.K) + L["dt_bias2"]
            gk = L["neg_A"] * _softplus(g)
            beta = _sigmoid(L["b_proj.weight"] @ h)
            q, k, v = (t.reshape(self.H, self.K) for t in (q, k, v))
            q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * self.scale
            k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
            S = st["S"] * np.exp(gk)[:, :, None]  # [H, K, V]
            vt = (v - (S * k[:, :, None]).sum(1)) * beta[:, None]
            S = S + k[:, :, None] * vt[:, None, :]
            o = (S * q[:, :, None]).sum(1)  # [H, V]
            gate = (L["g_proj.1.weight"] @ (L["g_proj.0.weight"] @ h) + L["g_proj.1.bias"]).reshape(self.H, self.K)
            o = o / np.sqrt((o * o).mean(-1, keepdims=True) + 1e-5) * L["o_norm.weight"] * _sigmoid(gate)
            x = x + L["o_proj.weight"] @ o.reshape(-1)
            new_state.append({"q": cq, "k": ck, "v": cv, "S": S.astype(np.float32)})
        g = _layer_norm(x, self.final_w, self.final_b)
        return (self.pred_w @ g + self.pred_b).astype(np.float32), new_state

    def project_encoder(self, enc: np.ndarray) -> np.ndarray:
        return enc @ self.enc_w.T + self.enc_b  # [T, 640]

    def joint(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        return self.out_w @ np.maximum(f + g, 0.0) + self.out_b


def _providers(gpu: int, available: list[str]):
    """gpu >= 0 なら CUDA、無ければ DirectML、どちらも無ければ CPU。"""
    if gpu >= 0 and "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {"device_id": gpu, "cudnn_conv_algo_search": "HEURISTIC",
                                           "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"]
    if gpu >= 0 and "DmlExecutionProvider" in available:
        return [("DmlExecutionProvider", {"device_id": gpu}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class HaasRecognizer:
    """1 発話 (16kHz mono float32) を文字列にする。"""

    def __init__(self, model_dir: str | Path, gpu: int = 0, threads: int = 2, session=None):
        d = Path(model_dir)
        if not (d / "config.json").is_file():
            raise FileNotFoundError(f"{d} に Haas-ja のモデル (config.json など) がありません")
        self.conf = json.loads((d / "config.json").read_text(encoding="utf-8"))
        w = dict(np.load(d / "decoder.npz"))
        self.features = Features(self.conf, w["preprocessor.featurizer.window"], w["preprocessor.featurizer.fb"])
        self.pred = Predictor(self.conf, w)
        self.pieces = (d / "tokens.txt").read_text(encoding="utf-8").split("\n")[:self.conf["vocab_size"]]
        self.blank = self.conf["blank"]
        self.durations = self.conf["durations"]
        self.max_symbols = self.conf["max_symbols"]
        self.start = self.pred.step(self.blank, self.pred.initial_state())  # 開始記号の出力は毎回同じ
        if session is None:
            import onnxruntime as ort

            providers = _providers(gpu, ort.get_available_providers())
            on_gpu = providers[0] != "CPUExecutionProvider"
            enc = d / ("encoder.fp16.onnx" if on_gpu and (d / "encoder.fp16.onnx").is_file() else "encoder.onnx")
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1 if on_gpu else threads
            so.inter_op_num_threads = 1
            if providers[0][0] == "DmlExecutionProvider":
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(enc), so, providers=providers)
            self.device = f"{session.get_providers()[0]} ({enc.name})"
        else:
            self.device = "injected"
        self.session = session

    def encode(self, audio: np.ndarray) -> np.ndarray:
        mel, n = self.features(audio)
        enc, enc_len = self.session.run(None, {"features": mel[None], "length": np.array([n], np.int64)})
        return enc[0, :, :int(enc_len[0])].T.astype(np.float32)  # [T, 1024]

    def decode(self, enc: np.ndarray) -> list[int]:
        f = self.pred.project_encoder(enc)
        T, nd = len(f), len(self.durations)
        g, state = self.start
        tokens, t, last_t, run = [], 0, -1, 0
        while t < T:
            out = self.pred.joint(f[t], g)
            k = int(out[:-nd].argmax())
            d = self.durations[int(out[-nd:].argmax())]
            if k == self.blank:
                t += max(d, 1)
                continue
            tokens.append(k)
            run = run + 1 if t == last_t else 1
            last_t = t
            t += d
            g, state = self.pred.step(k, state)
            if run >= self.max_symbols and t == last_t:
                t += 1
        return tokens

    def detokenize(self, ids: list[int]) -> str:
        buf, out = bytearray(), []
        for i in ids:
            m = _BYTE.fullmatch(self.pieces[i])
            if m:
                buf.append(int(m.group(1), 16))
                continue
            if buf:
                out.append(buf.decode("utf-8", errors="replace"))
                buf.clear()
            out.append(self.pieces[i])
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out).replace("▁", " ").strip()

    def __call__(self, audio: np.ndarray) -> str:
        if len(audio) < self.features.hop * 2:
            return ""
        return self.detokenize(self.decode(self.encode(audio)))
