"""Haas-ja-v1.0 (.nemo) を idle-talk の [asr].engine = "haas" 用に書き出す (1 回だけ、Linux / WSL2 で実行)。

python tools/haas_export_onnx.py --out data/models/haas-ja [--nemo Haas-ja-v1.0.nemo] [--fp16]

要るもの: pip install "nemo-toolkit[asr]>=2.7.3,<2.8" onnx (GPU も flash-linear-attention も要らない)。
--nemo を省くと Hugging Face (third-intelligence/Haas-ja-v1.0) から取ってくる。

書き出すもの (<out>/):
- encoder.onnx (+ encoder.onnx.data): FastConformer エンコーダ (609M)。入力 features [1, 128, T] / length [1]
- encoder.fp16.onnx (+ .data): --fp16 のとき。GPU 用 (VRAM と計算が半分)。入出力は float32 のまま
- decoder.npz: 予測ネットワーク (KDA) と joint の重み、メルフィルタと窓 (すべて float32)
- tokens.txt: SentencePiece の語彙 (1 行 1 語、行番号が ID)
- config.json: 特徴量・デコードの設定
実行時 (src/idle_talk/haas_onnx.py) は onnxruntime と numpy だけで動く。

Windows の idle-talk に持っていくときは <out> ごとコピーし、idle_talk.toml の [asr].haas_dir に置き場所を書く。
"""
from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

REPO = "third-intelligence/Haas-ja-v1.0"


def read_nemo(path: Path, work: Path) -> tuple[dict, dict, list[str]]:
    import sentencepiece as spm
    import torch
    import yaml

    with tarfile.open(path) as tf:
        tf.extractall(work, filter="data")
    cfg = yaml.safe_load((work / "model_config.yaml").read_text(encoding="utf-8"))
    cfg = cfg.get("model", cfg)
    sd = torch.load(work / "model_weights.ckpt", map_location="cpu", weights_only=True)
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    sp = spm.SentencePieceProcessor(model_file=str(next(work.glob("*tokenizer.model"))))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    return cfg, sd, pieces


def export_encoder(cfg: dict, sd: dict, out: Path) -> None:
    import torch
    from nemo.collections.asr.modules import ConformerEncoder

    enc_cfg = {k: v for k, v in cfg["encoder"].items() if k != "_target_"}
    enc = ConformerEncoder(**enc_cfg)
    missing, unexpected = enc.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in sd.items() if k.startswith("encoder.")}, strict=True)
    enc.eval()
    enc.freeze()
    enc._prepare_for_export()  # NeMo の書き出し用の切り替え (位置埋め込みの事前計算など)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, features, length):
            y, n = self.m(audio_signal=features, length=length)
            return y, n

    import onnx

    feats = torch.randn(1, cfg["encoder"]["feat_in"], 800)
    length = torch.tensor([800], dtype=torch.int64)
    # 2GB を超えるので torch は重みを個別ファイルに分けて書く。一時ディレクトリに書いてから 1 ファイルにまとめる
    with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
        tmp_path = Path(tmp) / "encoder.onnx"
        torch.onnx.export(Wrap(enc), (feats, length), str(tmp_path), input_names=["features", "length"],
                          output_names=["encoded", "encoded_length"], opset_version=17, dynamo=False,
                          dynamic_axes={"features": {0: "batch", 2: "frames"}, "length": {0: "batch"},
                                        "encoded": {0: "batch", 2: "steps"}, "encoded_length": {0: "batch"}})
        model = onnx.load(str(tmp_path))
        onnx.save_model(model, str(out / "encoder.onnx"), save_as_external_data=True,
                        all_tensors_to_one_file=True, location="encoder.onnx.data", size_threshold=1024)


def to_fp16(out: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(out / "encoder.onnx"))
    half = float16.convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=True)
    onnx.save_model(half, str(out / "encoder.fp16.onnx"), save_as_external_data=True,
                    all_tensors_to_one_file=True, location="encoder.fp16.onnx.data", size_threshold=1024)


def export_decoder(cfg: dict, sd: dict, pieces: list[str], out: Path) -> None:
    keep = ("decoder.", "joint.", "preprocessor.featurizer.window", "preprocessor.featurizer.fb")
    arrays = {k: v.numpy().astype(np.float32) for k, v in sd.items()
              if k.startswith(keep) and not k.endswith("num_batches_tracked")}
    np.savez(out / "decoder.npz", **arrays)
    (out / "tokens.txt").write_text("\n".join(pieces) + "\n", encoding="utf-8")
    pre, dec, joint = cfg["preprocessor"], cfg["decoder"], cfg["joint"]
    p = dec["prednet"]
    conf = {
        "model": REPO,
        "sample_rate": pre["sample_rate"],
        "n_fft": pre["n_fft"],
        "win_length": round(pre["window_size"] * pre["sample_rate"]),
        "hop_length": round(pre["window_stride"] * pre["sample_rate"]),
        "features": pre["features"],
        "preemph": pre.get("preemph", 0.97),
        "log_zero_guard": 2 ** -24,
        "normalize": pre["normalize"],
        "subsampling_factor": cfg["encoder"]["subsampling_factor"],
        "vocab_size": dec["vocab_size"],
        "blank": dec["vocab_size"],
        "durations": cfg["decoding"]["durations"],
        "max_symbols": cfg["decoding"]["greedy"]["max_symbols"],
        "pred_hidden": p["pred_hidden"],
        "pred_layers": p["pred_rnn_layers"],
        "kda_num_heads": p["kda_num_heads"],
        "kda_head_dim": p["kda_head_dim"],
        "kda_conv_size": p["kda_conv_size"],
        "joint_activation": joint["jointnet"]["activation"],
    }
    (out / "config.json").write_text(json.dumps(conf, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nemo", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fp16", action="store_true", help="GPU 用の encoder.fp16.onnx も書く")
    args = ap.parse_args(argv)
    nemo = args.nemo
    if nemo is None:
        from huggingface_hub import hf_hub_download

        nemo = Path(hf_hub_download(REPO, "Haas-ja-v1.0.nemo"))
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cfg, sd, pieces = read_nemo(nemo, Path(tmp))
    export_decoder(cfg, sd, pieces, args.out)
    print("decoder.npz / tokens.txt / config.json", flush=True)
    export_encoder(cfg, sd, args.out)
    print("encoder.onnx", flush=True)
    if args.fp16:
        to_fp16(args.out)
        print("encoder.fp16.onnx", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
