"""ReazonSpeech (idle-talk の既定) と Haas-ja-v1.0 を公開ベンチマーク asr-ja-bench で比べる。

python tools/asr_bench_ja.py --bench <dir> --engine reazon [--reazon-dir data/models/reazonspeech] [--threads 2]
python tools/asr_bench_ja.py --bench <dir> --engine haas   [--device cuda|cpu] [--batch-size 16]

<dir> は asr-ja-bench (https://huggingface.co/datasets/third-intelligence/asr-ja-bench) の rebuild.py が
書き出すディレクトリ (<subset>.jsonl と audio/)。Haas-ja のモデルカードと同じ jacer で CER strict / flex / kana を出す
(サブセット内はマイクロ平均、全体は音声の長さで重み付け)。

- reazon: idle-talk の asr_worker と同じ読み込み (int8 があれば int8、greedy) で 1 発話ずつ認識する。
  1 発話の認識時間 (壁時計) の中央値と p90 も出す。idle-talk は 1 発話ずつ呼ぶので、これが体感の遅延に近い。
- haas: pip install "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git" と jacer が要る。
  公式は NVIDIA GPU 専用。--device cpu は TRITON_INTERPRET=1 で予測ネットワークの Triton カーネルを CPU で解釈実行する
  (認識結果は同じだが遅い。精度を確かめる用)。

--limit N で各サブセットの先頭 N 件だけ使う。結果は --out/<engine>/ に <subset>.jsonl と summary.json として書く。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SUBSETS = ("jsut", "jvnv", "tedxjp10k", "fleurs", "speech-bsd")
RATE = 16000


def read_subset(bench: Path, subset: str, limit: int | None) -> list[dict]:
    with open(bench / f"{subset}.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return rows[:limit] if limit else rows


def load_audio(bench: Path, row: dict) -> np.ndarray:
    x, sr = sf.read(bench / row["audio"], dtype="float32")
    if sr != RATE:
        raise SystemExit(f"{row['audio']}: {sr} Hz, expected {RATE}")
    return np.ascontiguousarray(x)


def reazon_engine(args):
    """idle_talk.asr_worker.load_reazon_model と同じ設定の recognizer で 1 発話ずつ認識する関数を返す。"""
    sys.path.insert(0, str(ROOT / "src"))
    from idle_talk.asr_worker import load_reazon_model
    from idle_talk.config import Asr

    model = load_reazon_model(Asr(engine="reazon", model_dir=str(args.reazon_dir), threads=args.threads))

    def run(audios: list[np.ndarray]) -> tuple[list[str], list[float]]:
        texts, secs = [], []
        for a in audios:
            t0 = time.perf_counter()
            s = model.rec.create_stream()  # 本番の ReazonModel.transcribe と違い、物音フィルタを通さない生の出力
            s.accept_waveform(RATE, a)
            model.rec.decode_stream(s)
            secs.append(time.perf_counter() - t0)
            texts.append(s.result.text.strip())
        return texts, secs

    return run, 1


def haas_engine(args):
    if args.device == "cpu":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import haas_cpu_patch

        haas_cpu_patch.apply()
    import torch

    import haas_asr

    torch.set_num_threads(args.threads)
    model = haas_asr.load_model(args.haas_model, device=args.device)

    def run(audios: list[np.ndarray]) -> tuple[list[str], list[float]]:
        t0 = time.perf_counter()
        out = model.transcribe(audios, batch_size=len(audios), verbose=False)
        dt = time.perf_counter() - t0
        return [h.text.strip() for h in out], [dt / len(audios)] * len(audios)

    return run, args.batch_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--engine", choices=("reazon", "haas"), required=True)
    ap.add_argument("--subset", action="append", choices=SUBSETS)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path, default=Path("data/asr_bench_ja"))
    ap.add_argument("--reazon-dir", type=Path, default=ROOT / "data/models/reazonspeech")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--haas-model", default="third-intelligence/Haas-ja-v1.0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    import jacer

    subsets = tuple(args.subset or [s for s in SUBSETS if (args.bench / f"{s}.jsonl").is_file()])
    run, batch = (reazon_engine if args.engine == "reazon" else haas_engine)(args)
    out = args.out / args.engine
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for subset in subsets:
        rows = read_subset(args.bench, subset, args.limit)
        hyps, secs = [], []
        for i in range(0, len(rows), batch):
            h, s = run([load_audio(args.bench, r) for r in rows[i:i + batch]])
            hyps += h
            secs += s
            print(f"\r{subset} {len(hyps)}/{len(rows)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        cer = jacer.cer([r["text"] for r in rows], hyps, jacer.METRICS)
        hours = sum(r["duration"] for r in rows) / 3600
        rtf = sum(secs) / (hours * 3600)
        results[subset] = {"samples": len(rows), "hours": hours, **cer, "rtf": rtf,
                           "sec_median": statistics.median(secs),
                           "sec_p90": statistics.quantiles(secs, n=10)[-1] if len(secs) > 1 else secs[0]}
        with open(out / f"{subset}.jsonl", "w", encoding="utf-8") as f:
            for r, h in zip(rows, hyps, strict=True):
                f.write(json.dumps({"id": r["id"], "reference": r["text"], "hypothesis": h},
                                   ensure_ascii=False) + "\n")
        print(f"{subset:11s} {len(rows):5d}  " + "  ".join(f"{m} {cer[m] * 100:5.2f}" for m in jacer.METRICS)
              + f"  RTF {rtf:.3f}", flush=True)

    total = sum(r["hours"] for r in results.values())
    weighted = {m: sum(r[m] * r["hours"] for r in results.values()) / total for m in jacer.METRICS}
    print("weighted    " + "  ".join(f"{m} {weighted[m] * 100:5.2f}" for m in jacer.METRICS))
    (out / "summary.json").write_text(json.dumps(
        {"engine": args.engine, "args": {k: str(v) for k, v in vars(args).items()}, "subsets": results,
         "weighted": weighted, "metrics_version": jacer.METRICS_VERSION}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
