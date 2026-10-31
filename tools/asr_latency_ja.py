"""ReazonSpeech と Haas-ja の 1 発話ずつの認識時間と、音声でない音への反応を比べる (他の処理を止めて実行する)。

python tools/asr_latency_ja.py --bench <asr-ja-bench dir> [--n 60] [--threads 2] [--reazon-dir ...] [--haas-device cpu]

- 認識時間: jsut / jvnv / fleurs から --n 件ずつ (先頭から等間隔) を 1 件ずつ認識し、壁時計の中央値・p90 と RTF を出す。
  最初の 3 件はウォームアップとして捨てる。
- 非音声: 無音・白色雑音・ピンク雑音・短いクリック列 (食器の音のつもり) を合成して認識させ、出た文字列を出す。
  idle-talk は ReazonSpeech が物音を「はい」「誰」と聞き違えるのを [asr].short_chars で抑えている。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RATE = 16000


def synth_noises(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 2 * RATE
    white = rng.standard_normal(n).astype(np.float32) * 0.05
    pink = np.cumsum(rng.standard_normal(n)).astype(np.float32)
    pink = (pink - np.convolve(pink, np.ones(400) / 400, mode="same")) * 0.01
    clicks = np.zeros(n, np.float32)
    for t in rng.integers(0, n - 800, 6):
        clicks[t:t + 800] += np.exp(-np.arange(800) / 120) * np.sin(2 * np.pi * rng.uniform(1500, 4000) * np.arange(800) / RATE) * 0.6
    hum = (np.sin(2 * np.pi * 120 * np.arange(n) / RATE) * 0.1).astype(np.float32)
    return {"silence": np.zeros(n, np.float32), "white": white, "pink": pink.astype(np.float32),
            "clicks": clicks.astype(np.float32), "hum120": hum}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--reazon-dir", type=Path, default=ROOT / "data/models/reazonspeech")
    ap.add_argument("--haas-device", default="cpu")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from asr_bench_ja import haas_engine, reazon_engine

    clips = []
    for subset in ("jsut", "jvnv", "fleurs"):
        rows = [json.loads(x) for x in (args.bench / f"{subset}.jsonl").read_text(encoding="utf-8").splitlines() if x]
        step = max(1, len(rows) // args.n)
        clips += [sf.read(args.bench / r["audio"], dtype="float32")[0] for r in rows[::step][:args.n]]

    ns = argparse.Namespace(reazon_dir=args.reazon_dir, threads=args.threads, device=args.haas_device,
                            haas_model="third-intelligence/Haas-ja-v1.0", batch_size=1)
    report = {}
    for name, factory in (("reazon", reazon_engine), ("haas", haas_engine)):
        run, _ = factory(ns)
        for c in clips[:3]:
            run([c])
        secs = []
        for c in clips:
            t0 = time.perf_counter()
            run([c])
            secs.append(time.perf_counter() - t0)
        audio_s = sum(len(c) for c in clips) / RATE
        noises = {k: run([v])[0][0] for k, v in synth_noises().items()}
        report[name] = {"clips": len(clips), "sec_median": statistics.median(secs),
                        "sec_p90": statistics.quantiles(secs, n=10)[-1], "rtf": sum(secs) / audio_s,
                        "noise": noises}
        print(name, json.dumps(report[name], ensure_ascii=False), flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
