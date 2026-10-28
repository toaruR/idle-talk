"""Moonshine (moonshine-voice、日本語 Small Streaming) と ReazonSpeech の CER と遅延を比べる (asr_bench.py と同じ 80 クリップ)。

python tools/asr_bench_moonshine.py [--wavs data/asr_bench_tv] [--model-dir data/models/moonshine] [--reazon-dir ...]
- reazon        : asr_bench.make_reazon (本番の asr_worker と同じ)。遅延は区間の終わりから結果まで
- moonshine     : ストリーミング。音を 100ms ずつ実時間の速さで渡したと想定して add_audio し (待たずに流す)、
                  終わりで stop。遅延は最後の音を渡してから最終結果まで (stop の時間)。最初の文字が出るまでの
                  音声上の時刻 (first_ms) と、途中経過が書き換わった回数も測る
- moonshine-off : transcribe_without_streaming (区間全体を一括)
CER の正規化は asr_bench.normalize。結果は <wavs>/results_moonshine.csv。
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_bench import RATE, ROOT, edit_distance, load_clips, make_reazon, normalize  # noqa: E402

CHUNK = RATE // 10  # 100ms
TAIL = np.zeros(RATE // 2, dtype=np.int16)  # 末尾に足す無音 (--tail)。クリップは発話の直後で切れており、先読みのあるストリーミングは最後の語を落とす
OPTIONS = {"max_tokens_per_second": 13.0}  # 日本語など非ラテン文字は 13 にしないと出力が切れる (Moonshine の資料)


def load_moonshine(model_dir: Path):
    from moonshine_voice import get_model_for_language
    from moonshine_voice.moonshine_api import ModelArch
    from moonshine_voice.transcriber import Transcriber

    path, arch = get_model_for_language("ja", ModelArch.SMALL_STREAMING, cache_root=model_dir)
    return Transcriber(path, arch, update_interval=0.5, options=OPTIONS)


def _text(transcript) -> str:
    return "".join(line.text for line in transcript.lines)


def run_stream(tr, pcm: np.ndarray) -> dict:
    audio = (pcm.astype(np.float32) / 32768).tolist()
    stream = tr.create_stream(update_interval=0.5)
    seen: list[str] = []
    first_ms = None

    def on_event(ev) -> None:
        nonlocal first_ms
        text = getattr(getattr(ev, "line", None), "text", "") or ""
        if text and (not seen or seen[-1] != text):
            seen.append(text)
            if first_ms is None:
                first_ms = pos * 1000 // RATE
    stream.add_listener(on_event)
    stream.start()
    t_total = time.perf_counter()
    pos = 0
    for pos in range(0, len(audio), CHUNK):
        stream.add_audio(audio[pos:pos + CHUNK], RATE)
    pos = len(audio)
    t0 = time.perf_counter()
    final = stream.stop()
    final_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000
    stream.remove_all_listeners()
    return {"hyp": _text(final) if final else (seen[-1] if seen else ""), "lat_ms": round(final_ms),
            "total_ms": round(total_ms), "first_ms": first_ms, "revisions": max(0, len(seen) - 1)}


def run_offline(tr, pcm: np.ndarray) -> dict:
    t0 = time.perf_counter()
    hyp = _text(tr.transcribe_without_streaming((pcm.astype(np.float32) / 32768).tolist(), RATE))
    ms = (time.perf_counter() - t0) * 1000
    return {"hyp": hyp, "lat_ms": round(ms), "total_ms": round(ms), "first_ms": None, "revisions": 0}


def bench(name: str, run, clips) -> list[dict]:
    run(clips[0][2])  # ウォームアップ
    print(f"\n== {name}")
    rows = []
    for sid, ref, pcm in clips:
        r = run(pcm)
        ref_n, hyp_n = normalize(ref), normalize(r["hyp"])
        rows.append({"engine": name, "id": sid, "dur_ms": round(len(pcm) / RATE * 1000), **r, "ref": ref,
                     "edits": edit_distance(ref_n, hyp_n) if ref_n else 0, "ref_chars": len(ref_n),
                     "noise_chars": 0 if ref_n else len(hyp_n)})
        print(f"{sid} {r['lat_ms']:6d}ms  {r['hyp']}{'' if ref_n == hyp_n else '  ✗'}")
    return rows


def summary(name: str, rows: list[dict]) -> str:
    speech = [r for r in rows if r["ref_chars"]]
    cer = sum(r["edits"] for r in speech) / max(1, sum(r["ref_chars"] for r in speech))
    lats = sorted(r["lat_ms"] for r in rows)
    p95 = lats[min(len(lats) - 1, round(0.95 * (len(lats) - 1)))]
    rtf = sum(r["total_ms"] for r in rows) / max(1, sum(r["dur_ms"] for r in rows))
    exact = sum(1 for r in speech if r["edits"] == 0)
    firsts = [r["first_ms"] for r in rows if r["first_ms"] is not None]
    extra = (f"  最初の文字 p50 {statistics.median(firsts):.0f}ms (音声上)  書き換え平均 "
             f"{statistics.mean(r['revisions'] for r in rows):.1f} 回" if firsts else "")
    return (f"{name:13} CER {cer * 100:5.1f}%  完全一致 {exact}/{len(speech)}  物音での出力 "
            f"{sum(r['noise_chars'] for r in rows)} 字  確定までの遅延 p50 {statistics.median(lats):.0f}ms "
            f"p95 {p95}ms  RTF {rtf:.3f}{extra}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="asr_bench_moonshine")
    p.add_argument("--wavs", type=Path, default=ROOT / "data/asr_bench_tv")
    p.add_argument("--model-dir", type=Path, default=ROOT / "data/models/moonshine")
    p.add_argument("--reazon-dir", type=Path, default=ROOT / "data/models/reazonspeech")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--engines", default="reazon,moonshine,moonshine-off")
    p.add_argument("--tail", action="store_true", help="moonshine に渡す音の末尾に 0.5 秒の無音を足す (実運用では音が続く)")
    args = p.parse_args(argv)
    clips = load_clips(args.wavs)
    results = {}
    tr = None
    for e in args.engines.split(","):
        if e == "reazon":
            rz = make_reazon(args.reazon_dir, args.threads)

            def run_rz(pcm, rz=rz):
                t0 = time.perf_counter()
                hyp = rz(pcm)
                ms = round((time.perf_counter() - t0) * 1000)
                return {"hyp": hyp, "lat_ms": ms, "total_ms": ms, "first_ms": None, "revisions": 0}
            results[e] = bench(e, run_rz, clips)
            continue
        tr = tr or load_moonshine(args.model_dir)
        pad = (lambda pcm: np.concatenate([pcm, TAIL])) if args.tail else (lambda pcm: pcm)
        results[e] = bench(e, (lambda pcm: run_stream(tr, pad(pcm))) if e == "moonshine" else
                           (lambda pcm: run_offline(tr, pad(pcm))), clips)
    with open(args.wavs / "results_moonshine.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(results.values()))[0]))
        w.writeheader()
        for rows in results.values():
            w.writerows(rows)
    print(f"\n== 集計 ({len(clips)} クリップ)")
    for e, rows in results.items():
        print(summary(e, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
