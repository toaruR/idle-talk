"""faster-whisper (本番と同じ設定) と ReazonSpeech (sherpa-onnx) の CER と遅延を比べる。

python tools/asr_bench.py [--wavs data/asr_bench] [--reazon-dir data/models/reazonspeech] [--threads 2]
入力は tools/asr_record.py が作る <wavs>/<id>.wav と <wavs>/refs.tsv。
- whisper: idle_talk.asr_worker の WhisperModel("small", cuda, float16) + transcribe_pcm (幻聴フィルタ込み)
- reazon : sherpa-onnx OfflineRecognizer (transducer, greedy)。<reazon-dir> の encoder/decoder/joiner*.onnx と tokens.txt
CER は NFKC + 句読点・空白除去 + 漢数字→算用数字の後の文字単位編集距離 / 正解文字数 (全体)。正解が空の録音は「出力文字数」を幻聴として数える。
遅延は 1 クリップの認識呼び出しの壁時計 (最初の 1 回はウォームアップとして別計測)。結果は <wavs>/results.csv にも書く。
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
import time
import unicodedata
from collections.abc import Callable
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RATE = 16000
_DROP = set("、。，．,.!?！？「」『』（）()…・〜~ー-　 ")


_KDIGIT = dict(zip("〇一二三四五六七八九", "0123456789"))
_KUNIT = {"十": 10, "百": 100, "千": 1000}


def _kanji_num(run: str) -> str:
    """「十八」→「18」。万・億は単位のまま残す (Whisper の「100億」と ReazonSpeech の「百億」がそろう)。"""
    if all(c in _KDIGIT for c in run):
        return "".join(_KDIGIT[c] for c in run)
    total, cur = 0, 0
    for c in run:
        if c in _KDIGIT:
            cur = int(_KDIGIT[c])
        else:
            total += (cur or 1) * _KUNIT[c]
            cur = 0
    return str(total + cur)


def normalize(text: str) -> str:
    """NFKC・句読点と空白の除去・漢数字を算用数字に (ReazonSpeech は数字を漢数字で出す)。"""
    t = unicodedata.normalize("NFKC", text)
    t = re.sub(f"[{''.join(_KDIGIT)}{''.join(_KUNIT)}]+", lambda m: _kanji_num(m.group()), t)
    return "".join(c for c in t if c not in _DROP and not c.isspace())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def load_clips(wav_dir: Path) -> list[tuple[str, str, np.ndarray]]:
    clips = []
    for line in (wav_dir / "refs.tsv").read_text(encoding="utf-8").splitlines():
        sid, _, ref = line.partition("\t")
        pcm, rate = sf.read(wav_dir / f"{sid}.wav", dtype="int16")
        if rate != RATE:
            raise SystemExit(f"{sid}.wav: {rate}Hz (16000Hz のみ対応)")
        clips.append((sid, ref, pcm))
    return clips


def make_whisper() -> Callable[[np.ndarray], str]:
    from idle_talk.asr_worker import load_whisper_model, transcribe_pcm

    model = load_whisper_model()
    return lambda pcm: "".join(s["text"] for s in transcribe_pcm(pcm.tobytes(), 0, model))


def make_reazon(model_dir: Path, threads: int) -> Callable[[np.ndarray], str]:
    """本番の asr_worker と同じ読み込み・物音除外・幻聴除外で認識する。"""
    from idle_talk.asr_worker import load_reazon_model, transcribe_pcm
    from idle_talk.config import Asr

    model = load_reazon_model(Asr("reazon", str(model_dir), threads))
    return lambda pcm: "".join(s["text"] for s in transcribe_pcm(pcm.tobytes(), 0, model))


def bench(name: str, load: Callable[[], Callable[[np.ndarray], str]], clips) -> list[dict]:
    t0 = time.perf_counter()
    run = load()
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    run(clips[0][2])
    print(f"\n== {name}: 読み込み {load_s:.1f}s / ウォームアップ {time.perf_counter() - t0:.2f}s")
    rows = []
    for sid, ref, pcm in clips:
        t0 = time.perf_counter()
        hyp = run(pcm)
        ms = (time.perf_counter() - t0) * 1000
        r, h = normalize(ref), normalize(hyp)
        rows.append({"engine": name, "id": sid, "dur_ms": round(len(pcm) / RATE * 1000), "lat_ms": round(ms),
                     "ref": ref, "hyp": hyp, "edits": edit_distance(r, h) if r else 0,
                     "ref_chars": len(r), "noise_chars": 0 if r else len(h)})
        mark = "" if r == h else "  ✗"
        print(f"{sid} {ms:6.0f}ms  {hyp}{mark}")
    return rows


def summary(name: str, rows: list[dict]) -> str:
    speech = [r for r in rows if r["ref_chars"]]
    cer = sum(r["edits"] for r in speech) / max(1, sum(r["ref_chars"] for r in speech))
    lats = sorted(r["lat_ms"] for r in rows)
    p95 = lats[min(len(lats) - 1, round(0.95 * (len(lats) - 1)))]
    rtf = sum(r["lat_ms"] for r in rows) / max(1, sum(r["dur_ms"] for r in rows))
    noise = sum(r["noise_chars"] for r in rows)
    exact = sum(1 for r in speech if r["edits"] == 0)
    return (f"{name:8} CER {cer * 100:5.1f}%  完全一致 {exact}/{len(speech)}  物音での出力 {noise} 字  "
            f"遅延 p50 {statistics.median(lats):.0f}ms p95 {p95}ms  RTF {rtf:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="asr_bench")
    p.add_argument("--wavs", type=Path, default=ROOT / "data/asr_bench")
    p.add_argument("--reazon-dir", type=Path, default=ROOT / "data/models/reazonspeech")
    p.add_argument("--threads", type=int, default=2, help="sherpa-onnx の CPU スレッド数 (話者識別と同じ 2 が既定)")
    p.add_argument("--engines", default="whisper,reazon")
    args = p.parse_args(argv)
    clips = load_clips(args.wavs)
    loaders = {"whisper": make_whisper, "reazon": lambda: make_reazon(args.reazon_dir, args.threads)}
    results = {e: bench(e, loaders[e], clips) for e in args.engines.split(",")}
    with open(args.wavs / "results.csv", "w", encoding="utf-8-sig", newline="") as f:
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
