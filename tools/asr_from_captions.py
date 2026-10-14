"""録画 (.ts) の放送字幕から ASR 比較用のクリップを作る (tools/asr_bench.py の入力)。

python tools/asr_from_captions.py <録画.ts> [--out data/asr_bench_tv] [--max-clips 80]
ffmpeg で字幕 (arib_caption) を SRT に、音声を 16kHz mono に取り出し、近い字幕をまとめて 1 クリップにする。
<out>/<id>.wav と <out>/refs.tsv を書く。字幕は要約や表示の遅れがあるので CER は絶対値ではなく 2 エンジンの比較に使う。
録画の字幕は終了時刻が壊れていることがある (1193 時間など) ので、終了は「次の字幕の開始」で上限を切る。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 16000
GAP_S = 3.5      # 前の字幕の開始からこれ以内に次が出たら同じクリップにまとめる
MAX_CLIP_S = 12  # まとめる上限 (本番の区間長に近づける)
MAX_CAP_S = 6    # 1 字幕の長さの上限 (最後の字幕の終了が壊れているとき用)
PAD_S = 0.3

_TS = re.compile(r"(\d+):(\d\d):(\d\d),(\d+) --> (\d+):(\d\d):(\d\d),(\d+)")


def _sec(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def clean(text: str) -> str:
    """タグ・{\\an7}・ナレーションの <> 括弧・話者名 (…)・♪ を除く。"""
    t = re.sub(r"<font[^>]*>|</font>|\{[^}]*\}", "", text)
    t = re.sub(r"[(（][^)）]*[)）]", "", t)
    t = t.replace("♪", "").replace("<", "").replace(">", "").replace("〈", "").replace("〉", "")
    return re.sub(r"\s+", " ", t).strip()


def parse_srt(srt: str) -> list[tuple[float, float, str]]:
    caps = []
    for block in re.split(r"\n\s*\n", srt.replace("\r", "")):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _TS.search(line)
            if m:
                text = clean(" ".join(lines[i + 1:]))
                if text:
                    caps.append((_sec(*m.groups()[:4]), _sec(*m.groups()[4:]), text))
                break
    caps.sort()
    # 終了は「次の字幕の開始」と MAX_CAP_S で切る
    return [(s, min(e, caps[i + 1][0] if i + 1 < len(caps) else e, s + MAX_CAP_S), t)
            for i, (s, e, t) in enumerate(caps)]


def group(caps: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    out: list[list] = []
    for s, e, t in caps:
        if out and s - out[-1][3] <= GAP_S and e - out[-1][0] <= MAX_CLIP_S:
            out[-1][1], out[-1][3] = e, s
            out[-1][2] += t
        else:
            out.append([s, e, t, s])  # [開始, 終了, 文, 最後の字幕の開始]
    return [(s, e, t) for s, e, t, _ in out]


def run_ffmpeg(args: list[str]) -> bytes:
    r = subprocess.run(["ffmpeg", "-v", "error", "-y", *args], capture_output=True, check=False)
    if r.returncode:
        raise SystemExit(r.stderr.decode("utf-8", "replace")[-500:])
    return r.stdout


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="asr_from_captions")
    p.add_argument("video", type=Path)
    p.add_argument("--out", type=Path, default=Path("data/asr_bench_tv"))
    p.add_argument("--max-clips", type=int, default=80, help="番組全体から等間隔に選ぶ数 (0 で全部)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        srt_path = Path(tmp) / "cap.srt"
        run_ffmpeg(["-i", str(args.video), "-map", "0:s:0", "-c:s", "srt", str(srt_path)])
        srt = srt_path.read_text(encoding="utf-8", errors="replace")
    pcm = np.frombuffer(run_ffmpeg(["-i", str(args.video), "-map", "0:a:0", "-ac", "1", "-ar", str(RATE),
                                    "-f", "s16le", "-"]), dtype=np.int16)
    clips = group(parse_srt(srt))
    if args.max_clips and len(clips) > args.max_clips:
        step = len(clips) / args.max_clips
        clips = [clips[int(i * step)] for i in range(args.max_clips)]
    with open(args.out / "refs.tsv", "w", encoding="utf-8", newline="\n") as f:
        for i, (s, e, text) in enumerate(clips, 1):
            sid = f"tv{i:03d}"
            a, b = max(0, int((s - PAD_S) * RATE)), min(len(pcm), int((e + PAD_S) * RATE))
            sf.write(args.out / f"{sid}.wav", pcm[a:b], RATE, subtype="PCM_16")
            f.write(f"{sid}\t{text}\n")
    print(f"{len(clips)} クリップ ({sum(e - s for s, e, _ in clips) / 60:.1f} 分) を {args.out} に書きました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
