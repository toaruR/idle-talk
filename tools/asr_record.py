"""ASR 比較用の録音 (tools/asr_bench.py の入力を作る)。

python tools/asr_record.py [--out data/asr_bench] [--redo s05]
読み上げ文を 1 つずつ表示し、Enter で録音開始、もう一度 Enter で停止して
<out>/<id>.wav (16kHz mono s16le) と <out>/refs.tsv を書く。録音済みの id は飛ばす (中断して再開できる)。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

RATE = 16000
HERE = Path(__file__).resolve().parent


def load_sentences(path: Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sid, _, text = line.partition("\t")
        out.append((sid.strip(), text.strip()))
    return out


def record_one() -> np.ndarray:
    chunks: list[np.ndarray] = []
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        callback=lambda data, *_: chunks.append(data.copy())):
        input("  録音中… 話し終えたら Enter ")
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, np.int16)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="asr_record")
    p.add_argument("--sentences", type=Path, default=HERE / "asr_bench_sentences.txt")
    p.add_argument("--out", type=Path, default=Path("data/asr_bench"))
    p.add_argument("--redo", nargs="*", default=[], help="録り直す id")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    items = load_sentences(args.sentences)
    print(f"マイク: {sd.query_devices(kind='input')['name']}  保存先: {args.out}")
    for i, (sid, text) in enumerate(items, 1):
        wav = args.out / f"{sid}.wav"
        if wav.exists() and sid not in args.redo:
            continue
        hint = text or "(何も言わない。キー入力・咳・机を叩く・無音などの物音だけ)"
        print(f"\n[{i}/{len(items)}] {sid}: {hint}")
        input("  Enter で録音開始 ")
        pcm = record_one()
        sf.write(wav, pcm, RATE, subtype="PCM_16")
        print(f"  保存 {wav.name} ({len(pcm) / RATE:.1f}s)")
    # refs.tsv は文リストの順で毎回書き直す (録音のある id だけ)
    with open(args.out / "refs.tsv", "w", encoding="utf-8", newline="\n") as f:
        for sid, text in items:
            if (args.out / f"{sid}.wav").exists():
                f.write(f"{sid}\t{text}\n")
    print("\n完了。python tools/asr_bench.py で比較できます")
    return 0


if __name__ == "__main__":
    sys.exit(main())
