"""文字起こしモードの確定のしかたを比べる。テレビの 80 クリップを BGM つきで途切れなくつなげ、実時間で流したときと同じ順に
ReazonSpeech を呼んで、全体の CER を出す (ASR の計算時間はかからないものとして、音の位置だけで進める)。

python tools/stream_bench.py [--wavs data/asr_bench_tv] [--gap-ms 200] [--bgm 300] [--methods segment,agree]
- segment: 今の方式。VAD の区間を 1 秒ごとに認識し直して仮の文字、区間の終わり (または 8 秒で直近 2 秒の一番静かな所)
           で区間全体を認識し直して確定し、最後の仮の文字と merge_final で合わせる
- agree  : 安定した部分から確定する方式。1 秒ごとに溜めた音を認識し、前回と今回で頭が一致した文字 (2 回続けて同じ) を
           確定して、その次の文字の時刻より前の音を捨てる。区間の終わりで残りを認識して確定する。
           溜めた音が FORCE_S を超えたら、最後の FORCE_KEEP_S より前に始まった文字を確定する
CER の正規化は asr_bench.normalize (句読点・空白・漢数字)。正解は refs.tsv をつなげたもの。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_bench import RATE, ROOT, edit_distance, load_clips, normalize  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from idle_talk.director import merge_final  # noqa: E402
from idle_talk.vad import SPEECH_END, SPEECH_START, EnergyVad, rms  # noqa: E402

FRAME = RATE // 50  # 20ms
PARTIAL_S = 1.0
SPLIT_S, LOOK_S = 8.0, 2.0
FORCE_S, FORCE_KEEP_S = 8.0, 2.0


class Reazon:
    def __init__(self, model_dir: Path, threads: int) -> None:
        from idle_talk.asr_worker import load_reazon_model
        from idle_talk.config import Asr

        self.rec = load_reazon_model(Asr("reazon", str(model_dir), threads)).rec
        self.calls, self.audio_s = 0, 0.0

    def __call__(self, pcm: np.ndarray) -> tuple[str, list[str], list[float]]:
        if len(pcm) < RATE * 3 // 10:  # 短すぎる音は畳み込みの入力にならず落ちる (0.3 秒未満は語なし)
            return "", [], []
        self.calls += 1
        self.audio_s += len(pcm) / RATE
        s = self.rec.create_stream()
        s.accept_waveform(RATE, pcm.astype(np.float32) / 32768)
        self.rec.decode_stream(s)
        r = s.result
        return r.text.strip(), list(r.tokens), list(r.timestamps)


def make_stream(clips, gap_ms: int, bgm: int) -> tuple[np.ndarray, str]:
    gap = np.zeros(RATE * gap_ms // 1000, dtype=np.int16)
    pcm = np.concatenate([np.concatenate([c, gap]) for _, _, c in clips]).astype(np.int32)
    t = np.arange(len(pcm)) / RATE
    noise = np.random.default_rng(0).normal(0, bgm / 2, len(pcm))
    pcm = pcm + bgm * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.3 * t)) + noise
    return pcm.clip(-32768, 32767).astype(np.int16), "".join(ref for _, ref, _ in clips)


VAD = "energy"  # energy = 今の音量の VAD、silero = Silero VAD (faster-whisper 同梱の silero_vad_v6.onnx)
SILERO_ON, SILERO_OFF = 0.5, 0.35  # Silero の声らしさがこれ以上で発話開始、これ未満が SILERO_MIN_SIL_MS 続いたら終わり
SILERO_MIN_SIL_MS = 300


def silero_frames(pcm: np.ndarray):
    """Silero VAD の区切り。声らしさは 32ms ごと (前から順に状態を渡すので実時間でも同じ)。"""
    from faster_whisper.vad import get_vad_model

    n = len(pcm) // 512 * 512
    probs = get_vad_model()(pcm[:n].astype(np.float32) / 32768).reshape(-1)
    speaking, quiet = False, 0
    for i in range(len(pcm) // FRAME):
        prob = probs[min(len(probs) - 1, i * FRAME // 512)]
        ev = None
        if not speaking and prob >= SILERO_ON:
            speaking, quiet, ev = True, 0, SPEECH_START
        elif speaking:
            quiet = quiet + 1 if prob < SILERO_OFF else 0
            if quiet * 20 >= SILERO_MIN_SIL_MS:
                speaking, ev = False, SPEECH_END
        yield i, ev


def vad_frames(pcm: np.ndarray, ratio: float):
    """文字起こしモードと同じ VAD (自動校正 + floor_track、onset 200ms)。フレームごとに (番号, イベント)。"""
    if VAD == "silero":
        yield from silero_frames(pcm)
        return
    if VAD == "app":  # 本体の idle_talk.silero_vad.SileroVad (20ms ずつ実時間と同じ順に渡す)
        from idle_talk.silero_vad import SileroVad, load_session

        vad = SileroVad(load_session(ROOT / "data/models/silero_vad_v6.onnx"))
        for i in range(len(pcm) // FRAME):
            yield i, vad.process(pcm[i * FRAME:(i + 1) * FRAME].tobytes())
        return
    vad = EnergyVad(adaptive=True, ratio=ratio, onset_frames=10)
    vad.floor_track = True
    for i in range(len(pcm) // FRAME):
        yield i, vad.process(pcm[i * FRAME:(i + 1) * FRAME].tobytes())


def run_segment(pcm: np.ndarray, asr: Reazon, ratio: float) -> list[str]:
    out: list[str] = []
    buf: list[np.ndarray] | None = None
    partial = ""

    def finish(seg: np.ndarray) -> None:
        nonlocal partial
        text = merge_final(partial, asr(seg)[0])
        if text:
            out.append(text)
        partial = ""

    for i, ev in vad_frames(pcm, ratio):
        frame = pcm[i * FRAME:(i + 1) * FRAME]
        if ev == SPEECH_START:
            buf = [pcm[max(0, i - 24) * FRAME:i * FRAME]]  # onset + 前の音 (300ms) をおおまかに
        if buf is None:
            continue
        buf.append(frame)
        n = sum(len(b) for b in buf)
        if ev == SPEECH_END:
            finish(np.concatenate(buf))
            buf = None
            continue
        if n >= SPLIT_S * RATE:
            seg = np.concatenate(buf)
            look = int(LOOK_S * 50)
            levels = [rms(seg[-(k + 1) * FRAME:len(seg) - k * FRAME].tobytes()) for k in range(look)]
            cut = len(seg) - levels.index(min(levels)) * FRAME
            finish(seg[:cut])
            buf = [seg[cut:]]
            continue
        if (i + 1) % int(PARTIAL_S * 50) == 0 and n >= PARTIAL_S * RATE:
            partial = asr(np.concatenate(buf))[0] or partial
    return out


def common_prefix(a: list[str], b: list[str]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


CUT = "next"  # 確定した文字の後で音を切る位置: next = 次の文字の時刻、mid = 確定の最後の文字と次の文字の間
OVERLAP_MAX = 6  # 確定済みの末尾と新しい確定の頭が重なっていれば (文字の時刻のずれで同じ語を 2 回読んだ) 重なりを除く


def dedupe(done: str, new: str) -> str:
    for n in range(min(OVERLAP_MAX, len(done), len(new)), 0, -1):
        if done.endswith(new[:n]):
            return new[n:]
    return new


def run_agree(pcm: np.ndarray, asr: Reazon, ratio: float) -> list[str]:
    out: list[str] = []
    buf: np.ndarray | None = None
    prev: list[str] = []

    def commit(tokens: list[str], times: list[float], k: int) -> None:
        """頭の k 文字を確定し、k 文字目 (次の文字) の時刻より前の音を捨てる。"""
        nonlocal buf, prev
        if k <= 0:
            return
        text = dedupe("".join(out)[-OVERLAP_MAX:], "".join(tokens[:k]))
        if text:
            out.append(text)
        if k >= len(times):
            cut = len(buf)
        elif CUT == "mid":
            cut = int((times[k - 1] + times[k]) / 2 * RATE)
        else:
            cut = int(times[k] * RATE)
        buf = buf[cut:]
        prev = tokens[k:]

    for i, ev in vad_frames(pcm, ratio):
        frame = pcm[i * FRAME:(i + 1) * FRAME]
        if ev == SPEECH_START:
            buf = pcm[max(0, i - 24) * FRAME:i * FRAME].copy()
            prev = []
        if buf is None:
            continue
        buf = np.concatenate([buf, frame])
        if ev == SPEECH_END:
            text, tokens, times = asr(buf)
            text = dedupe("".join(out)[-OVERLAP_MAX:], "".join(tokens))
            if text:
                out.append(text)
            buf, prev = None, []
            continue
        if (i + 1) % int(PARTIAL_S * 50) or len(buf) < PARTIAL_S * RATE:
            continue
        _, tokens, times = asr(buf)
        k = common_prefix(prev, tokens)
        if len(buf) >= FORCE_S * RATE:  # 頭が揃わないまま長くなった: 最後の FORCE_KEEP_S より前の文字を確定する
            k = max(k, sum(1 for t in times if t < len(buf) / RATE - FORCE_KEEP_S))
        if k:
            commit(tokens, times, k)
        else:
            prev = tokens
    return out


CTX_S = 3.0  # ctx: 確定した音のうち、これだけを左の文脈として残して認識する (その範囲の文字は確定済みなので捨てる)
HOLD_S = 1.0
NO_VAD_END = False  # ctx: VAD の発話の終わりでは確定しない (音が続く前提で、頭が揃った所から確定し続ける)  # ctx: 溜めた音の終わりからこれより近い文字は、頭が揃っても確定しない (まだ変わりうる)


def run_ctx(pcm: np.ndarray, asr: Reazon, ratio: float) -> list[str]:
    """agree に左の文脈を足したもの。確定した所の手前 CTX_S の音を残して認識し、その範囲 (ctx 秒より前) の文字は捨てる。"""
    out: list[str] = []
    buf: np.ndarray | None = None
    ctx = 0.0  # buf の頭の、確定済みの音の長さ (秒)
    prev: list[str] = []

    def fresh(tokens, times):
        return [(tok, t) for tok, t in zip(tokens, times) if t >= ctx]

    for i, ev in vad_frames(pcm, ratio):
        frame = pcm[i * FRAME:(i + 1) * FRAME]
        if ev == SPEECH_START and buf is None:
            buf, ctx, prev = pcm[max(0, i - 24) * FRAME:i * FRAME].copy(), 0.0, []
        if buf is None:
            continue
        buf = np.concatenate([buf, frame])
        end = len(buf) / RATE
        if ev == SPEECH_END and not NO_VAD_END:
            _, tokens, times = asr(buf)
            text = dedupe("".join(out)[-OVERLAP_MAX:], "".join(tok for tok, _ in fresh(tokens, times)))
            if text:
                out.append(text)
            buf, ctx, prev = None, 0.0, []
            continue
        if (i + 1) % int(PARTIAL_S * 50) or end - ctx < PARTIAL_S:
            continue
        _, tokens, times = asr(buf)
        new = fresh(tokens, times)
        k = common_prefix(prev, [tok for tok, _ in new])
        k = min(k, sum(1 for _, t in new if t < end - HOLD_S))  # 終わりに近い文字はまだ確定しない
        if end - ctx >= FORCE_S:  # 頭が揃わないまま長くなった
            k = max(k, sum(1 for _, t in new if t < end - FORCE_KEEP_S))
        if not k:
            prev = [tok for tok, _ in new]
            continue
        text = dedupe("".join(out)[-OVERLAP_MAX:], "".join(tok for tok, _ in new[:k]))
        if text:
            out.append(text)
        boundary = (new[k - 1][1] + new[k][1]) / 2 if k < len(new) else end
        drop = max(0.0, boundary - CTX_S)
        buf = buf[int(drop * RATE):]
        ctx = boundary - drop
        prev = [tok for tok, _ in new[k:]]
    return out


def cer(ref: str, hyp: str) -> float:
    r, h = normalize(ref), normalize(hyp)
    return edit_distance(r, h) / max(1, len(r))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stream_bench")
    p.add_argument("--wavs", type=Path, default=ROOT / "data/asr_bench_tv")
    p.add_argument("--reazon-dir", type=Path, default=ROOT / "data/models/reazonspeech")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--gap-ms", type=int, default=200, help="クリップの間の無音 (BGM は続く)")
    p.add_argument("--bgm", type=int, default=300, help="BGM の振幅 (16bit)。0 なら無し")
    p.add_argument("--ratio", type=float, default=2.0, help="[transcribe].vad_ratio")
    p.add_argument("--methods", default="segment,agree")
    p.add_argument("--cut", default=CUT, choices=["next", "mid"])
    p.add_argument("--no-vad-end", action="store_true")
    p.add_argument("--vad", default=VAD, choices=["energy", "silero", "app"])
    p.add_argument("--min-sil-ms", type=int, default=SILERO_MIN_SIL_MS)
    args = p.parse_args(argv)
    globals()["CUT"] = args.cut
    globals()["NO_VAD_END"] = args.no_vad_end
    globals()["VAD"] = args.vad
    globals()["SILERO_MIN_SIL_MS"] = args.min_sil_ms
    clips = load_clips(args.wavs)
    pcm, ref = make_stream(clips, args.gap_ms, args.bgm)
    asr = Reazon(args.reazon_dir, args.threads)
    print(f"vad {args.vad}, stream {len(pcm) / RATE:.0f}s, {len(normalize(ref))} chars, gap {args.gap_ms}ms, bgm {args.bgm}")
    for m in args.methods.split(","):
        asr.calls, asr.audio_s = 0, 0.0
        t0 = time.perf_counter()
        out = {"segment": run_segment, "agree": run_agree, "ctx": run_ctx}[m](pcm, asr, args.ratio)
        hyp = "".join(out)
        print(f"{m:8} CER {cer(ref, hyp) * 100:5.1f}%  出力 {len(normalize(hyp))} 字 / {len(out)} 行  認識 {asr.calls} 回 "
              f"(音 {asr.audio_s:.0f}s、計算 {time.perf_counter() - t0:.0f}s)")
        (args.wavs / f"stream_{m}.txt").write_text("\n".join(out), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
