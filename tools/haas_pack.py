"""Haas-ja の ONNX モデルを配布用に小さく詰める / 元に戻す (numpy だけで動く)。

python tools/haas_pack.py pack   <model_dir> <pack_dir>   # encoder.onnx.data (2.4GB) → 約 1.2GB の分割ファイル
python tools/haas_pack.py unpack <pack_dir> <model_dir>   # 元の encoder.onnx.data をバイト単位で復元 (SHA-256 で確認)

Haas-ja の重みは元々 bfloat16 なので、fp32 に広げた encoder.onnx.data は下位 16 ビットがすべて 0。
pack はその上位 16 ビットだけを残し (それ以外の領域はそのまま)、GitHub の 1 ファイル 100MB 制限に収まるよう分割する。
unpack は 0 を足して元に戻し、記録した SHA-256 と照合する。encoder.onnx / decoder.npz / tokens.txt / config.json は
そのままコピーする。
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

DATA = "encoder.onnx.data"
PART = 95 * 1024 * 1024
COPY = ("encoder.onnx", "decoder.npz", "tokens.txt", "config.json")
CHUNK = 64 * 1024 * 1024  # 4 の倍数


def _regions(model_dir: Path) -> list[tuple[int, int]]:
    """encoder.onnx.data の中で float32 の重みが入っている (offset, length)。"""
    import onnx

    model = onnx.load(str(model_dir / "encoder.onnx"), load_external_data=False)
    out = []
    for t in model.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL or t.data_type != onnx.TensorProto.FLOAT:
            continue
        info = {e.key: e.value for e in t.external_data}
        out.append((int(info.get("offset", 0)), int(info["length"])))
    return sorted(out)


def pack(model_dir: Path, pack_dir: Path) -> None:
    pack_dir.mkdir(parents=True, exist_ok=True)
    data = model_dir / DATA
    size = data.stat().st_size
    segments, pos = [], 0
    for off, length in _regions(model_dir):
        if off > pos:
            segments.append([pos, off - pos, "raw"])
        segments.append([off, length, "bf16"])
        pos = off + length
    if pos < size:
        segments.append([pos, size - pos, "raw"])

    sha = hashlib.sha256()
    part_no, part, part_len, parts = 0, None, 0, []

    def write(b: bytes):
        nonlocal part_no, part, part_len
        while b:
            if part is None or part_len >= PART:
                if part:
                    part.close()
                name = f"encoder.pack.{part_no:03d}"
                parts.append(name)
                part, part_len, part_no = open(pack_dir / name, "wb"), 0, part_no + 1
            n = min(len(b), PART - part_len)
            part.write(b[:n])
            part_len += n
            b = b[n:]

    with open(data, "rb") as f:
        for seg in segments:  # bf16 で表せない値 (下位 16 ビットが 0 でない) を含む領域はそのまま残す
            if seg[2] == "bf16":
                f.seek(seg[0])
                left = seg[1]
                while left:
                    buf = f.read(min(left, CHUNK))
                    left -= len(buf)
                    if np.any(np.frombuffer(buf, np.uint32) & 0xFFFF):
                        seg[2] = "raw"
                        break
        for off, length, mode in segments:
            f.seek(off)
            left = length
            while left:
                buf = f.read(min(left, CHUNK))
                left -= len(buf)
                sha.update(buf)
                write((np.frombuffer(buf, np.uint32) >> 16).astype(np.uint16).tobytes() if mode == "bf16" else buf)
    if part:
        part.close()
    manifest = {"data": DATA, "size": size, "sha256": sha.hexdigest(), "segments": segments, "parts": parts}
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    for name in COPY:
        shutil.copy2(model_dir / name, pack_dir / name)
    print(f"{len(parts)} parts, sha256 {manifest['sha256']}")


def unpack(pack_dir: Path, model_dir: Path) -> None:
    manifest = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    model_dir.mkdir(parents=True, exist_ok=True)

    def reader():
        for name in manifest["parts"]:
            with open(pack_dir / name, "rb") as f:
                while buf := f.read(CHUNK):
                    yield buf

    src, pending, pos = reader(), bytearray(), 0

    def take(n: int) -> bytes:
        nonlocal pending, pos
        if len(pending) - pos < n:
            pending = pending[pos:]
            pos = 0
            while len(pending) < n:
                pending += next(src)
        b = bytes(pending[pos:pos + n])
        pos += n
        return b

    sha = hashlib.sha256()
    tmp = model_dir / (manifest["data"] + ".tmp")
    shown = -1
    with open(tmp, "wb") as out:
        for off, length, mode in manifest["segments"]:
            left = length
            while left:
                n = min(left, CHUNK)
                if mode == "bf16":
                    u = np.frombuffer(take(n // 2), np.uint16).astype(np.uint32) << 16
                    buf = u.tobytes()
                else:
                    buf = take(n)
                out.write(buf)
                sha.update(buf)
                left -= n
            if (pct := out.tell() * 100 // manifest["size"]) != shown:
                shown = pct
                print(f"\r{pct}%", end="", flush=True)
    print()
    if sha.hexdigest() != manifest["sha256"]:
        tmp.unlink()
        raise SystemExit("SHA-256 が一致しません。ダウンロードし直してください")
    tmp.replace(model_dir / manifest["data"])
    for name in COPY:
        if (pack_dir / name).resolve() != (model_dir / name).resolve():
            shutil.copy2(pack_dir / name, model_dir / name)
    print(f"ok: {model_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("pack", "unpack"):
        raise SystemExit(__doc__)
    (pack if sys.argv[1] == "pack" else unpack)(Path(sys.argv[2]), Path(sys.argv[3]))
