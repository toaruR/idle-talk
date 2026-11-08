#!/usr/bin/env bash
# 学習から GGUF 化まで (WSL で実行)。先に idle-talk を止めて GPU0 を空ける (idle-talk.cmd stop)。
#   bash /mnt/d/vagrant/idle-talk/tools/dialect/train.sh [学習の追加引数...]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
OUT=~/lora/yamagata
GGUF=/mnt/d/models/lora-yamagata-murayama.gguf
BASE=unsloth/gemma-4-12B-it-qat-q4_0-unquantized
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0  # GPU1 は他のアプリが使う

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$used" -gt 1500 ]; then
  echo "GPU0 が ${used}MiB 使用中。idle-talk (llama-server) を止めてから実行する" >&2
  exit 1
fi
# shellcheck disable=SC1090
source ~/venvs/dialect/bin/activate
python "$HERE/train_lora.py" --data "$REPO/data/dialect" --out "$OUT" --base "$BASE" "$@"
python ~/src/llama.cpp/convert_lora_to_gguf.py "$OUT/adapter" --base-model-id "$BASE" --outtype f16 --outfile "$GGUF"
echo "LoRA -> $GGUF"
