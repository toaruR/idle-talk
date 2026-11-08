#!/usr/bin/env bash
# WSL Ubuntu で学習環境を作る (初回のみ)。venv は ~/venvs/dialect、llama.cpp は ~/src/llama.cpp。
set -euo pipefail
VENV=~/venvs/dialect
mkdir -p ~/venvs ~/src
uv venv --python 3.12 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install unsloth  # torch (CUDA) / transformers / trl / peft / bitsandbytes も入る
if [ ! -d ~/src/llama.cpp ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
fi
uv pip install -e ~/src/llama.cpp/gguf-py sentencepiece
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
EOF
