"""LoRA あり / なしで eval_prompts.txt に答えさせて並べる (Windows、llama-server を 2 回起動)。

  python tools/dialect/eval_lora.py [--lora D:/models/lora-yamagata-murayama.gguf]
出力: data/dialect/eval.md
idle-talk を止めてから実行する (同じ GPU を使うため)。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from idle_talk.config import load_llm_launch  # noqa: E402
from idle_talk.llm_supervisor import build_llama_argv, llama_env  # noqa: E402
from build_dataset import system_prompt  # noqa: E402

PORT = 8093


def wait_ready(timeout: float = 180) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
            return
        except OSError:
            time.sleep(2)
    raise TimeoutError("llama-server not ready")


def chat(system: str, user: str) -> str:
    body = json.dumps({"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                       "max_tokens": 300, "temperature": 0.7}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=120))["choices"][0]["message"]["content"].strip()


def run(lora: str, prompts: list[str], system: str) -> list[str]:
    launch = load_llm_launch(ROOT / "idle_talk.toml")._replace(lora=lora, mmproj="")
    p = subprocess.Popen(build_llama_argv(launch, PORT), env=llama_env(),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_ready()
        return [chat(system, q) for q in prompts]
    finally:
        p.terminate()
        p.wait(30)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lora", default="D:/models/lora-yamagata-murayama.gguf")
    ap.add_argument("--prev", default="", help="前の版の LoRA (指定すると「前の版」の行も並べる)")
    ap.add_argument("--persona", default="yamagata")
    ap.add_argument("--n", type=int, default=0, help="使う問いかけの数 (0 で全部)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "dialect" / "eval.md")
    a = ap.parse_args()
    prompts = [l for l in Path(__file__).with_name("eval_prompts.txt").read_text(encoding="utf-8").splitlines() if l]
    prompts = prompts[:a.n] if a.n else prompts
    system = system_prompt(a.persona)
    base, tuned = run("", prompts, system), run(a.lora, prompts, system)
    prev = run(a.prev, prompts, system) if a.prev else [None] * len(prompts)
    out = a.out
    lines = ["# LoRA 比較\n"]
    for q, b, t, v in zip(prompts, base, tuned, prev):
        lines += [f"## {q}\n", f"- なし: {b}"] + ([f"- 前の版: {v}"] if v is not None else []) + [f"- あり: {t}\n"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
