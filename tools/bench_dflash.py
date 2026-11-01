"""Compare a private llama-server with/without DFlash; never stop other servers.

Standard library only. All subprocess windows are hidden on Windows. Models and
ports are explicit; each invocation stops only the child it created.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
HIDDEN = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SYSTEM = "あなたは日本語で自然に話す会話相手です。前置きや見出しは使わず、相手の気持ちに沿って答えてください。"
CASES = [
    ("chat_rain", [{"role": "user", "content": "今日は雨で散歩に行けなかった。家でのんびりしたいけど、少し楽しくなる過ごし方はある？会話として自然な4文で答えて。"}]),
    ("chat_work", [{"role": "user", "content": "仕事で小さなミスをして、帰宅してからも気になってる。説教や質問はせず、ほっとできるような言葉を自然な4文で返して。"}]),
    ("chat_history", [
        {"role": "user", "content": "週末、近場で少し気分転換したい。"},
        {"role": "assistant", "content": "近場でゆっくりできる場所なら、公園や図書館もよさそうだね。"},
        {"role": "user", "content": "人が多い場所は疲れるから、静かなところがいいな。"},
        {"role": "assistant", "content": "それなら朝の公園や、小さな喫茶店が落ち着けそうだね。"},
        {"role": "user", "content": "予算は千円くらい。雨が降った場合も考えて、無理のない過ごし方を6文くらいで提案して。"},
    ]),
    ("json_control", [{"role": "user", "content": "雨の日に家でできる趣味を4つ、JSON配列だけで出力して。各要素はnameとreasonという文字列のキーを持ち、reasonは短い日本語1文にして。"}]),
]


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def get_json(base: str, path: str, timeout: float = 10) -> object:
    with urllib.request.urlopen(base + path, timeout=timeout) as response:
        return json.load(response)


def get_text(base: str, path: str) -> str:
    with urllib.request.urlopen(base + path, timeout=10) as response:
        return response.read().decode("utf-8")


def gpu_snapshot() -> dict:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, creationflags=HIDDEN, timeout=10,
    )
    return {"utc": dt.datetime.now(dt.timezone.utc).isoformat(), "csv": result.stdout.strip()}


def gpu_monitor(path: Path, stop: threading.Event) -> None:
    with path.open("w", encoding="utf-8") as output:
        while not stop.is_set():
            try:
                output.write(json.dumps(gpu_snapshot()) + "\n")
                output.flush()
            except Exception as exc:
                output.write(json.dumps({"error": str(exc)}) + "\n")
            stop.wait(1)


def request(base: str, messages: list[dict], seed: int, max_tokens: int, temperature: float | None) -> dict:
    payload = {
        "model": "local", "messages": messages, "stream": True,
        "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
        "seed": seed, "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    content, reasoning = "", ""
    ttft = first_sentence = None
    events, arrivals, usage, timings = [], [], {}, {}
    with urllib.request.urlopen(req, timeout=180) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"])
            events.append(event)
            elapsed = time.perf_counter() - started
            if event.get("usage"):
                usage = event["usage"]
            if event.get("timings"):
                timings = event["timings"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                text = delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or ""
                if text:
                    if ttft is None:
                        ttft = elapsed
                    content += text
                    arrivals.append({"seconds": elapsed, "characters": len(text)})
                    if first_sentence is None and re.search(r"[。！？!?]", content):
                        first_sentence = elapsed
    wall = time.perf_counter() - started
    tokens = usage.get("completion_tokens", timings.get("predicted_n"))
    return {
        "seed": seed, "max_tokens": max_tokens, "temperature_override": temperature,
        "wall_s": wall, "ttft_s": ttft, "first_sentence_s": first_sentence,
        "completion_tokens": tokens, "wall_tokens_per_s": tokens / wall if tokens else None,
        "content": content, "reasoning": reasoning, "usage": usage, "timings": timings,
        "arrivals": arrivals, "events": events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="D:/tools/llama.cpp/llama-server.exe")
    parser.add_argument("--model", default="D:/models/gemma-4-12b-it-qat-q4_0.gguf")
    parser.add_argument("--draft", default=str(ROOT / "data/models/dflash/gemma4-12B-it-DFlash-Q4_K_M-Anbeeld.gguf"))
    parser.add_argument("--output", default=str(ROOT / "data/bench/dflash-20261028"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--draft-n", type=int, default=0)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--tensor-split", default="3,1")
    parser.add_argument("--draft-device", default="CUDA1")
    parser.add_argument("--single-gpu", type=int, choices=(0, 1))
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--ubatch", type=int, default=128)
    parser.add_argument("--cases", default=",".join(case for case, _ in CASES))
    args = parser.parse_args()
    output = Path(args.output) / args.label
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    base = f"http://127.0.0.1:{args.port}"
    command = [
        args.server, "-m", args.model, "--host", "127.0.0.1", "--port", str(args.port),
        "-ngl", "99", "-c", str(args.ctx), "-np", "1", "-fa", "on",
        "-b", str(args.batch), "-ub", str(args.ubatch), "--cache-ram", "0", "--jinja",
        "--chat-template-kwargs", '{"enable_thinking":false}', "--metrics",
        "--spec-type", "draft-dflash" if args.draft_n else "none",
    ]
    if args.single_gpu is None:
        command += ["--split-mode", "layer", "--tensor-split", args.tensor_split, "--main-gpu", "0"]
    else:
        command += ["--split-mode", "none", "--device", f"CUDA{args.single_gpu}", "--main-gpu", str(args.single_gpu)]
    if args.draft_n:
        command += ["-md", args.draft, "--spec-draft-ngl", "99", "--spec-draft-device", args.draft_device,
                    "--spec-draft-n-max", str(args.draft_n)]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0,1"
    version_result = subprocess.run([args.server, "--version"], capture_output=True, text=True, creationflags=HIDDEN)
    metadata = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "args": vars(args),
        "command": command, "gpu_before": gpu_snapshot(), "cases": CASES,
        "server_version": (version_result.stdout + version_result.stderr).strip(),
        "notes": "Synthetic prompts; server-default sampling unless overridden; cache_prompt=false; existing services retained.",
    }
    save(output / "metadata.json", metadata)
    stop = threading.Event()
    monitor = threading.Thread(target=gpu_monitor, args=(output / "gpu.jsonl", stop), daemon=True)
    monitor.start()
    process = None
    rows = []
    try:
        with (output / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, creationflags=HIDDEN)
            save(output / "pid.json", {"pid": process.pid})
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Server exited with {process.returncode}; see {output / 'server.log'}")
                try:
                    if get_json(base, "/health", 2).get("status") == "ok":
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("Server readiness timeout")
                time.sleep(0.5)
            save(output / "props.json", get_json(base, "/props"))
            warmup = request(base, [{"role": "user", "content": "こんにちは。短く挨拶して。"}], 1234, 24, args.temperature)
            save(output / "warmup.json", warmup)
            (output / "metrics-before.txt").write_text(get_text(base, "/metrics"), encoding="utf-8")
            selected = set(args.cases.split(","))
            for repeat in range(args.repeats):
                for index, (name, turns) in enumerate(CASES):
                    if name not in selected:
                        continue
                    messages = [{"role": "system", "content": SYSTEM}, *turns]
                    result = request(base, messages, 1000 + 10 * repeat + index, args.max_tokens, args.temperature)
                    row = {"case": name, "repeat": repeat, **result}
                    rows.append(row)
                    save(output / f"{repeat}-{name}.json", row)
                    (output / f"{repeat}-{name}-metrics.txt").write_text(get_text(base, "/metrics"), encoding="utf-8")
                    print(json.dumps({"label": args.label, "case": name, "repeat": repeat,
                                      "tokens": result["completion_tokens"], "ttft_s": result["ttft_s"],
                                      "first_sentence_s": result["first_sentence_s"], "wall_s": result["wall_s"],
                                      "timings": result["timings"]}, ensure_ascii=False), flush=True)
            (output / "metrics-after.txt").write_text(get_text(base, "/metrics"), encoding="utf-8")
            summaries = {}
            for name in selected:
                case_rows = [row for row in rows if row["case"] == name]
                summaries[name] = {
                    key: statistics.median([row[key] for row in case_rows if row[key] is not None])
                    if any(row[key] is not None for row in case_rows) else None
                    for key in ("wall_s", "ttft_s", "first_sentence_s", "completion_tokens", "wall_tokens_per_s")
                }
            save(output / "summary.json", summaries)
    except BaseException as exc:
        save(output / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        intentional_shutdown = process is not None and process.poll() is None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        stop.set()
        monitor.join(timeout=15)
        save(output / "finished.json", {"utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                                       "server_returncode": process.returncode if process else None,
                                       "intentional_shutdown": intentional_shutdown,
                                       "requests_completed": len(rows), "gpu_after": gpu_snapshot()})


if __name__ == "__main__":
    main()
