"""Gemma 4 12B に村山弁の QLoRA を学習する (WSL + Unsloth、GPU 1 枚 12GB)。

  python train_lora.py --data /mnt/d/vagrant/idle-talk/data/dialect --out ~/lora/yamagata

損失は最後の assistant の返答だけにかける (system・利用者・履歴は -100)。
チャットテンプレートは推論と同じく enable_thinking=False で描く (llama-server の --chat-template-kwargs と揃える)。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from unsloth import FastModel  # unsloth は transformers より先に import する

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import DataCollatorForSeq2Seq  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

BASE = "unsloth/gemma-4-12B-it-qat-q4_0-unquantized"  # 推論に使う QAT q4_0 GGUF と同じ重み
IGNORE = -100


def render(tok, msgs: list[dict], gen: bool) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen, enable_thinking=False)


def encode(tok, ex: dict, max_len: int) -> dict:
    msgs = ex["messages"]
    answer = msgs[-1]["content"]
    # 推論 (llama-server) は生成プロンプトの形 (思考無効だと空の思考チャンネル付き) から続けるので、
    # 学習も「生成プロンプト + 返答 + ターンの終わり」にする。終わりの記号はテンプレートから取る
    prefix = render(tok, msgs[:-1], gen=True)
    whole = render(tok, msgs, gen=False)
    at = whole.rfind(answer)
    if at < 0:
        raise ValueError("chat template: answer not found")
    full = prefix + answer + whole[at + len(answer):]
    p_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [IGNORE] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="動作確認用 (既定 -1 で epochs に従う)")
    a = ap.parse_args()

    model, proc = FastModel.from_pretrained(a.base, max_seq_length=a.max_len, load_in_4bit=True,
                                            full_finetuning=False)
    tok = getattr(proc, "tokenizer", proc)  # 画像つきモデルは processor が返る。文字だけ使う
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", random_state=3407,
        use_gradient_checkpointing="unsloth")

    def ds(name: str) -> Dataset:
        rows = [encode(tok, ex, a.max_len) for ex in load(a.data / f"{name}.jsonl")]
        return Dataset.from_list(rows)

    train, valid = ds("train"), ds("valid")
    print(f"train {len(train)}, valid {len(valid)}, max tokens {max(len(r) for r in train['input_ids'])}")
    print("sample:\n" + tok.decode(train[0]["input_ids"]))

    trainer = SFTTrainer(
        model=model, tokenizer=tok, train_dataset=train, eval_dataset=valid,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=IGNORE),
        args=SFTConfig(
            output_dir=str(a.out / "ckpt"), per_device_train_batch_size=1, gradient_accumulation_steps=a.accum,
            num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.05,
            logging_steps=5, eval_strategy="steps", eval_steps=40, save_strategy="steps", save_steps=40,
            save_total_limit=2, optim="adamw_8bit", weight_decay=0.01, bf16=torch.cuda.is_bf16_supported(),
            max_length=a.max_len, dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=False,
            report_to="none", seed=3407))
    trainer.train()
    print("final eval:", trainer.evaluate())
    model.save_pretrained(str(a.out / "adapter"))
    tok.save_pretrained(str(a.out / "adapter"))
    print(f"adapter -> {a.out / 'adapter'}")


if __name__ == "__main__":
    main()
