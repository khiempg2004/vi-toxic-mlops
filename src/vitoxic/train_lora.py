"""Tinh chỉnh LoRA cho Qwen2.5-Instruct trên ViHSD (sinh nhãn dạng chat).

Chạy (trong container GPU): python -m vitoxic.train_lora [hydra overrides...]
Ví dụ: python -m vitoxic.train_lora train.run_id=v1 data.batches_upto=2 model=qwen2.5-3b
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from omegaconf import DictConfig
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from vitoxic.config import cfg_path, load_cfg, repo_root, to_dict
from vitoxic.data.io import load_split, load_training_frame
from vitoxic.labels import LABELS, build_messages
from vitoxic.metrics import classification_metrics


def oversample(df: pd.DataFrame, target_ratio: float, seed: int) -> pd.DataFrame:
    counts = df["label"].value_counts()
    target = int(counts.max() * target_ratio)
    parts = []
    for label, n in counts.items():
        part = df[df["label"] == label]
        if n < target:
            part = pd.concat([part, part.sample(target - n, replace=True, random_state=seed)])
        parts.append(part)
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


class ChatSFTDataset(Dataset):
    """Mã hoá prompt (chat template) + nhãn; chỉ tính loss trên phần trả lời."""

    def __init__(self, texts, labels, tokenizer, max_chars: int, max_length: int):
        self.items = [self._encode(t, y, tokenizer, max_chars, max_length) for t, y in zip(texts, labels, strict=True)]

    @staticmethod
    def _encode(text, label, tok, max_chars, max_length):
        while True:
            msgs = build_messages(text, max_chars)
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            full = tok.apply_chat_template(
                msgs + [{"role": "assistant", "content": label}], tokenize=False
            )
            assert full.startswith(prompt), "Chat template không nhất quán giữa prompt và hội thoại đầy đủ"
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            a_ids = tok(full[len(prompt) :], add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(a_ids) <= max_length or max_chars < 32:
                break
            max_chars //= 2
        ids = (p_ids + a_ids)[:max_length]
        labels = ([-100] * len(p_ids) + a_ids)[:max_length]
        return {"input_ids": ids, "labels": labels}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(x["input_ids"]) for x in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            L = len(x["input_ids"])
            ids[i, :L] = torch.tensor(x["input_ids"])
            labels[i, :L] = torch.tensor(x["labels"])
            mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def first_token_eval(model, tok, texts, labels, max_chars: int, batch_size: int = 32) -> dict | None:
    """Đánh giá nhanh: so sánh logit token đầu tiên của 3 nhãn (một lượt forward)."""
    first_ids = [tok(lab, add_special_tokens=False)["input_ids"][0] for lab in LABELS]
    if len(set(first_ids)) != len(LABELS):
        print("[warn] Token đầu của các nhãn trùng nhau, bỏ qua first_token_eval")
        return None
    model.eval()
    tok.padding_side = "left"
    preds = []
    for i in range(0, len(texts), batch_size):
        prompts = [
            tok.apply_chat_template(build_messages(t, max_chars), add_generation_prompt=True, tokenize=False)
            for t in texts[i : i + batch_size]
        ]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :]
        idx = logits[:, first_ids].argmax(-1).tolist()
        preds.extend(LABELS[j] for j in idx)
    tok.padding_side = "right"
    return classification_metrics(list(labels), preds)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main(overrides: list[str] | None = None) -> Path:
    cfg: DictConfig = load_cfg(overrides)
    set_seed(cfg.seed)
    tc = cfg.train
    run_id = tc.run_id or f"{cfg.model.short}-{datetime.now():%Y%m%d-%H%M%S}"
    out_dir = cfg_path(cfg, "models_dir") / "candidates" / str(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(tc.wandb) and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        os.environ.setdefault("WANDB_PROJECT", cfg.project)

    tok = AutoTokenizer.from_pretrained(cfg.model.name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    train_df = load_training_frame(cfg)
    data_stats = {"raw": train_df["label"].value_counts().to_dict(), "sources": train_df["source"].value_counts().to_dict()}
    if tc.oversample_minority:
        train_df = oversample(train_df, tc.minority_target_ratio, cfg.seed)
    data_stats["after_oversample"] = train_df["label"].value_counts().to_dict()
    print("Dữ liệu huấn luyện:", json.dumps(data_stats, ensure_ascii=False))

    val_df = load_split(cfg, "val")
    val_df = val_df.sample(min(tc.val_samples, len(val_df)), random_state=cfg.seed)

    t0 = time.time()
    train_ds = ChatSFTDataset(train_df["text"], train_df["label"], tok, cfg.data.max_chars, tc.max_length)
    val_ds = ChatSFTDataset(val_df["text"], val_df["label"], tok, cfg.data.max_chars, tc.max_length)
    print(f"Tokenize xong {len(train_ds)} mẫu trong {time.time() - t0:.1f}s")

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if tc.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=tc.lora.r,
        lora_alpha=tc.lora.alpha,
        lora_dropout=tc.lora.dropout,
        target_modules=list(tc.lora.target_modules),
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(out_dir / "_checkpoints"),
        num_train_epochs=tc.epochs,
        learning_rate=tc.lr,
        per_device_train_batch_size=tc.batch_size,
        per_device_eval_batch_size=tc.batch_size * 2,
        gradient_accumulation_steps=tc.grad_accum,
        warmup_ratio=tc.warmup_ratio,
        weight_decay=tc.weight_decay,
        lr_scheduler_type=tc.lr_scheduler,
        logging_steps=tc.logging_steps,
        eval_strategy="steps",
        eval_steps=tc.eval_steps,
        save_strategy="no",
        bf16=True,
        group_by_length=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=["wandb"] if use_wandb else [],
        run_name=str(run_id),
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=PadCollator(tok.pad_token_id),
    )
    t0 = time.time()
    train_out = trainer.train()
    train_seconds = time.time() - t0

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    val_metrics = first_token_eval(model, tok, list(val_df["text"]), list(val_df["label"]), cfg.data.max_chars)
    if val_metrics:
        print(f"val macro-F1 (first-token) = {val_metrics['macro_f1']:.4f}")
        if use_wandb:
            import wandb

            wandb.log({"val/macro_f1": val_metrics["macro_f1"], "val/accuracy": val_metrics["accuracy"]})

    meta = {
        "run_id": run_id,
        "base_model": cfg.model.name,
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit(),
        "train_seconds": train_seconds,
        "train_loss": train_out.training_loss,
        "data": data_stats,
        "val_first_token": val_metrics,
        "config": to_dict(cfg),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"Đã lưu LoRA tại {out_dir}")
    if use_wandb:
        import wandb

        wandb.finish()
    return out_dir


if __name__ == "__main__":
    main(sys.argv[1:])
