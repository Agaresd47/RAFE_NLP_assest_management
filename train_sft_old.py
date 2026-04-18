from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from train_common import (
    configure_runtime,
    ensure_dir,
    find_latest_checkpoint,
    load_local_hf_dataset,
    load_unsloth_model,
    normalize_text_dataset,
    pick_train_eval_splits,
    trainer_init_kwargs,
)


MAX_SEQ_LEN_DEFAULT = 4096
DEFAULT_MODEL_PATH = "unsloth/Qwen3-4B-Instruct-2507"
DEFAULT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/models/cot_sft_adapter"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--disable_auto_resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    datasets.disable_caching()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)
    ensure_dir(args.output_dir)

    print("[1/4] Loading model...", flush=True)
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
    model.print_trainable_parameters()

    print("[2/4] Loading local HF dataset...", flush=True)
    dataset_obj = load_local_hf_dataset(args.dataset_path)
    train_ds, eval_ds = pick_train_eval_splits(
        dataset_obj,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    if eval_ds is None:
        raise ValueError(
            "SFT training needs a validation split or eval_ratio > 0 to create one."
        )

    print("[3/4] Formatting datasets...", flush=True)
    train_ds = normalize_text_dataset(
        train_ds,
        tokenizer,
        max_seq_len=args.max_seq_len,
        num_proc=args.num_proc,
    )
    eval_ds = normalize_text_dataset(
        eval_ds,
        tokenizer,
        max_seq_len=args.max_seq_len,
        num_proc=args.num_proc,
    )
    print(f"train={len(train_ds)}  eval={len(eval_ds)}", flush=True)

    cfg = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=True,
        packing_strategy="bfd",
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=args.save_steps,
        save_total_limit=20,
        report_to="none",
        dataloader_num_workers=2,
        dataset_num_proc=args.num_proc,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        **trainer_init_kwargs(SFTTrainer, tokenizer),
    )

    print("[4/4] Training...", flush=True)
    resume_path = args.resume_from_checkpoint
    if resume_path is None and not args.disable_auto_resume:
        resume_path = find_latest_checkpoint(args.output_dir)
        if resume_path:
            print(f"Auto-resuming from {resume_path}", flush=True)
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter + tokenizer to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
