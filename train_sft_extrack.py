from __future__ import annotations

import argparse
import json
from pathlib import Path

import datasets
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from train_common import (
    configure_qwen_tokenizer_and_model,
    configure_runtime,
    ensure_dir,
    load_local_hf_dataset,
    load_lora_adapter_weights,
    load_unsloth_model,
    pick_train_eval_splits,
    resolve_latest_adapter_path,
)


datasets.disable_caching()

DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B"
DEFAULT_DATASET_PATH = "/gpfs/projects/p32908/data/nlp/hf_miner_v2"
DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/data/nlp/qwen3_4b_extract_sft_out"
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
    parser.add_argument("--max_seq_len", type=int, default=37200)
    parser.add_argument("--num_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--min_extractions", type=int, default=1)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--auto_continue",
        choices=["warm_start_latest", "from_scratch"],
        default="warm_start_latest",
    )
    return parser


def has_messages(example) -> bool:
    messages = example.get("messages")
    return isinstance(messages, list) and len(messages) >= 3


def count_extractions_from_messages(example) -> int:
    try:
        assistant_content = example["messages"][-1]["content"]
        obj = json.loads(assistant_content)
        extractions = obj.get("extractions", [])
        if isinstance(extractions, list):
            return len(extractions)
    except Exception:
        pass
    return 0


def is_good_extract_sample(example, *, min_extractions: int) -> bool:
    return has_messages(example) and count_extractions_from_messages(example) >= min_extractions


def render_messages_fallback(messages: list[dict]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip().capitalize()
        content = str(message.get("content", "")).strip()
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def build_text(example, tokenizer) -> dict:
    messages = example["messages"]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = render_messages_fallback(messages)
    return {"text": text}


def add_length(example, tokenizer) -> dict:
    return {"length": len(tokenizer.encode(example["text"], add_special_tokens=False))}


def tokenize_for_causal_lm(example, tokenizer, max_seq_len: int) -> dict:
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_len,
        padding=False,
    )
    encoded["labels"] = list(encoded["input_ids"])
    return encoded


def load_miner_dataset(dataset_path: str | Path):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    try:
        return load_local_hf_dataset(path)
    except FileNotFoundError:
        parquet_dir = path / "data"
        split_names = ("train", "validation", "test")
        parquet_files = {
            split: str(parquet_dir / f"{split}-00000-of-00001.parquet")
            for split in split_names
            if (parquet_dir / f"{split}-00000-of-00001.parquet").exists()
        }
        if parquet_files:
            return load_dataset("parquet", data_files=parquet_files)
        available = sorted(p.name for p in path.iterdir())
        raise FileNotFoundError(
            "Dataset path is neither a save_to_disk dataset nor a parquet export directory. "
            f"path={path} children={available[:20]}"
        )


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    output_dir = ensure_dir(args.output_dir)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)
    print(f"[MODEL] {args.model_path}", flush=True)
    print(f"[DATASET] {args.dataset_path}", flush=True)
    print(f"[OUTPUT] {output_dir}", flush=True)

    print("[1/5] Loading model + tokenizer...", flush=True)
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
    )
    configure_qwen_tokenizer_and_model(tokenizer, model)
    model.print_trainable_parameters()

    latest_adapter_path = resolve_latest_adapter_path(output_dir)
    if args.auto_continue == "warm_start_latest":
        if latest_adapter_path and load_lora_adapter_weights(model, latest_adapter_path):
            print(f"[warm_start] loaded adapter weights from: {latest_adapter_path}", flush=True)
        else:
            print("[warm_start] no previous adapter found, training from scratch", flush=True)
    else:
        print("[warm_start] disabled, training from scratch", flush=True)

    print("[2/5] Loading local miner dataset...", flush=True)
    dataset_obj = load_miner_dataset(args.dataset_path)
    if isinstance(dataset_obj, DatasetDict):
        print(f"splits={list(dataset_obj.keys())}", flush=True)
    train_ds, eval_ds = pick_train_eval_splits(
        dataset_obj,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    print(f"raw train={len(train_ds)}", flush=True)
    if eval_ds is not None:
        print(f"raw eval={len(eval_ds)}", flush=True)

    train_ds = train_ds.filter(
        lambda x: is_good_extract_sample(x, min_extractions=args.min_extractions),
        desc="filter train miner samples",
    )
    if eval_ds is not None:
        eval_ds = eval_ds.filter(
            lambda x: is_good_extract_sample(x, min_extractions=args.min_extractions),
            desc="filter eval miner samples",
        )
    print(f"filtered train={len(train_ds)}", flush=True)
    if eval_ds is not None:
        print(f"filtered eval={len(eval_ds)}", flush=True)

    if len(train_ds) == 0:
        raise ValueError("No usable miner samples after filtering.")

    print("[3/5] Formatting chat dataset to text...", flush=True)
    train_ds = train_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="format train text",
        num_proc=args.num_proc,
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda x: build_text(x, tokenizer),
            desc="format eval text",
            num_proc=args.num_proc,
        )

    train_ds = train_ds.map(
        lambda x: add_length(x, tokenizer),
        desc="compute train length",
        num_proc=args.num_proc,
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda x: add_length(x, tokenizer),
            desc="compute eval length",
            num_proc=args.num_proc,
        )

    before_train = len(train_ds)
    train_ds = train_ds.filter(lambda x: x["length"] <= args.max_seq_len)
    print(f"length filtered train: {before_train} -> {len(train_ds)}", flush=True)

    if eval_ds is not None:
        before_eval = len(eval_ds)
        eval_ds = eval_ds.filter(lambda x: x["length"] <= args.max_seq_len)
        print(f"length filtered eval:  {before_eval} -> {len(eval_ds)}", flush=True)
        if len(eval_ds) == 0:
            eval_ds = None
            print("eval split became empty after filtering; training without eval", flush=True)

    if len(train_ds) == 0:
        raise ValueError("No train samples remain after max_seq_len filtering.")

    sample = train_ds[0]
    print("[sample length]", sample["length"], flush=True)
    print("[sample text preview]", sample["text"][:1000], flush=True)

    print("[4/5] Building trainer...", flush=True)
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    else:
        raise ValueError("Could not find a valid eos token in tokenizer vocab.")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.init_kwargs["eos_token"] = tokenizer.eos_token
    tokenizer.init_kwargs["pad_token"] = tokenizer.pad_token
    if hasattr(tokenizer, "_special_tokens_map"):
        tokenizer._special_tokens_map["eos_token"] = tokenizer.eos_token
        tokenizer._special_tokens_map["pad_token"] = tokenizer.pad_token
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("tokenizer.eos_token =", repr(tokenizer.eos_token), flush=True)
    print("tokenizer.pad_token =", repr(tokenizer.pad_token), flush=True)
    print("tokenizer.eos_token_id =", tokenizer.eos_token_id, flush=True)
    print("tokenizer.pad_token_id =", tokenizer.pad_token_id, flush=True)
    print("tokenizer.eos_in_vocab =", tokenizer.eos_token in vocab, flush=True)

    train_ds = train_ds.map(
        lambda x: tokenize_for_causal_lm(x, tokenizer, args.max_seq_len),
        desc="tokenize train",
        num_proc=args.num_proc,
        remove_columns=train_ds.column_names,
    )
    if eval_ds is not None:
        eval_ds = eval_ds.map(
            lambda x: tokenize_for_causal_lm(x, tokenizer, args.max_seq_len),
            desc="tokenize eval",
            num_proc=args.num_proc,
            remove_columns=eval_ds.column_names,
        )

    cfg = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=args.eval_steps if eval_ds is not None else None,
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
        remove_unused_columns=False,
        seed=args.seed,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if eval_ds is not None else None,
    )

    print("[5/5] Training...", flush=True)
    trainer.train()

    trainer.model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved adapter + tokenizer to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
