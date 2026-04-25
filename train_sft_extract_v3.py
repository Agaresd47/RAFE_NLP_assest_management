from __future__ import annotations

import argparse
import warnings

import unsloth
import datasets
import torch
from transformers import (
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from train_common import (
    configure_runtime,
    ensure_dir,
    load_local_hf_dataset,
    load_lora_adapter_weights,
    load_unsloth_model,
    pick_train_eval_splits,
    resolve_latest_adapter_path,
)


warnings.filterwarnings("ignore", category=FutureWarning)

MAX_SEQ_LEN_DEFAULT = 32768
DEFAULT_MODEL_PATH = "Qwen/Qwen3-4B"
DEFAULT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v3"
DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/data/nlp/qwen3_4b_extract_sft_v3_out"
NUM_EPOCHS = 2.0
LEARNING_RATE = 5e-5
SAVE_STEPS = 50
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 5
EVAL_STEPS = 50
SAVE_TOTAL_LIMIT = 5
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
    parser.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    return parser


def build_text(example, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def add_length(example, tokenizer) -> dict:
    return {"length": len(tokenizer.encode(example["text"], add_special_tokens=False))}


def tokenize_row_assistant_only(example, tokenizer, max_seq_len: int) -> dict:
    prompt_messages = example["messages"][:-1]
    full_messages = example["messages"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    input_ids = full_ids[:max_seq_len]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)

    prompt_len = min(len(prompt_ids), len(input_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "prompt_length": prompt_len,
        "total_length": len(input_ids),
        "assistant_target_tokens": sum(1 for x in labels if x != -100),
        "prompt_text": prompt_text,
        "full_text": full_text,
    }


def add_assistant_target_length(example) -> dict:
    labels = example["labels"]
    return {"assistant_target_tokens": sum(1 for x in labels if x != -100)}


def drop_non_learning_rows(example) -> bool:
    return int(example.get("assistant_target_tokens", 0)) > 0


def build_assistant_preview(example) -> str:
    assistant = example["messages"][-1]["content"]
    assistant = str(assistant)
    return assistant if len(assistant) <= 1000 else assistant[:1000]


def _remove_columns(dataset, keep: set[str]) -> list[str]:
    return [name for name in dataset.column_names if name not in keep]


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    datasets.disable_caching()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)
    ensure_dir(args.output_dir)

    print("[1/4] Loading base model...", flush=True)
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
    resolved_adapter_path = resolve_latest_adapter_path(args.output_dir)
    if resolved_adapter_path:
        loaded = load_lora_adapter_weights(model, resolved_adapter_path)
        print(f"Loaded previous adapter: {loaded}", flush=True)
        print(f"Previous adapter path: {resolved_adapter_path}", flush=True)
    else:
        print("Loaded previous adapter: False", flush=True)
    model.print_trainable_parameters()
    print("tokenizer.eos_token =", repr(tokenizer.eos_token), flush=True)
    print("tokenizer.pad_token =", repr(tokenizer.pad_token), flush=True)
    print("model.config.eos_token_id =", getattr(model.config, "eos_token_id", None), flush=True)
    print("model.config.pad_token_id =", getattr(model.config, "pad_token_id", None), flush=True)

    print("[2/4] Loading local HF SFT dataset...", flush=True)
    dataset_obj = load_local_hf_dataset(args.dataset_path)
    train_ds, eval_ds = pick_train_eval_splits(
        dataset_obj,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    if eval_ds is None:
        raise ValueError("SFT training needs a validation split or eval_ratio > 0 to create one.")

    print("[3/4] Formatting + tokenizing SFT dataset...", flush=True)
    train_ds = train_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="format train",
        num_proc=args.num_proc,
    )
    eval_ds = eval_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="format eval",
        num_proc=args.num_proc,
    )
    train_ds = train_ds.map(
        lambda x: add_length(x, tokenizer),
        desc="length train",
        num_proc=args.num_proc,
    )
    eval_ds = eval_ds.map(
        lambda x: add_length(x, tokenizer),
        desc="length eval",
        num_proc=args.num_proc,
    )
    before_train = len(train_ds)
    before_eval = len(eval_ds)
    train_ds = train_ds.filter(lambda x: x["length"] <= args.max_seq_len)
    eval_ds = eval_ds.filter(lambda x: x["length"] <= args.max_seq_len)
    print(f"train={before_train} -> {len(train_ds)}", flush=True)
    print(f"eval={before_eval} -> {len(eval_ds)}", flush=True)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise ValueError("No samples remain after length filtering.")

    print("[sample length]", train_ds[0]["length"], flush=True)
    print("[sample text preview]", train_ds[0]["text"][:1000], flush=True)
    print("[sample assistant preview]", build_assistant_preview(train_ds[0]), flush=True)

    train_ds = train_ds.map(
        lambda x: tokenize_row_assistant_only(x, tokenizer, args.max_seq_len),
        desc="tokenize train assistant-only",
        num_proc=args.num_proc,
    )
    eval_ds = eval_ds.map(
        lambda x: tokenize_row_assistant_only(x, tokenizer, args.max_seq_len),
        desc="tokenize eval assistant-only",
        num_proc=args.num_proc,
    )
    train_ds = train_ds.map(add_assistant_target_length, desc="assistant target length train", num_proc=args.num_proc)
    eval_ds = eval_ds.map(add_assistant_target_length, desc="assistant target length eval", num_proc=args.num_proc)
    train_ds = train_ds.filter(drop_non_learning_rows, desc="drop empty target train")
    eval_ds = eval_ds.filter(drop_non_learning_rows, desc="drop empty target eval")
    print(f"assistant-only train={len(train_ds)} eval={len(eval_ds)}", flush=True)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise ValueError("No rows remain after assistant-only masking.")
    print("[sample prompt_length]", train_ds[0]["prompt_length"], flush=True)
    print("[sample total_length]", train_ds[0]["total_length"], flush=True)
    print("[sample assistant_target_tokens]", train_ds[0]["assistant_target_tokens"], flush=True)

    keep_columns = {"input_ids", "attention_mask", "labels"}
    train_ds = train_ds.remove_columns(_remove_columns(train_ds, keep_columns))
    eval_ds = eval_ds.remove_columns(_remove_columns(eval_ds, keep_columns))

    cfg = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("[4/4] Training...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter + tokenizer to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
