from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import torch
from trl import SFTConfig, SFTTrainer

from train_common import (
    configure_runtime,
    load_local_hf_dataset,
    load_unsloth_model,
    normalize_text_dataset,
    pick_train_eval_splits,
    trainer_init_kwargs,
)
from nlp_code.train_sft_old import DEFAULT_DATASET_PATH, DEFAULT_MODEL_PATH, MAX_SEQ_LEN_DEFAULT, TARGET_MODULES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--num_proc", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    datasets.disable_caching()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)
    print(f"[MODEL] {args.model_path}", flush=True)

    print("[1/5] Loading model + tokenizer...", flush=True)
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
    print("tokenizer class:", type(tokenizer).__name__, flush=True)
    print("eos_token:", tokenizer.eos_token, tokenizer.eos_token_id, flush=True)
    print("pad_token:", tokenizer.pad_token, tokenizer.pad_token_id, flush=True)
    print("has chat_template:", getattr(tokenizer, "chat_template", None) is not None, flush=True)

    print("[2/5] Loading tiny dataset slice...", flush=True)
    dataset_obj = load_local_hf_dataset(args.dataset_path)
    train_ds, eval_ds = pick_train_eval_splits(dataset_obj, seed=args.seed)
    train_limit = min(args.num_samples, len(train_ds))
    train_ds = train_ds.select(range(train_limit))
    if eval_ds is not None and len(eval_ds) > 0:
        eval_ds = eval_ds.select(range(min(max(1, train_limit // 4), len(eval_ds))))
    print(f"train slice={len(train_ds)}", flush=True)
    if eval_ds is not None:
        print(f"eval slice={len(eval_ds)}", flush=True)

    print("[3/5] Normalizing to text...", flush=True)
    train_ds = normalize_text_dataset(
        train_ds,
        tokenizer,
        max_seq_len=args.max_seq_len,
        num_proc=args.num_proc,
    )
    sample_text = train_ds[0]["text"]
    print("sample text preview:", sample_text[:600].replace("\n", "\\n"), flush=True)
    print("sample length:", train_ds[0]["length"], flush=True)

    print("[4/5] Single forward pass...", flush=True)
    batch = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=min(args.max_seq_len, 1024),
    )
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        out = model(**batch)
    print("forward logits shape:", tuple(out.logits.shape), flush=True)

    print("[5/5] SFTTrainer init only...", flush=True)
    cfg = SFTConfig(
        output_dir=str(Path("/tmp") / "codex_sft_smoke"),
        dataset_text_field="text",
        max_length=args.max_seq_len,
        packing=True,
        packing_strategy="bfd",
        num_train_epochs=1,
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=1,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
    )
    _trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        args=cfg,
        **trainer_init_kwargs(SFTTrainer, tokenizer),
    )
    print("SFTTrainer init ok", flush=True)


if __name__ == "__main__":
    main()
