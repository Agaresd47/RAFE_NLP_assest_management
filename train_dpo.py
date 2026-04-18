from __future__ import annotations

import argparse
import logging
import warnings

import datasets
import torch
from trl import DPOConfig, DPOTrainer

from train_common import (
    configure_runtime,
    ensure_dir,
    load_local_hf_dataset,
    load_lora_adapter_weights,
    load_unsloth_model,
    normalize_dpo_dataset,
    pick_train_eval_splits,
    resolve_latest_adapter_path,
    trainer_init_kwargs,
)

warnings.filterwarnings(
    "ignore",
    message=r".*attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

_ORIGINAL_LOGGER_WARNING = logging.Logger.warning


def _patched_logger_warning(self, msg, *args, **kwargs):
    # Transformers currently emits one deprecation path as:
    # logger.warning_once(message, FutureWarning)
    # which breaks stdlib logging string formatting. Drop the stray arg.
    if (
        args == (FutureWarning,)
        and isinstance(msg, str)
        and "attention mask API under `transformers.modeling_attn_mask_utils`" in msg
    ):
        args = ()
    return _ORIGINAL_LOGGER_WARNING(self, msg, *args, **kwargs)


logging.Logger.warning = _patched_logger_warning


MAX_SEQ_LEN_DEFAULT = 4096
MAX_PROMPT_LEN_DEFAULT = 4096
DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
DEFAULT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo"
DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/models/cot_dpo_adapter"
DEFAULT_SFT_ADAPTER_PATH = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_thinking_sft_out"
NUM_EPOCHS = 3.0
LEARNING_RATE = 2e-5
SAVE_STEPS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 6
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 5
EVAL_STEPS = 10
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
    parser.add_argument("--sft_adapter_path", default=DEFAULT_SFT_ADAPTER_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--max_prompt_len", type=int, default=MAX_PROMPT_LEN_DEFAULT)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    return parser


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
    if args.sft_adapter_path:
        resolved_sft_adapter_path = resolve_latest_adapter_path(args.sft_adapter_path)
        if resolved_sft_adapter_path:
            loaded = load_lora_adapter_weights(model, resolved_sft_adapter_path)
            print(f"Loaded SFT adapter: {loaded}", flush=True)
            print(f"SFT adapter path: {resolved_sft_adapter_path}", flush=True)
        else:
            print(f"Loaded SFT adapter: False", flush=True)
            print(f"SFT adapter path not found under: {args.sft_adapter_path}", flush=True)
    model.print_trainable_parameters()
    print("tokenizer.eos_token =", repr(tokenizer.eos_token), flush=True)
    print("tokenizer.pad_token =", repr(tokenizer.pad_token), flush=True)
    print("model.config.eos_token_id =", getattr(model.config, "eos_token_id", None), flush=True)
    print("model.config.pad_token_id =", getattr(model.config, "pad_token_id", None), flush=True)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        print(
            "generation_config.eos_token_id =",
            getattr(model.generation_config, "eos_token_id", None),
            flush=True,
        )
        print(
            "generation_config.pad_token_id =",
            getattr(model.generation_config, "pad_token_id", None),
            flush=True,
        )

    print("[2/4] Loading local HF preference dataset...", flush=True)
    dataset_obj = load_local_hf_dataset(args.dataset_path)
    train_ds, eval_ds = pick_train_eval_splits(
        dataset_obj,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )
    if eval_ds is None:
        raise ValueError(
            "DPO training needs a validation split or eval_ratio > 0 to create one."
        )

    print("[3/4] Normalizing DPO dataset...", flush=True)
    train_ds = normalize_dpo_dataset(train_ds, num_proc=args.num_proc)
    eval_ds = normalize_dpo_dataset(eval_ds, num_proc=args.num_proc)
    print(f"train={len(train_ds)}  eval={len(eval_ds)}", flush=True)

    # TRL expects this mutable warning registry on some model wrappers, but
    # Unsloth + PEFT models may not expose it by default.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    cfg = DPOConfig(
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
        save_steps=args.save_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        report_to="none",
        dataloader_num_workers=2,
        dataset_num_proc=args.num_proc,
        max_length=args.max_seq_len,
        max_prompt_length=args.max_prompt_len,
        beta=args.beta,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        **trainer_init_kwargs(DPOTrainer, tokenizer),
    )

    print("[4/4] Training...", flush=True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter + tokenizer to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
