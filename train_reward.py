from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import datasets
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from train_common import (
    configure_runtime,
    ensure_dir,
    find_latest_checkpoint,
    load_local_hf_dataset,
    pick_train_eval_splits,
)


DEFAULT_MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-4B"
DEFAULT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo"
DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/models/cot_reward_model"
DEFAULT_MAX_LENGTH = 4096


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune a reward / PRM-style model from DPO chosen-rejected pairs.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--num_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--disable_auto_resume", action="store_true")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--disable_lora", action="store_false", dest="use_lora")
    return parser


def join_prompt_and_response(prompt: str, response: str) -> str:
    return f"{prompt.rstrip()}\n{response.strip()}"


def truncate_pair_text(
    tokenizer,
    prompt: str,
    response: str,
    *,
    max_length: int,
    prompt_budget: int = 3072,
) -> str:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    prompt_ids = prompt_ids[-min(len(prompt_ids), prompt_budget) :]

    remaining = max(32, max_length - len(prompt_ids))
    response_ids = response_ids[:remaining]
    merged = prompt_ids + response_ids
    return tokenizer.decode(merged, skip_special_tokens=False)


def tokenize_pairs(dataset: Dataset, tokenizer, max_length: int, num_proc: int) -> Dataset:
    def preprocess(example):
        chosen_text = truncate_pair_text(
            tokenizer,
            example["prompt"],
            example["chosen"],
            max_length=max_length,
        )
        rejected_text = truncate_pair_text(
            tokenizer,
            example["prompt"],
            example["rejected"],
            max_length=max_length,
        )
        chosen_tokens = tokenizer(
            chosen_text,
            truncation=True,
            max_length=max_length,
        )
        rejected_tokens = tokenizer(
            rejected_text,
            truncation=True,
            max_length=max_length,
        )
        return {
            "input_ids_chosen": chosen_tokens["input_ids"],
            "attention_mask_chosen": chosen_tokens["attention_mask"],
            "input_ids_rejected": rejected_tokens["input_ids"],
            "attention_mask_rejected": rejected_tokens["attention_mask"],
        }

    keep_columns = {"input_ids_chosen", "attention_mask_chosen", "input_ids_rejected", "attention_mask_rejected"}
    tokenized = dataset.map(preprocess, num_proc=num_proc, desc="tokenize reward pairs")
    drop_cols = [col for col in tokenized.column_names if col not in keep_columns]
    return tokenized.remove_columns(drop_cols)


@dataclass
class RewardPairCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features):
        chosen = [
            {
                "input_ids": feature["input_ids_chosen"],
                "attention_mask": feature["attention_mask_chosen"],
            }
            for feature in features
        ]
        rejected = [
            {
                "input_ids": feature["input_ids_rejected"],
                "attention_mask": feature["attention_mask_rejected"],
            }
            for feature in features
        ]
        chosen_batch = self.tokenizer.pad(chosen, padding=True, return_tensors="pt")
        rejected_batch = self.tokenizer.pad(rejected, padding=True, return_tensors="pt")
        return {
            "input_ids_chosen": chosen_batch["input_ids"],
            "attention_mask_chosen": chosen_batch["attention_mask"],
            "input_ids_rejected": rejected_batch["input_ids"],
            "attention_mask_rejected": rejected_batch["attention_mask"],
        }


class RewardTrainer(Trainer):
    def __init__(self, margin: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_outputs = model(
            input_ids=inputs["input_ids_chosen"],
            attention_mask=inputs["attention_mask_chosen"],
        )
        rejected_outputs = model(
            input_ids=inputs["input_ids_rejected"],
            attention_mask=inputs["attention_mask_rejected"],
        )
        chosen_scores = chosen_outputs.logits.squeeze(-1)
        rejected_scores = rejected_outputs.logits.squeeze(-1)
        loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores - self.margin).mean()
        if return_outputs:
            return loss, {
                "chosen_scores": chosen_scores.detach(),
                "rejected_scores": rejected_scores.detach(),
            }
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Unsloth patches Trainer.prediction_step for RL trainers and assumes
        # raw "prompt" text exists in eval batches. Reward training here uses
        # tokenized chosen/rejected pairs instead, so evaluation must stay on
        # the plain pairwise loss path.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        loss = loss.mean().detach()
        if prediction_loss_only:
            return loss, None, None
        logits = torch.stack(
            (outputs["chosen_scores"], outputs["rejected_scores"]),
            dim=-1,
        )
        return loss, logits, None


def maybe_apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    return get_peft_model(model, config)


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    datasets.disable_caching()
    ensure_dir(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    if args.use_lora:
        model = maybe_apply_lora(model)
        model.print_trainable_parameters()

    dataset_obj = load_local_hf_dataset(args.dataset_path)
    train_ds, eval_ds = pick_train_eval_splits(dataset_obj, eval_ratio=args.eval_ratio, seed=args.seed)
    if eval_ds is None:
        raise ValueError("Reward training needs a validation split or eval_ratio > 0.")

    train_ds = tokenize_pairs(train_ds, tokenizer, args.max_length, args.num_proc)
    eval_ds = tokenize_pairs(eval_ds, tokenizer, args.max_length, args.num_proc)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=5,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        save_total_limit=10,
        remove_unused_columns=False,
    )

    trainer = RewardTrainer(
        margin=args.margin,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=RewardPairCollator(tokenizer),
    )
    resume_path = args.resume_from_checkpoint
    if resume_path is None and not args.disable_auto_resume:
        resume_path = find_latest_checkpoint(args.output_dir)
        if resume_path:
            print(f"Auto-resuming from {resume_path}", flush=True)
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved reward model artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
