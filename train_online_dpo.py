from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import datasets
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from online_dpo_utils import batch_logit_scores, extract_prompt_text, score_candidate, select_pair
from train_common import (
    configure_runtime,
    ensure_dir,
    load_local_hf_dataset,
    load_lora_adapter_weights,
    load_unsloth_model,
    normalize_dpo_dataset,
    pick_train_eval_splits,
    trainer_init_kwargs,
)


MAX_SEQ_LEN_DEFAULT = 4096
MAX_PROMPT_LEN_DEFAULT = 4096
DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-9B-Base"
DEFAULT_PROMPT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo"
DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/models/cot_online_dpo_adapter"
DEFAULT_ROLLOUT_DIR = "/scratch/xla2767/hold2/models/cot_online_dpo_rollouts"
DEFAULT_SFT_ADAPTER_PATH = "/scratch/xla2767/hold2/models/cot_sft_adapter"
DEFAULT_RM_MODEL_PATH = "Skywork/Skywork-Reward-V2-Qwen3-4B"
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
    parser.add_argument("--prompt_dataset_path", default=DEFAULT_PROMPT_DATASET_PATH)
    parser.add_argument("--prompt_split", default="train")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rollout_dir", default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--sft_adapter_path", default=DEFAULT_SFT_ADAPTER_PATH)
    parser.add_argument("--rm_model_path", default=DEFAULT_RM_MODEL_PATH)
    parser.add_argument("--prm_model_path", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--prompts_per_round", type=int, default=64)
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_pair_margin", type=float, default=0.10)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN_DEFAULT)
    parser.add_argument("--max_prompt_len", type=int, default=MAX_PROMPT_LEN_DEFAULT)
    parser.add_argument("--judge_max_length", type=int, default=4096)
    parser.add_argument("--num_epochs_per_round", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--train_steps_per_round", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--buffer_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    return parser


def _resolve_split(dataset_obj, split: str):
    if isinstance(dataset_obj, DatasetDict):
        if split in dataset_obj:
            return dataset_obj[split]
        if len(dataset_obj) > 0:
            return dataset_obj[list(dataset_obj.keys())[0]]
        raise ValueError("DatasetDict has no splits")
    if isinstance(dataset_obj, Dataset):
        return dataset_obj
    raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)!r}")


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def _load_scalar_judge(model_path: str | None, *, label: str):
    if not model_path:
        return None, None

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _ensure_pad_token(tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
        print(f"Loaded {label}: {model_path}", flush=True)
        return model, tokenizer
    except Exception as exc:
        print(f"Falling back to heuristic {label} scoring for {model_path}: {exc}", flush=True)
        return None, None


def _score_judge_batch(model, tokenizer, prompts: list[str], completions: list[str], max_length: int) -> list[float] | None:
    if model is None or tokenizer is None:
        return None
    device = _model_device(model)
    enc = tokenizer(
        prompts,
        completions,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {key: value.to(device) for key, value in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
    return batch_logit_scores(outputs.logits)


def _generate_candidates(
    model,
    tokenizer,
    prompt_text: str,
    *,
    num_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_prompt_len: int,
) -> list[str]:
    _ensure_pad_token(tokenizer)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    )
    device = _model_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])

    input_ids = inputs["input_ids"].repeat(num_candidates, 1)
    attention_mask = inputs["attention_mask"].repeat(num_candidates, 1)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    completions = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [text.strip() for text in completions]


def _build_round_pairs(
    *,
    prompt_dataset,
    tokenizer,
    model,
    rm_model,
    rm_tokenizer,
    prm_model,
    prm_tokenizer,
    args,
    round_idx: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(args.seed + round_idx)
    if len(prompt_dataset) == 0:
        return [], []

    indices = [rng.randrange(len(prompt_dataset)) for _ in range(args.prompts_per_round)]
    rollout_records: list[dict[str, Any]] = []
    train_records: list[dict[str, Any]] = []

    for local_idx, row_idx in enumerate(indices):
        example = prompt_dataset[row_idx]
        prompt_text = extract_prompt_text(example, tokenizer)
        target_label = example.get("return_label") or example.get("teacher_label")
        prompt_meta = {
            "ticker": example.get("ticker"),
            "form": example.get("form"),
            "report_date": example.get("report_date") or example.get("parsed_report_date"),
            "factor": example.get("factor"),
            "return_label": example.get("return_label"),
            "teacher_label": example.get("teacher_label"),
            "teacher_confidence": example.get("teacher_confidence"),
            "source_index": row_idx,
        }

        candidate_texts = _generate_candidates(
            model,
            tokenizer,
            prompt_text,
            num_candidates=args.num_candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_prompt_len=args.max_prompt_len,
        )

        prompts = [prompt_text] * len(candidate_texts)
        rm_scores = _score_judge_batch(rm_model, rm_tokenizer, prompts, candidate_texts, args.judge_max_length)
        prm_scores = _score_judge_batch(prm_model, prm_tokenizer, prompts, candidate_texts, args.judge_max_length)

        scored_candidates: list[dict[str, Any]] = []
        for cand_idx, candidate_text in enumerate(candidate_texts):
            rm_score = rm_scores[cand_idx] if rm_scores is not None else None
            prm_score = prm_scores[cand_idx] if prm_scores is not None else None
            candidate_name = f"round{round_idx:03d}_prompt{local_idx:03d}_cand{cand_idx:02d}"
            candidate_score = score_candidate(
                candidate_name=candidate_name,
                prompt_text=prompt_text,
                completion_text=candidate_text,
                target_label=target_label,
                prm_score=prm_score,
                rm_score=rm_score,
            )
            scored_candidates.append(
                {
                    "candidate_name": candidate_score.candidate_name,
                    "candidate_text": candidate_text,
                    "prm_score": candidate_score.prm_score,
                    "rm_score": candidate_score.rm_score,
                    "return_score": candidate_score.return_score,
                    "confidence_score": candidate_score.confidence_score,
                    "format_score": candidate_score.format_score,
                    "total": candidate_score.total,
                    "parsed_json": candidate_score.parsed_json,
                    "sentiment_label": candidate_score.sentiment_label,
                }
            )

        try:
            chosen, rejected = select_pair(scored_candidates)
        except ValueError:
            continue

        if chosen["total"] - rejected["total"] < args.min_pair_margin:
            continue

        pair_source = f"{chosen['candidate_name']}__over__{rejected['candidate_name']}"
        score_blob = json.dumps(scored_candidates, ensure_ascii=False)
        rollout_records.append(
            {
                "round": round_idx,
                "prompt_index": row_idx,
                "prompt": prompt_text,
                "chosen": chosen["candidate_text"],
                "rejected": rejected["candidate_text"],
                "preference_source": pair_source,
                "candidate_scores": score_blob,
                **prompt_meta,
            }
        )
        train_records.append(
            {
                "prompt": prompt_text,
                "chosen": chosen["candidate_text"],
                "rejected": rejected["candidate_text"],
            }
        )

    return rollout_records, train_records


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    datasets.disable_caching()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)

    output_root = ensure_dir(args.output_dir)
    rollout_root = ensure_dir(args.rollout_dir)

    print("[1/5] Loading policy model...", flush=True)
    policy_model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
    if args.sft_adapter_path:
        loaded = load_lora_adapter_weights(policy_model, args.sft_adapter_path)
        print(f"Loaded SFT adapter: {loaded}", flush=True)
    policy_model.print_trainable_parameters()

    print("[2/5] Loading prompt pool...", flush=True)
    prompt_dataset_obj = load_local_hf_dataset(args.prompt_dataset_path)
    prompt_dataset = _resolve_split(prompt_dataset_obj, args.prompt_split)
    if len(prompt_dataset) == 0:
        raise RuntimeError("Prompt dataset split is empty.")
    print(f"Prompt pool size: {len(prompt_dataset)}", flush=True)

    print("[3/5] Loading judges...", flush=True)
    rm_model, rm_tokenizer = _load_scalar_judge(args.rm_model_path, label="RM")
    prm_model, prm_tokenizer = _load_scalar_judge(args.prm_model_path, label="PRM")

    buffer_records: list[dict[str, Any]] = []

    print("[4/5] Online rollout + update loop...", flush=True)
    for round_idx in range(args.rounds):
        print(f"[Round {round_idx + 1}/{args.rounds}] generating candidates...", flush=True)
        rollout_records, train_records = _build_round_pairs(
            prompt_dataset=prompt_dataset,
            tokenizer=tokenizer,
            model=policy_model,
            rm_model=rm_model,
            rm_tokenizer=rm_tokenizer,
            prm_model=prm_model,
            prm_tokenizer=prm_tokenizer,
            args=args,
            round_idx=round_idx,
        )
        print(
            f"[Round {round_idx + 1}/{args.rounds}] retained {len(train_records)} preference pairs",
            flush=True,
        )
        if not train_records:
            continue

        buffer_records.extend(rollout_records)
        if len(buffer_records) > args.buffer_size:
            buffer_records = buffer_records[-args.buffer_size :]

        round_rollout_dir = rollout_root / f"round_{round_idx:03d}"
        round_rollout_dir.mkdir(parents=True, exist_ok=True)
        with (round_rollout_dir / "rollouts.jsonl").open("w", encoding="utf-8") as handle:
            for record in rollout_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        with (round_rollout_dir / "buffer.jsonl").open("w", encoding="utf-8") as handle:
            for record in buffer_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        train_dataset = Dataset.from_list(train_records)
        train_dataset = normalize_dpo_dataset(train_dataset, num_proc=args.num_proc)
        train_ds, eval_ds = pick_train_eval_splits(
            train_dataset,
            eval_ratio=args.eval_ratio,
            seed=args.seed + round_idx,
        )

        round_output_dir = output_root / f"round_{round_idx:03d}"
        round_output_dir.mkdir(parents=True, exist_ok=True)
        dpo_args = DPOConfig(
            output_dir=str(round_output_dir),
            num_train_epochs=args.num_epochs_per_round,
            max_steps=args.train_steps_per_round,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            bf16=True,
            fp16=False,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.0,
            logging_steps=5,
            eval_strategy="steps" if eval_ds is not None else "no",
            eval_steps=max(1, min(25, args.train_steps_per_round)),
            save_steps=args.save_steps,
            save_total_limit=5,
            report_to="none",
            dataloader_num_workers=2,
            dataset_num_proc=args.num_proc,
            max_length=args.max_seq_len,
            max_prompt_length=args.max_prompt_len,
            beta=args.beta,
        )

        trainer = DPOTrainer(
            model=policy_model,
            ref_model=None,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=dpo_args,
            **trainer_init_kwargs(DPOTrainer, tokenizer),
        )
        print(f"[Round {round_idx + 1}/{args.rounds}] training on {len(train_ds)} pairs...", flush=True)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint if round_idx == 0 else None)
        policy_model = trainer.model
        trainer.model.save_pretrained(str(round_output_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(round_output_dir))
        torch.cuda.empty_cache()

    final_dir = ensure_dir(output_root / "final")
    policy_model.save_pretrained(str(final_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved final online-DPO adapter to {final_dir}", flush=True)


if __name__ == "__main__":
    main()
