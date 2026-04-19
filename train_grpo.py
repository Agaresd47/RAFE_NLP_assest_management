from __future__ import annotations

import unsloth
import argparse
import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any

import datasets
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from online_dpo_utils import parse_completion_json
from task13_dataset_common import LABEL_ORDER, coerce_confidence, label_distance, normalize_label
from train_common import (
    configure_runtime,
    ensure_dir,
    find_latest_checkpoint,
    load_local_hf_dataset,
    load_lora_adapter_weights,
    load_unsloth_model,
    pick_train_eval_splits,
    resolve_latest_adapter_path,
)


warnings.filterwarnings(
    "ignore",
    message=r".*attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

_ORIGINAL_LOGGER_WARNING = logging.Logger.warning


def _patched_logger_warning(self, msg, *args, **kwargs):
    if (
        args == (FutureWarning,)
        and isinstance(msg, str)
        and "attention mask API under `transformers.modeling_attn_mask_utils`" in msg
    ):
        args = ()
    return _ORIGINAL_LOGGER_WARNING(self, msg, *args, **kwargs)


logging.Logger.warning = _patched_logger_warning


MODEL_PATH = "Qwen/Qwen3-8B"
DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo"
DPO_ADAPTER_PATH = "/scratch/xla2767/hold2/models/cot_dpo_adapter_v2"
OUTPUT_DIR = "/scratch/xla2767/hold2/models/cot_grpo_adapter_v2"

MAX_SEQ_LEN = 5200
MAX_PROMPT_LENGTH = 3072
MAX_COMPLETION_LENGTH = 768
NUM_EPOCHS = 1.0
LEARNING_RATE = 5e-6
SAVE_STEPS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 6
NUM_GENERATIONS = 2
TEMPERATURE = 0.8
TOP_P = 0.9
LOGGING_STEPS = 1
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.0
BETA = 0.02
CONFIDENCE_REWARD_WEIGHT = 0.6
RETURN_REWARD_WEIGHT = 0.4
MISSING_JSON_PENALTY = 0.6

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TOKENISH_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
THINK_BLOCK_RE = re.compile(r"^\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*(?P<tail>.*)$", re.DOTALL)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--dataset_path", default=DATASET_PATH)
    parser.add_argument("--dpo_adapter_path", default=DPO_ADAPTER_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max_prompt_length", type=int, default=MAX_PROMPT_LENGTH)
    parser.add_argument("--max_completion_length", type=int, default=MAX_COMPLETION_LENGTH)
    parser.add_argument("--num_epochs", type=float, default=NUM_EPOCHS)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--per_device_train_batch_size", type=int, default=PER_DEVICE_TRAIN_BATCH_SIZE)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=PER_DEVICE_EVAL_BATCH_SIZE)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--warmup_steps", type=int, default=WARMUP_STEPS)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument("--eval_ratio", type=float, default=0.02)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence_reward_weight", type=float, default=CONFIDENCE_REWARD_WEIGHT)
    parser.add_argument("--return_reward_weight", type=float, default=RETURN_REWARD_WEIGHT)
    return parser


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _align_batch_size(value: int, num_generations: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if value % num_generations == 0:
        return value

    aligned = max(num_generations, (value // num_generations) * num_generations)
    print(
        f"[config] {name}={value} 与 num_generations={num_generations} 不整除，"
        f"自动下调为 {aligned}",
        flush=True,
    )
    return aligned


def _estimate_epoch_stats(
    *,
    train_size: int,
    train_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
) -> dict[str, int]:
    global_train_micro_batch = max(1, train_batch_size * max(1, world_size))
    train_micro_steps = train_size // global_train_micro_batch
    optimizer_steps = math.ceil(train_micro_steps / max(1, gradient_accumulation_steps))
    return {
        "global_train_micro_batch": global_train_micro_batch,
        "train_micro_steps_per_epoch": train_micro_steps,
        "optimizer_steps_per_epoch": optimizer_steps,
    }


def _resolve_generation_batch_size(
    *,
    train_batch_size: int,
    world_size: int,
    num_generations: int,
) -> int:
    global_train_batch = max(1, train_batch_size * max(1, world_size))
    if global_train_batch % num_generations != 0:
        raise ValueError(
            f"global_train_batch={global_train_batch} must be divisible by "
            f"num_generations={num_generations}"
        )
    return global_train_batch


def _sync_trainer_state_config(resume_path: str | None, args) -> None:
    if not resume_path:
        return

    trainer_state_path = Path(resume_path) / "trainer_state.json"
    if not trainer_state_path.exists():
        return

    try:
        original_text = trainer_state_path.read_text()
        trainer_state = json.loads(original_text)
    except Exception as exc:
        print(f"[resume] failed to read trainer_state.json: {exc}", flush=True)
        return

    updates = {
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
    }
    changed = {}
    for key, value in updates.items():
        old_value = trainer_state.get(key)
        if old_value != value:
            trainer_state[key] = value
            changed[key] = {"old": old_value, "new": value}

    if not changed:
        print("[resume] trainer_state.json already matches current logging/save steps", flush=True)
        return

    backup_path = trainer_state_path.with_name("trainer_state.json.codex.bak")
    if not backup_path.exists():
        backup_path.write_text(original_text)

    trainer_state_path.write_text(json.dumps(trainer_state, ensure_ascii=False, indent=2))
    print(f"[resume] patched trainer_state.json with current args: {changed}", flush=True)


def _label_sign(label: str | None) -> int:
    label = normalize_label(label)
    if label in {"positive", "very_positive"}:
        return 1
    if label in {"negative", "very_negative"}:
        return -1
    return 0


def _distance_score(candidate_label: str | None, target_label: str | None) -> float:
    candidate_label = normalize_label(candidate_label)
    target_label = normalize_label(target_label)
    if not candidate_label or not target_label:
        return 0.0
    max_distance = max(1, len(LABEL_ORDER) - 1)
    return 1.0 - label_distance(candidate_label, target_label) / max_distance


def _return_strength(excess_1m: Any, ret_1m: Any) -> float:
    for raw in (excess_1m, ret_1m):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if not math.isnan(value) and not math.isinf(value):
            return min(1.0, abs(value) / 0.12)
    return 0.0


def _normalize_grpo_dataset(dataset: Dataset, *, num_proc: int) -> Dataset:
    required = {"prompt", "return_label"}
    missing = sorted(required.difference(dataset.column_names))
    if missing:
        raise ValueError("GRPO dataset needs prompt/return_label columns; missing: " + ", ".join(missing))

    keep_cols = [
        "prompt",
        "return_label",
        "ret_1m",
        "excess_1m",
        "ticker",
        "factor",
        "teacher_label",
        "teacher_confidence",
        "preference_source",
    ]

    def clean(example):
        out = {"prompt": str(example["prompt"]).strip(), "return_label": normalize_label(example["return_label"])}
        for col in keep_cols:
            if col in ("prompt", "return_label"):
                continue
            out[col] = example.get(col)
        return out

    cleaned = dataset.map(clean, desc="normalize grpo prompts", num_proc=num_proc)
    cleaned = cleaned.filter(lambda x: bool(x["prompt"]) and bool(x["return_label"]))
    drop_cols = [col for col in cleaned.column_names if col not in keep_cols]
    if drop_cols:
        cleaned = cleaned.remove_columns(drop_cols)
    return cleaned


def _approx_completion_tokens(text: str) -> int:
    return len(TOKENISH_RE.findall(str(text)))


def _extract_final_json(completion: str) -> dict[str, Any] | None:
    clean = str(completion).strip().replace("<tool_call>", "").replace("</tool_call>", "")
    clean = clean.replace("<|im_end|>", "").strip()
    think_match = THINK_BLOCK_RE.match(clean)
    if think_match:
        clean = str(think_match.group("tail") or "").strip()
    if not clean or not clean.endswith("}"):
        return None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _structure_penalty(completion: str) -> tuple[float, dict[str, Any], int]:
    approx_tokens = _approx_completion_tokens(completion)
    final_json = _extract_final_json(completion)
    parsed = parse_completion_json(completion) if final_json is not None else None

    penalty = 0.0
    if final_json is None:
        penalty += MISSING_JSON_PENALTY

    return penalty, (parsed or final_json or {}), approx_tokens


def _return_reward_func(
    prompts,
    completions,
    return_label,
    excess_1m=None,
    ret_1m=None,
    **kwargs,
):
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        structure_penalty, parsed, _ = _structure_penalty(completion)
        if not parsed:
            rewards.append(_clamp(-structure_penalty))
            continue

        target = normalize_label(return_label[idx])
        candidate = normalize_label(parsed.get("sentiment_label"))
        strength = _return_strength(
            excess_1m[idx] if excess_1m is not None else None,
            ret_1m[idx] if ret_1m is not None else None,
        )

        dist_score = _distance_score(candidate, target)
        cand_sign = _label_sign(candidate)
        target_sign = _label_sign(target)
        same_sign = cand_sign == target_sign and target_sign != 0
        opposite_sign = cand_sign == -target_sign and cand_sign != 0 and target_sign != 0

        reward = 0.75 * dist_score
        if target_sign == 0:
            reward += 0.10 if cand_sign == 0 else -0.05
        else:
            if candidate == target:
                reward += 0.15 + 0.10 * strength
            elif same_sign:
                reward += 0.08
            elif cand_sign == 0:
                reward -= 0.10 + 0.10 * strength
            elif opposite_sign:
                reward -= 0.30 + 0.20 * strength

        if candidate == "neutral" and target != "neutral":
            reward -= 0.10 + 0.10 * strength

        reward -= structure_penalty
        rewards.append(_clamp(reward))
    return rewards


def _confidence_reward_func(
    prompts,
    completions,
    return_label,
    excess_1m=None,
    ret_1m=None,
    **kwargs,
):
    rewards: list[float] = []
    for idx, completion in enumerate(completions):
        structure_penalty, parsed, _ = _structure_penalty(completion)
        if not parsed:
            rewards.append(_clamp(-structure_penalty))
            continue

        target = normalize_label(return_label[idx])
        candidate = normalize_label(parsed.get("sentiment_label"))
        confidence = coerce_confidence(parsed.get("confidence_score"))
        strength = _return_strength(
            excess_1m[idx] if excess_1m is not None else None,
            ret_1m[idx] if ret_1m is not None else None,
        )

        dist_score = _distance_score(candidate, target)
        cand_sign = _label_sign(candidate)
        target_sign = _label_sign(target)
        same_sign = cand_sign == target_sign and target_sign != 0
        opposite_sign = cand_sign == -target_sign and cand_sign != 0 and target_sign != 0

        if target_sign == 0:
            target_conf = 0.48 if cand_sign == 0 else 0.32
        elif candidate == target:
            target_conf = 0.62 + 0.18 * strength
        elif same_sign:
            target_conf = 0.50 + 0.08 * strength
        elif cand_sign == 0:
            target_conf = 0.34 - 0.06 * strength
        else:
            target_conf = 0.18

        conf_fit = 1.0 - min(1.0, abs(confidence - target_conf) / 0.75)
        reward = conf_fit

        if candidate == "neutral" and target != "neutral":
            reward -= 0.08 + 0.10 * strength

        if opposite_sign:
            reward -= max(0.0, confidence - 0.30) * (0.9 + 0.5 * strength)
        elif candidate != target:
            reward -= max(0.0, confidence - 0.55) * 0.20

        if candidate == target and target != "neutral":
            reward += max(0.0, confidence - 0.55) * (0.10 + 0.10 * strength)

        if candidate == "neutral":
            reward -= max(0.0, confidence - 0.55) * 0.20

        reward += 0.10 * (dist_score - 0.5)
        reward -= structure_penalty
        rewards.append(_clamp(reward))
    return rewards


def _reward_breakdown(example: dict[str, Any], completion: str) -> dict[str, Any]:
    structure_penalty, parsed, approx_tokens = _structure_penalty(completion)
    confidence_reward = _confidence_reward_func(
        prompts=[example["prompt"]],
        completions=[completion],
        return_label=[example["return_label"]],
        excess_1m=[example.get("excess_1m")],
        ret_1m=[example.get("ret_1m")],
    )[0]
    return_reward = _return_reward_func(
        prompts=[example["prompt"]],
        completions=[completion],
        return_label=[example["return_label"]],
        excess_1m=[example.get("excess_1m")],
        ret_1m=[example.get("ret_1m")],
    )[0]
    total = _clamp(0.6 * confidence_reward + 0.4 * return_reward)
    return {
        "confidence_reward": round(confidence_reward, 4),
        "return_reward": round(return_reward, 4),
        "total_reward": round(total, 4),
        "structure_penalty": round(structure_penalty, 4),
        "approx_completion_tokens": approx_tokens,
        "has_final_json": bool(_extract_final_json(completion)),
        "parsed_label": normalize_label(parsed.get("sentiment_label") if parsed else None),
        "parsed_confidence": round(coerce_confidence(parsed.get("confidence_score")), 4) if parsed else None,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    configure_runtime()
    datasets.disable_caching()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)
    ensure_dir(args.output_dir)
    resume_path = find_latest_checkpoint(args.output_dir)
    _sync_trainer_state_config(resume_path, args)

    print("[1/4] Loading policy model...", flush=True)
    model, tokenizer = load_unsloth_model(
        model_name=args.model_path,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
        dtype=torch.float16,
    )
    if resume_path:
        loaded = load_lora_adapter_weights(model, resume_path)
        print(f"Loaded GRPO resume adapter: {loaded}", flush=True)
        print(f"GRPO resume path: {resume_path}", flush=True)
    else:
        resolved_dpo_adapter_path = resolve_latest_adapter_path(args.dpo_adapter_path)
        if resolved_dpo_adapter_path:
            loaded = load_lora_adapter_weights(model, resolved_dpo_adapter_path)
            print(f"Loaded DPO adapter: {loaded}", flush=True)
            print(f"DPO adapter path: {resolved_dpo_adapter_path}", flush=True)
        else:
            print("Loaded DPO adapter: False", flush=True)
            print(f"DPO adapter path not found under: {args.dpo_adapter_path}", flush=True)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()

    print("[2/4] Loading GRPO prompt dataset...", flush=True)
    dataset_obj = load_local_hf_dataset(args.dataset_path)
    train_ds, _ = pick_train_eval_splits(dataset_obj, eval_ratio=args.eval_ratio, seed=args.seed)

    print("[3/4] Normalizing prompt pool...", flush=True)
    train_ds = _normalize_grpo_dataset(train_ds, num_proc=args.num_proc)
    print(f"train={len(train_ds)}", flush=True)

    raw_train_batch_size = args.per_device_train_batch_size
    args.per_device_train_batch_size = _align_batch_size(
        args.per_device_train_batch_size,
        args.num_generations,
        name="per_device_train_batch_size",
    )

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = int(torch.distributed.get_world_size())
    else:
        world_size = 1
    epoch_stats = _estimate_epoch_stats(
        train_size=len(train_ds),
        train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        world_size=world_size,
    )
    generation_batch_size = _resolve_generation_batch_size(
        train_batch_size=args.per_device_train_batch_size,
        world_size=world_size,
        num_generations=args.num_generations,
    )
    print("[config] effective GRPO setup", flush=True)
    print(
        f"[config] train_batch raw={raw_train_batch_size} aligned={args.per_device_train_batch_size}",
        flush=True,
    )
    print(
        f"[config] grad_accum={args.gradient_accumulation_steps} num_generations={args.num_generations} "
        f"world_size={world_size}",
        flush=True,
    )
    print(
        f"[config] global_train_micro_batch={epoch_stats['global_train_micro_batch']} "
        f"generation_batch_size={generation_batch_size}",
        flush=True,
    )
    print(
        f"[config] est_train_micro_steps/epoch={epoch_stats['train_micro_steps_per_epoch']} "
        f"est_optimizer_steps/epoch={epoch_stats['optimizer_steps_per_epoch']}",
        flush=True,
    )
    print(
        f"[config] max_prompt_length={args.max_prompt_length} "
        f"max_completion_length={args.max_completion_length} max_seq_len={args.max_seq_len}",
        flush=True,
    )

    cfg = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=False,
        fp16=True,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_steps=args.save_steps,
        save_total_limit=5,
        report_to="none",
        dataloader_drop_last=True,
        dataloader_num_workers=2,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        generation_batch_size=generation_batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        reward_weights=[args.confidence_reward_weight, args.return_reward_weight],
        scale_rewards="group",
        remove_unused_columns=False,
        seed=args.seed,
    )

    print("[4/4] Training GRPO...", flush=True)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[_confidence_reward_func, _return_reward_func],
        args=cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )
    trainer.create_model_card = lambda *args, **kwargs: None
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved GRPO adapter + tokenizer to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
