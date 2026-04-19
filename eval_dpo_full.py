import unsloth
import argparse
import json
import logging
import re
import warnings
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from task13_dataset_common import LABEL_ORDER, coerce_confidence, label_distance, normalize_label
from train_common import load_lora_adapter_weights, load_unsloth_model, resolve_latest_adapter_path


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/scratch/xla2767/hold2/models/cot_grpo_adapter_v2"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
OUTPUT_DIR = "/gpfs/projects/p32908/nlp_result/dpo_test"
SPLIT = "test"
MAX_SEQ_LEN = 8192
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

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
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    return parser


def make_prompt(messages, tokenizer) -> str:
    prompt_messages = messages[:2]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_model_output(text: str) -> dict:
    raw = str(text or "")
    think_match = THINK_RE.search(raw)
    thinking = think_match.group(1).strip() if think_match else None

    parsed = None
    json_text = None
    for match in reversed(list(JSON_BLOCK_RE.finditer(raw))):
        candidate = match.group(0).strip()
        parsed = safe_json_load(candidate)
        if parsed is not None:
            json_text = candidate
            break

    if parsed is None:
        parsed = {}

    sentiment = normalize_label(parsed.get("sentiment_label"))
    confidence = coerce_confidence(parsed.get("confidence_score")) if "confidence_score" in parsed else None
    reasoning = parsed.get("reasoning_chain")
    if not reasoning and thinking:
        reasoning = thinking

    return {
        "raw_text": raw,
        "think_text": thinking,
        "json_text": json_text,
        "parsed_json": parsed if isinstance(parsed, dict) else {},
        "reasoning_chain": reasoning,
        "sentiment_label": sentiment,
        "confidence_score": confidence,
        "parse_success": json_text is not None,
    }


def parse_gt_output(text: str) -> dict:
    raw = str(text or "")
    parsed = safe_json_load(raw)
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "raw_text": raw,
        "reasoning_chain": parsed.get("reasoning_chain"),
        "sentiment_label": normalize_label(parsed.get("sentiment_label")),
        "confidence_score": coerce_confidence(parsed.get("confidence_score")) if "confidence_score" in parsed else None,
        "parsed_json": parsed,
    }


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def compute_metrics(gt_rows: list[dict], pred_rows: list[dict]) -> dict:
    total = len(gt_rows)
    exact = 0
    parse_success = 0
    sentiment_distances: list[float] = []
    confidence_abs_errors: list[float] = []

    for gt, pred in zip(gt_rows, pred_rows):
        gt_label = normalize_label(gt["sentiment_label"])
        pred_label = normalize_label(pred["sentiment_label"])
        if pred.get("parse_success"):
            parse_success += 1
        if gt_label == pred_label:
            exact += 1
        sentiment_distances.append(float(label_distance(gt_label, pred_label)))
        gt_conf = gt.get("confidence_score")
        pred_conf = pred.get("confidence_score")
        if gt_conf is not None and pred_conf is not None:
            confidence_abs_errors.append(abs(float(gt_conf) - float(pred_conf)))

    max_distance = max(1, len(LABEL_ORDER) - 1)
    return {
        "count": total,
        "parse_success_rate": round(parse_success / total, 6) if total else None,
        "sentiment_accuracy": round(exact / total, 6) if total else None,
        "sentiment_mae_5way": round(mean(sentiment_distances), 6) if sentiment_distances else None,
        "sentiment_mae_normalized": round(mean([d / max_distance for d in sentiment_distances]), 6)
        if sentiment_distances
        else None,
        "confidence_mae": round(mean(confidence_abs_errors), 6) if confidence_abs_errors else None,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    adapter_path = resolve_latest_adapter_path(args.adapter_path)
    if not adapter_path:
        raise FileNotFoundError(f"Could not find a DPO adapter under: {args.adapter_path}")

    print("[1/4] loading model...", flush=True)
    model, _ = load_unsloth_model(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    loaded = load_lora_adapter_weights(model, adapter_path)
    if not loaded:
        raise RuntimeError(f"Failed to load DPO adapter weights from: {adapter_path}")
    FastLanguageModel.for_inference(model)
    print(f"[model] base={args.base_model}", flush=True)
    print(f"[adapter] using {adapter_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] loading dataset...", flush=True)
    ds = load_from_disk(args.data_dir)
    eval_ds = ds[args.split]
    print(f"[dataset] split={args.split} count={len(eval_ds)}", flush=True)

    print("[3/4] running full inference...", flush=True)
    gt_rows: list[dict] = []
    pred_rows: list[dict] = []
    for idx, ex in enumerate(eval_ds):
        messages = ex["messages"]
        prompt = make_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        do_sample = args.temperature > 0
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=max(args.temperature, 1e-5) if do_sample else None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_len:]
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        gt_parsed = parse_gt_output(messages[-1]["content"])
        pred_parsed = parse_model_output(pred_text)

        shared_meta = {
            "idx": idx,
            "ticker": ex.get("ticker"),
            "factor": ex.get("factor"),
            "form": ex.get("form"),
            "report_date": ex.get("report_date"),
            "source_path": ex.get("source_path"),
        }

        gt_rows.append(
            {
                **shared_meta,
                **gt_parsed,
            }
        )
        pred_rows.append(
            {
                **shared_meta,
                **pred_parsed,
            }
        )

        if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_ds):
            print(f"[progress] {idx + 1}/{len(eval_ds)}", flush=True)

    print("[4/4] writing outputs...", flush=True)
    metrics = compute_metrics(gt_rows, pred_rows)
    metrics["base_model"] = args.base_model
    metrics["adapter_path"] = adapter_path
    metrics["dataset_path"] = args.data_dir
    metrics["split"] = args.split

    (out_dir / "gt.json").write_text(json.dumps(gt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "pred.json").write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {out_dir / 'gt.json'}", flush=True)
    print(f"[saved] {out_dir / 'pred.json'}", flush=True)
    print(f"[saved] {out_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
