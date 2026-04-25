import unsloth
import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from train_common import load_lora_adapter_weights, load_unsloth_model, resolve_latest_adapter_path


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5"
OUTPUT_DIR = "/gpfs/projects/p32908/nlp_result/miner_full"
SPLIT = "validation"
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.0
REPETITION_PENALTY = 1.02
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
    parser.add_argument("--repetition_penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--keep_empty", action="store_true")
    return parser


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_model_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")
    if path.is_dir() and (path / "adapter_config.json").exists() and not (path / "config.json").exists():
        checkpoints = sorted(
            [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-", 1)[1]),
        )
        if checkpoints:
            return str(checkpoints[-1])
    return str(path)


def make_prompt(messages, tokenizer) -> str:
    prompt_messages = messages[:2]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_model_output(text: str) -> dict:
    raw = str(text or "")
    clean = raw.replace("<|im_end|>", "").strip()
    clean = re.sub(r"<think>\s*.*?\s*</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()

    parsed = safe_json_load(clean)
    if parsed is None:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = safe_json_load(clean[start : end + 1])
    if not isinstance(parsed, dict):
        parsed = {}

    extractions = parsed.get("extractions", [])
    if not isinstance(extractions, list):
        extractions = []

    return {
        "raw_text": raw,
        "parsed_json": parsed,
        "metadata": parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {},
        "extractions": extractions,
        "parse_success": bool(parsed),
    }


def parse_gt_output(text: str) -> dict:
    parsed = safe_json_load(str(text or ""))
    if not isinstance(parsed, dict):
        parsed = {}
    extractions = parsed.get("extractions", [])
    if not isinstance(extractions, list):
        extractions = []
    return {
        "parsed_json": parsed,
        "metadata": parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {},
        "extractions": extractions,
    }


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def dedupe_extractions(extractions: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in extractions:
        grouped[(str(item.get("factor", "")), str(item.get("question_key", "")))].append(item)

    output: list[dict] = []
    for key in sorted(grouped.keys()):
        seen_quotes: set[str] = set()
        items = sorted(
            grouped[key],
            key=lambda x: float(x.get("relevance_confidence", 0.0)),
            reverse=True,
        )
        for item in items:
            norm_quote = normalize_whitespace(item.get("original_quote", ""))
            if not norm_quote or norm_quote in seen_quotes:
                continue
            seen_quotes.add(norm_quote)
            output.append(
                {
                    "factor": item.get("factor"),
                    "question_key": item.get("question_key"),
                    "original_quote": item.get("original_quote"),
                    "relevance_confidence": round(float(item.get("relevance_confidence", 0.0)), 4),
                }
            )
    return output


def aggregate_by_filing(pred_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str], dict] = {}
    for row in pred_rows:
        key = (str(row.get("ticker", "")), str(row.get("form", "")), str(row.get("report_date", "")))
        bucket = buckets.setdefault(
            key,
            {
                "metadata": {
                    "ticker": row.get("ticker"),
                    "filing": row.get("form"),
                    "year": int(str(row.get("report_date", ""))[:4]) if row.get("report_date") else None,
                    "sector": "info tech",
                    "report_date": row.get("report_date"),
                },
                "extractions": [],
            },
        )
        bucket["extractions"].extend(row.get("pred_extractions", []))

    merged = []
    for key in sorted(buckets.keys()):
        item = buckets[key]
        item["extractions"] = dedupe_extractions(item["extractions"])
        merged.append(item)
    return merged


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    print("[1/4] loading model...", flush=True)
    model, _ = load_unsloth_model(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
    adapter_path = resolve_latest_adapter_path(args.adapter_path)
    if not adapter_path:
        raise FileNotFoundError(f"No adapter checkpoint found under: {args.adapter_path}")
    loaded = load_lora_adapter_weights(model, adapter_path)
    print(f"[adapter] loaded={loaded} path={adapter_path}", flush=True)
    FastLanguageModel.for_inference(model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    print(f"[model] using base={args.base_model}", flush=True)

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
    if args.limit > 0:
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))
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
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        pred_text = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=False)

        gt = parse_gt_output(messages[-1]["content"])
        pred = parse_model_output(pred_text)

        shared = {
            "idx": idx,
            "ticker": ex.get("ticker"),
            "form": ex.get("form"),
            "report_date": ex.get("report_date"),
            "factor": ex.get("factor"),
            "question_count": ex.get("question_count"),
            "gt_extraction_count": len(gt["extractions"]),
            "pred_extraction_count": len(pred["extractions"]),
            "target_mode": ex.get("target_mode"),
            "source_path": ex.get("source_path"),
        }

        gt_rows.append({**shared, "gt_json": gt["parsed_json"], "gt_extractions": gt["extractions"]})
        pred_rows.append(
            {
                **shared,
                "prompt": prompt,
                "raw_output": pred["raw_text"],
                "parse_success": pred["parse_success"],
                "pred_json": pred["parsed_json"],
                "pred_extractions": pred["extractions"],
            }
        )

        if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_ds):
            print(f"[progress] {idx + 1}/{len(eval_ds)}", flush=True)

    if not args.keep_empty:
        pred_rows = [row for row in pred_rows if row["pred_extraction_count"] > 0]

    filing_aggregates = aggregate_by_filing(pred_rows)

    print("[4/4] writing outputs...", flush=True)
    (out_dir / "miner_pred_rows.json").write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "miner_gt_rows.json").write_text(json.dumps(gt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "miner_filing_aggregates.json").write_text(
        json.dumps(filing_aggregates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "split": args.split,
        "input_rows": len(eval_ds),
        "kept_pred_rows": len(pred_rows),
        "nonempty_rate": round(len(pred_rows) / len(eval_ds), 6) if len(eval_ds) else None,
        "filing_count": len(filing_aggregates),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
