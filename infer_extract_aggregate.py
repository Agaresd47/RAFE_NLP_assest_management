from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import unsloth
import datasets
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel

from train_common import load_lora_adapter_weights, load_unsloth_model, resolve_latest_adapter_path


datasets.disable_caching()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out"
QUESTION_SOURCE_DATASET = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5"
EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 4096
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
RAW_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9._-]+)_(?P<form>10-K|10-Q)_(?P<date>\d{4}-\d{2}-\d{2})\.md$",
    re.IGNORECASE,
)
SYSTEM_MINER = (
    "You are a Financial Data Engineer. "
    "Extract original quotes from the provided source text that answer the specific questions in the schema. "
    "For each question, if an answer is found, provide the original_quote and a relevance_confidence from 0 to 1. "
    "If no answer is found, ignore the question entirely. "
    "Output only valid JSON."
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate full-schema extraction by looping single-question prompts.")
    parser.add_argument("--raw-file", default="")
    parser.add_argument("--ticker", default="")
    parser.add_argument("--form", default="")
    parser.add_argument("--date", default="")
    parser.add_argument("--raw-root", default="/gpfs/projects/p32908/data/nlp/MDA_Raw")
    parser.add_argument("--merged-model", default="")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--question-source-dataset", default=QUESTION_SOURCE_DATASET)
    parser.add_argument("--extract-root", default=EXTRACT_ROOT)
    parser.add_argument("--sector", default="info tech")
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--qnum", type=int, default=0)
    parser.add_argument("--question-key", default="")
    parser.add_argument("--only-known-answered", action="store_true")
    parser.add_argument("--top_conf_percent", type=float, default=1.0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-json", default="")
    return parser


def parse_raw_filename(path: Path) -> dict[str, str]:
    match = RAW_RE.match(path.name)
    if not match:
        raise ValueError(f"Could not parse ticker/form/date from filename: {path.name}")
    return {
        "ticker": match.group("ticker").upper(),
        "filing": match.group("form").upper(),
        "report_date": match.group("date"),
    }


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str)
    candidates = [path]

    raw = str(path)
    if raw.startswith("/gpfs/projects/"):
        candidates.append(Path(raw.replace("/gpfs/projects/", "/projects/", 1)))
    elif raw.startswith("/projects/"):
        candidates.append(Path(raw.replace("/projects/", "/gpfs/projects/", 1)))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find raw file at any known prefix variant: {path_str}")


def resolve_raw_file(args) -> Path:
    if args.raw_file:
        return resolve_existing_path(args.raw_file)

    if args.ticker and args.form and args.date:
        ticker = str(args.ticker).upper()
        form = str(args.form).upper()
        filename = f"{ticker}_{form}_{args.date}.md"
        candidates = [
            Path(args.raw_root) / ticker / form / filename,
            Path(args.raw_root) / filename,
        ]
        for candidate in candidates:
            try:
                return resolve_existing_path(str(candidate))
            except FileNotFoundError:
                continue
        raise FileNotFoundError(
            "Could not resolve raw file from --ticker/--form/--date. "
            f"Tried under raw root: {args.raw_root}"
        )

    raise ValueError("Provide either --raw-file or all of --ticker --form --date.")


def safe_json_load(text: str):
    cleaned = str(text or "").replace("<|im_end|>", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_question_specs(extract_root: str) -> list[dict]:
    root = resolve_existing_path(extract_root)
    seen: dict[tuple[str, str], dict] = {}
    for path in root.rglob("*.json"):
        if "TASK12_EXTRACTIONS" not in path.name.upper():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for item in obj.get("extractions", []):
            if not isinstance(item, dict):
                continue
            factor = str(item.get("factor") or "").strip()
            question_key = str(item.get("question_key") or "").strip()
            if not factor or not question_key:
                continue
            key = (factor, question_key)
            if key in seen:
                continue
            question_text = question_key.split("::", 1)[1].strip() if "::" in question_key else question_key
            seen[key] = {
                "factor": factor,
                "question_key": question_key,
                "question_text": question_text,
            }
    def sort_key(item: dict):
        q = item["question_key"]
        m = re.match(r"Q(\d+)::", q)
        qnum = int(m.group(1)) if m else 9999
        return (qnum, item["factor"], q)
    return sorted(seen.values(), key=sort_key)


def resolve_task12_extract_file(extract_root: str, *, ticker: str, filing: str, report_date: str) -> Path:
    root = resolve_existing_path(extract_root)
    ticker = str(ticker).upper()
    filing = str(filing).upper()
    yyyy, mm, dd = report_date.split("-")
    filename = f"{ticker}_{mm}-{dd}-{yyyy}_{filing}_TASK12_EXTRACTIONS.json"
    candidates = [
        root / ticker / filename,
        root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve TASK12 extraction file for {ticker} {filing} {report_date} under {extract_root}"
    )


def load_answered_question_specs_for_filing(extract_root: str, *, ticker: str, filing: str, report_date: str) -> list[dict]:
    path = resolve_task12_extract_file(
        extract_root,
        ticker=ticker,
        filing=filing,
        report_date=report_date,
    )
    obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    seen: dict[tuple[str, str], dict] = {}
    for item in obj.get("extractions", []):
        if not isinstance(item, dict):
            continue
        factor = str(item.get("factor") or "").strip()
        question_key = str(item.get("question_key") or "").strip()
        if not factor or not question_key:
            continue
        key = (factor, question_key)
        if key in seen:
            continue
        question_text = question_key.split("::", 1)[1].strip() if "::" in question_key else question_key
        seen[key] = {
            "factor": factor,
            "question_key": question_key,
            "question_text": question_text,
        }

    def sort_key(item: dict):
        q = item["question_key"]
        m = re.match(r"Q(\d+)::", q)
        qnum = int(m.group(1)) if m else 9999
        return (qnum, item["factor"], q)

    return sorted(seen.values(), key=sort_key)


def build_prompt(raw_text: str, ticker: str, filing: str, report_date: str, factor: str, question_key: str, question_text: str) -> str:
    return (
        f"Task: Miner (1.2)\n"
        f"Ticker: {ticker} | Filing: {filing} | Date: {report_date}\n"
        f"Target Factor: {factor}\n"
        f"Target Question Key: {question_key}\n"
        f"Target Question: {question_text}\n\n"
        "Instructions:\n"
        "- Return only valid JSON.\n"
        "- Output only the metadata and extractions schema shown by prior examples.\n"
        "- Extract only original quotes from the filing text.\n"
        "- Include only quotes relevant to the single target question above.\n"
        "- Do not answer other questions.\n"
        "- Do not summarize; copy exact supporting quotes.\n\n"
        f"Text:\n{raw_text}"
    )


def make_chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MINER},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def dedupe_aggregated_extractions(extractions: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in extractions:
        grouped[(str(item.get("factor")), str(item.get("question_key")))].append(item)

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


def keep_top_confidence_fraction(extractions: list[dict], top_conf_percent: float) -> list[dict]:
    if not extractions:
        return []
    if top_conf_percent <= 0:
        return []
    if top_conf_percent >= 1:
        return extractions

    sorted_items = sorted(
        extractions,
        key=lambda x: float(x.get("relevance_confidence", 0.0)),
        reverse=True,
    )
    keep_n = max(1, int(len(sorted_items) * top_conf_percent))
    return sorted_items[:keep_n]


def filter_question_specs(question_specs: list[dict], *, qnum: int, question_key: str, limit_questions: int) -> list[dict]:
    filtered = question_specs

    if question_key:
        filtered = [item for item in filtered if item["question_key"] == question_key]
    elif qnum > 0:
        prefix = f"Q{qnum}::"
        filtered = [item for item in filtered if item["question_key"].startswith(prefix)]

    if limit_questions > 0:
        filtered = filtered[:limit_questions]
    return filtered


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_path = resolve_raw_file(args)
    raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
    meta = parse_raw_filename(raw_path)
    metadata = {
        "ticker": meta["ticker"],
        "filing": meta["filing"],
        "year": int(meta["report_date"][:4]),
        "sector": args.sector,
        "report_date": meta["report_date"],
    }

    print("[1/4] loading model...", flush=True)
    if args.merged_model:
        merged_model_path = str(resolve_existing_path(args.merged_model))
        model = AutoModelForCausalLM.from_pretrained(
            merged_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            merged_model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        print(f"[merged] loaded=True path={merged_model_path}", flush=True)
    else:
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
    tokenizer.padding_side = "left"
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] loading question specs...", flush=True)
    if args.only_known_answered:
        question_specs = load_answered_question_specs_for_filing(
            args.extract_root,
            ticker=metadata["ticker"],
            filing=metadata["filing"],
            report_date=metadata["report_date"],
        )
    else:
        question_specs = load_question_specs(args.extract_root)
    question_specs = filter_question_specs(
        question_specs,
        qnum=args.qnum,
        question_key=args.question_key,
        limit_questions=args.limit_questions,
    )
    if not question_specs:
        raise ValueError("No question specs selected. Check --qnum / --question-key / --limit-questions.")
    print(f"question_count={len(question_specs)}", flush=True)

    print("[3/4] running per-question extraction...", flush=True)
    aggregated: list[dict] = []
    for idx, spec in enumerate(question_specs, 1):
        if args.verbose:
            print(
                f"[q {idx}/{len(question_specs)}] {spec['question_key']} | factor={spec['factor']}",
                flush=True,
            )
        prompt = build_prompt(
            raw_text=raw_text,
            ticker=metadata["ticker"],
            filing=metadata["filing"],
            report_date=metadata["report_date"],
            factor=spec["factor"],
            question_key=spec["question_key"],
            question_text=spec["question_text"],
        )
        chat_text = make_chat_text(tokenizer, prompt)
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1] :]
        pred_text = tokenizer.decode(generated, skip_special_tokens=False)
        obj = safe_json_load(pred_text)
        if not isinstance(obj, dict):
            if args.verbose:
                preview = pred_text[:300].replace("\n", " ")
                print(f"  parse_failed preview={preview}", flush=True)
            continue
        extractions = obj.get("extractions", [])
        if not isinstance(extractions, list):
            if args.verbose:
                print("  invalid_extractions_field", flush=True)
            continue
        if len(extractions) == 0:
            if args.verbose:
                print("  returned_empty", flush=True)
            continue
        if args.verbose:
            print(f"  returned={len(extractions)}", flush=True)
        for item in extractions:
            if not isinstance(item, dict):
                continue
            factor = item.get("factor") or spec["factor"]
            question_key = item.get("question_key") or spec["question_key"]
            aggregated.append(
                {
                    "factor": factor,
                    "question_key": question_key,
                    "original_quote": item.get("original_quote", ""),
                    "relevance_confidence": item.get("relevance_confidence", 0.0),
                }
            )
            if args.verbose:
                quote = normalize_whitespace(item.get("original_quote", ""))[:180]
                conf = item.get("relevance_confidence", 0.0)
                print(f"    conf={conf} quote={quote}", flush=True)
        if idx % 10 == 0 or idx == len(question_specs):
            print(f"processed {idx}/{len(question_specs)} questions | raw_extractions={len(aggregated)}", flush=True)

    print("[4/4] aggregating...", flush=True)
    deduped = dedupe_aggregated_extractions(aggregated)
    filtered = keep_top_confidence_fraction(deduped, args.top_conf_percent)
    merged = {
        "metadata": metadata,
        "extractions": filtered,
    }
    print(json.dumps(merged, ensure_ascii=False, indent=2), flush=True)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
