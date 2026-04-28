from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

from tqdm import tqdm

try:
    from .common import (
        NLP_CODE_DIR,
        TARGET_MODULES,
        aggregate_filing_extractions,
        apply_filing_limits,
        build_all_factor_questions,
        build_auditor_filing_dataset_row,
        build_factor_row,
        ensure_dir,
        list_raw_filing_specs,
        load_answered_factors,
        load_answered_factor_questions,
        load_task12_indexes,
        load_task13_index,
        metadata_from_task12_obj,
        parse_model_output,
        resolve_existing_path,
    )
except ImportError:
    from common import (
        NLP_CODE_DIR,
        TARGET_MODULES,
        aggregate_filing_extractions,
        apply_filing_limits,
        build_all_factor_questions,
        build_auditor_filing_dataset_row,
        build_factor_row,
        ensure_dir,
        list_raw_filing_specs,
        load_answered_factors,
        load_answered_factor_questions,
        load_task12_indexes,
        load_task13_index,
        metadata_from_task12_obj,
        parse_model_output,
        resolve_existing_path,
    )

import sys

sys.path.insert(0, str(NLP_CODE_DIR))


warnings.filterwarnings("ignore")


BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out"
DEFAULT_FACTOR_DATASET = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5"
DEFAULT_RAW_ROOT = "/gpfs/projects/p32908/data/nlp/MDA_Raw"
DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_OUTPUT_DIR = "/gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/miner_step"
DEFAULT_SPLIT = "validation"
MAX_SEQ_LEN = 32768
MAX_NEW_TOKENS = 768
REPETITION_PENALTY = 1.02
LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_FACTOR_DATASET = str(LOCAL_DATA_DIR / "hf_extract_sft_v5")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local miner inference and build an auditor-ready HF dataset.")
    parser.add_argument("--input-source", choices=["factor_dataset", "raw"], default="factor_dataset")
    parser.add_argument("--factor-dataset", default=DEFAULT_FACTOR_DATASET)
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "validation", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--question-mode", choices=["all", "teacher_answered"], default="all")
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--max-filings", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--repetition-penalty", type=float, default=REPETITION_PENALTY)
    parser.add_argument("--keep-empty-factors", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def build_factor_rows_from_raw_specs(raw_specs: list[dict], *, task12_index: dict, factor_to_qkeys: dict[str, set[str]], question_mode: str) -> tuple[list[dict], list[dict]]:
    global_factor_questions = build_all_factor_questions(factor_to_qkeys)
    factor_rows: list[dict] = []
    selected_manifest: list[dict] = []

    for spec in raw_specs:
        filing_key = spec["filing_key"]
        raw_path = resolve_existing_path(spec["source_path"])
        raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")

        task12_obj = None
        if question_mode == "teacher_answered":
            answered_factors, task12_obj = load_answered_factors(task12_index, filing_key)
            filing_factor_questions = {
                factor: global_factor_questions[factor]
                for factor in answered_factors
                if factor in global_factor_questions
            }
        else:
            filing_factor_questions = global_factor_questions
            task12_path = task12_index.get(filing_key)
            if task12_path is not None:
                task12_obj = json.loads(Path(task12_path).read_text(encoding="utf-8", errors="ignore"))

        if not filing_factor_questions:
            continue

        metadata = metadata_from_task12_obj(task12_obj, filing_key)
        selected_manifest.append(
            {
                "filing_key": filing_key,
                "source_path": str(raw_path),
                "factor_count": len(filing_factor_questions),
                "question_mode": question_mode,
            }
        )
        for factor, factor_questions in filing_factor_questions.items():
            factor_rows.append(
                build_factor_row(
                    raw_text=raw_text,
                    filing_key=filing_key,
                    factor=factor,
                    factor_questions=factor_questions,
                    source_path=str(raw_path),
                    metadata_override=metadata,
                )
            )
    return factor_rows, selected_manifest


def build_factor_rows_from_dataset(dataset_path: str, split: str, *, raw_root: str, task12_index: dict, factor_to_qkeys: dict[str, set[str]], question_mode: str, max_tickers: int, max_filings: int) -> tuple[list[dict], list[dict]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(resolve_existing_path(dataset_path)))
    if split not in ds:
        raise KeyError(f"Split '{split}' not found in factor dataset. Available: {list(ds.keys())}")

    filings_by_key: dict[tuple[str, str, str], dict] = {}
    for row in ds[split]:
        filing_key = (str(row["ticker"]), str(row["form"]), str(row["report_date"]))
        filings_by_key.setdefault(
            filing_key,
            {
                "filing_key": filing_key,
                "ticker": filing_key[0],
                "form": filing_key[1],
                "report_date": filing_key[2],
                "split": split,
                "source_path": row.get("source_path") or "",
            },
        )

    specs = sorted(filings_by_key.values(), key=lambda x: (x["ticker"], x["report_date"], x["form"]))
    specs = apply_filing_limits(specs, max_tickers=max_tickers, max_filings=max_filings)

    patched_specs = []
    for spec in specs:
        if spec["source_path"]:
            patched_specs.append(spec)
            continue
        ticker, form, report_date = spec["filing_key"]
        raw_candidate = Path(raw_root) / ticker / form / f"{ticker}_{form}_{report_date}.md"
        spec = dict(spec)
        spec["source_path"] = str(raw_candidate)
        patched_specs.append(spec)

    return build_factor_rows_from_raw_specs(
        patched_specs,
        task12_index=task12_index,
        factor_to_qkeys=factor_to_qkeys,
        question_mode=question_mode,
    )


def make_prompt(messages, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def batched(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    args = build_arg_parser().parse_args()
    import datasets
    import torch
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import AutoTokenizer
    from unsloth import FastLanguageModel

    from train_common import load_lora_adapter_weights, load_unsloth_model, resolve_latest_adapter_path

    datasets.disable_caching()
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

    out_dir = Path(args.output_dir)
    auditor_ds_dir = out_dir / "auditor_hf_dataset"
    ensure_dir(out_dir)

    print("[1/6] indexing source data...", flush=True)
    task12_index, factor_to_qkeys = load_task12_indexes(args.extract_root)
    task13_index = load_task13_index(args.extract_root)

    if args.input_source == "raw":
        raw_specs = [x for x in list_raw_filing_specs(args.raw_root) if x["split"] == args.split]
        raw_specs = apply_filing_limits(raw_specs, max_tickers=args.max_tickers, max_filings=args.max_filings)
        factor_rows, selected_manifest = build_factor_rows_from_raw_specs(
            raw_specs,
            task12_index=task12_index,
            factor_to_qkeys=factor_to_qkeys,
            question_mode=args.question_mode,
        )
    else:
        factor_rows, selected_manifest = build_factor_rows_from_dataset(
            args.factor_dataset,
            args.split,
            raw_root=args.raw_root,
            task12_index=task12_index,
            factor_to_qkeys=factor_to_qkeys,
            question_mode=args.question_mode,
            max_tickers=args.max_tickers,
            max_filings=args.max_filings,
        )

    if not factor_rows:
        raise ValueError("No factor rows selected. Check --split / --max-tickers / --max-filings / --question-mode.")

    (out_dir / "selected_filings.json").write_text(
        json.dumps(selected_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[selection] filings={len(selected_manifest)} factor_rows={len(factor_rows)}", flush=True)

    print("[2/6] loading base model + adapter...", flush=True)
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

    print("[3/6] running miner factor-level inference...", flush=True)
    pred_rows: list[dict] = []
    progress = tqdm(total=len(factor_rows), desc="Miner inference", unit="factor")
    for start_idx, batch_rows in batched(factor_rows, max(1, args.batch_size)):
        prompts = [make_prompt(row["messages"], tokenizer) for row in batch_rows]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for offset, row in enumerate(batch_rows):
            idx = start_idx + offset + 1
            pred_text = tokenizer.decode(outputs[offset][prompt_lens[offset] :], skip_special_tokens=False)
            pred = parse_model_output(pred_text)
            pred_rows.append(
                {
                    "idx": idx - 1,
                    "ticker": row["ticker"],
                    "form": row["form"],
                    "report_date": row["report_date"],
                    "factor": row["factor"],
                    "question_count": row["question_count"],
                    "factor_questions": row["factor_questions"],
                    "prompt": row["prompt"],
                    "parse_success": pred["parse_success"],
                    "raw_output": pred["raw_text"],
                    "pred_json": pred["parsed_json"],
                    "pred_extractions": pred["extractions"],
                    "pred_extraction_count": len(pred["extractions"]),
                    "metadata": row["metadata"],
                    "source_path": row["source_path"],
                }
            )
            if args.verbose and pred["extractions"]:
                print(
                    f"[factor {idx}/{len(factor_rows)}] {row['ticker']} {row['report_date']} {row['factor']} extractions={len(pred['extractions'])}",
                    flush=True,
                )
        progress.update(len(batch_rows))
        if progress.n % 25 == 0 or progress.n == len(factor_rows):
            print(f"[progress] factor_rows={progress.n}/{len(factor_rows)}", flush=True)
    progress.close()

    print("[4/6] aggregating miner outputs...", flush=True)
    filing_aggregates = aggregate_filing_extractions(pred_rows)
    (out_dir / "miner_pred_rows.json").write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "miner_filing_aggregates.json").write_text(
        json.dumps(filing_aggregates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[5/6] building auditor-ready HF dataset...", flush=True)
    auditor_rows: list[dict] = []
    auditor_preview: list[dict] = []
    for filing_item in filing_aggregates:
        filing_key = tuple(filing_item["filing_key"])
        ticker, form, report_date = filing_key
        task12_path = task12_index.get(filing_key)
        task13_bundle = task13_index.get(filing_key)
        sector = None
        if task13_bundle:
            sector = task13_bundle["metadata"].get("sector")
        elif filing_item["metadata"]:
            sector = filing_item["metadata"].get("sector")

        factor_groups = {}
        for factor_row in filing_item["factor_rows"]:
            factor_groups[factor_row["factor"]] = factor_row["pred_extractions"]

        factor_blocks: list[dict] = []
        for factor, evidence_used in sorted(factor_groups.items()):
            if not evidence_used and not args.keep_empty_factors:
                continue
            historical_context = []
            if task13_bundle:
                teacher_audit = task13_bundle["factor_map"].get(factor)
                historical_context = (teacher_audit or {}).get("historical_context", []) or []
            factor_blocks.append(
                {
                    "factor": factor,
                    "evidence_used": evidence_used,
                    "historical_context": historical_context,
                }
            )

        if not factor_blocks:
            continue

        source_task13_file = task13_bundle["path"] if task13_bundle else None
        dataset_row = build_auditor_filing_dataset_row(
            filing_key=filing_key,
            factor_blocks=factor_blocks,
            sector=sector,
            source_raw_path=filing_item["source_raw_path"],
            source_task12_file=str(task12_path) if task12_path is not None else None,
            source_task13_file=source_task13_file,
            source_miner_output_dir=str(out_dir),
        )
        auditor_rows.append(dataset_row)
        auditor_preview.append(
            {
                "ticker": ticker,
                "form": form,
                "report_date": report_date,
                "factor_count": dataset_row["factor_count"],
                "evidence_count": dataset_row["evidence_count"],
                "historical_count": dataset_row["historical_count"],
            }
        )

    if not auditor_rows:
        raise ValueError("Miner finished but no auditor rows were built. Consider --keep-empty-factors or inspect miner outputs.")

    ensure_dir(auditor_ds_dir)
    auditor_dataset = DatasetDict({args.split: Dataset.from_list(auditor_rows)})
    auditor_dataset.save_to_disk(str(auditor_ds_dir))
    (out_dir / "auditor_rows_preview.json").write_text(
        json.dumps(auditor_preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[6/6] writing summary...", flush=True)
    nonempty_factor_rows = sum(1 for row in pred_rows if row["pred_extraction_count"] > 0)
    summary = {
        "input_source": args.input_source,
        "split": args.split,
        "question_mode": args.question_mode,
        "selected_filing_count": len(selected_manifest),
        "miner_factor_row_count": len(pred_rows),
        "miner_nonempty_factor_row_count": nonempty_factor_rows,
        "miner_filing_count": len(filing_aggregates),
        "auditor_row_count": len(auditor_rows),
        "auditor_dataset_dir": str(auditor_ds_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
