from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from build_dataset import (
    SYSTEM_MINER,
    assign_fixed_split,
    load_questions,
    parse_extract_filename,
    parse_raw_filename,
)


DEFAULT_RAW_ROOT = "/gpfs/projects/p32908/data/nlp/MDA_Raw"
DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_QUESTIONS = "/gpfs/projects/p32908/data/nlp/tech_sentiment_questions.json"
DEFAULT_OUTPUT = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v4"
DEFAULT_MAX_QUOTES_PER_SAMPLE = 5
DEFAULT_MAX_TRAIN_ROWS_PER_QKEY = 48


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build extract-only SFT v4 dataset with per-question rows.")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-extractions", type=int, default=1)
    parser.add_argument("--max-approx-tokens", type=int, default=259935)
    parser.add_argument("--max-quotes-per-sample", type=int, default=DEFAULT_MAX_QUOTES_PER_SAMPLE)
    parser.add_argument("--max-train-rows-per-qkey", type=int, default=DEFAULT_MAX_TRAIN_ROWS_PER_QKEY)
    return parser


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def dedupe_grouped_extractions(extractions: list[dict], max_quotes_per_sample: int) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []

    def score(item: dict) -> float:
        try:
            return float(item.get("relevance_confidence", 0.0))
        except Exception:
            return 0.0

    for extraction in sorted(extractions, key=score, reverse=True):
        quote = normalize_whitespace(extraction.get("original_quote", ""))
        if not quote:
            continue
        if quote in seen:
            continue
        seen.add(quote)
        deduped.append(
            {
                "factor": extraction.get("factor"),
                "question_key": extraction.get("question_key"),
                "original_quote": extraction.get("original_quote"),
                "relevance_confidence": round(score(extraction), 4),
            }
        )
        if len(deduped) >= max_quotes_per_sample:
            break
    return deduped


def parse_question_text(question_key: str) -> str:
    question_key = str(question_key or "")
    if "::" in question_key:
        return question_key.split("::", 1)[1].strip()
    return question_key


def build_single_question_prompt(
    *,
    raw_text: str,
    ticker: str,
    form: str,
    report_date: str,
    factor: str,
    question_key: str,
) -> str:
    question_text = parse_question_text(question_key)
    return (
        f"Task: Miner (1.2)\n"
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}\n"
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


def build_row(
    *,
    raw_text: str,
    metadata: dict,
    grouped_extractions: list[dict],
    ticker: str,
    form: str,
    report_date: str,
    source_path: str,
) -> dict:
    factor = grouped_extractions[0]["factor"]
    question_key = grouped_extractions[0]["question_key"]
    prompt = build_single_question_prompt(
        raw_text=raw_text,
        ticker=ticker,
        form=form,
        report_date=report_date,
        factor=factor,
        question_key=question_key,
    )
    assistant_obj = {
        "metadata": metadata,
        "extractions": grouped_extractions,
    }
    assistant = json.dumps(assistant_obj, ensure_ascii=False, indent=2)
    year = int(report_date[:4])
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MINER},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "prompt": prompt,
        "response": assistant,
        "ticker": ticker,
        "form": form,
        "report_date": report_date,
        "year": year,
        "factor": factor,
        "question_key": question_key,
        "extraction_count": len(grouped_extractions),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "source_path": source_path,
    }


def deterministic_keep_key(row: dict) -> str:
    raw = "||".join(
        [
            str(row.get("ticker")),
            str(row.get("report_date")),
            str(row.get("factor")),
            str(row.get("question_key")),
            str(row.get("source_path")),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def rebalance_train_rows(rows: list[dict], max_rows_per_qkey: int) -> list[dict]:
    if max_rows_per_qkey <= 0:
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_key"])].append(row)

    balanced: list[dict] = []
    for _, part in grouped.items():
        part = sorted(part, key=deterministic_keep_key)
        balanced.extend(part[:max_rows_per_qkey])
    balanced.sort(key=deterministic_keep_key)
    return balanced


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    raw_root = Path(args.raw_root)
    extract_root = Path(args.extract_root)
    _questions = load_questions(Path(args.questions))

    raw_index = {
        parsed: path
        for path in raw_root.rglob("*.md")
        if (parsed := parse_raw_filename(path))
    }
    task12_index = {}
    for path in extract_root.rglob("*.json"):
        parsed = parse_extract_filename(path)
        if not parsed:
            continue
        key, kind = parsed[:3], parsed[3]
        if kind == "TASK12_EXTRACTIONS":
            task12_index[key] = path

    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for key, t12_path in sorted(task12_index.items()):
        raw_path = raw_index.get(key)
        if raw_path is None:
            logging.warning("No raw filing found for %s, skipping.", key)
            continue

        split = assign_fixed_split(int(key[2][:4]))
        if split is None:
            continue

        try:
            raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
            approx_tokens = len(raw_text) // 4
            if approx_tokens > args.max_approx_tokens:
                logging.warning("Skipping overlong raw filing %s (~%d tokens).", key, approx_tokens)
                continue

            t12_obj = json.loads(t12_path.read_text(encoding="utf-8", errors="ignore"))
            extractions = t12_obj.get("extractions", [])
            if not isinstance(extractions, list) or len(extractions) < args.min_extractions:
                continue

            grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
            for extraction in extractions:
                factor = extraction.get("factor")
                question_key = extraction.get("question_key")
                if not factor or not question_key:
                    continue
                grouped[(str(factor), str(question_key))].append(extraction)

            metadata = t12_obj.get("metadata") if isinstance(t12_obj.get("metadata"), dict) else {}
            for (_, _), group_items in grouped.items():
                clean_group = dedupe_grouped_extractions(group_items, args.max_quotes_per_sample)
                if not clean_group:
                    continue
                row = build_row(
                    raw_text=raw_text,
                    metadata=metadata,
                    grouped_extractions=clean_group,
                    ticker=key[0],
                    form=key[1],
                    report_date=key[2],
                    source_path=str(t12_path),
                )
                split_rows[split].append(row)
        except Exception as exc:
            logging.error("Failed to process %s: %s", t12_path, exc)

    before_train = len(split_rows["train"])
    split_rows["train"] = rebalance_train_rows(split_rows["train"], args.max_train_rows_per_qkey)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in split_rows.items()})
    dataset.save_to_disk(str(output_dir))
    logging.info(
        "Saved extract SFT v4 dataset to %s | train=%d (before rebalance=%d) validation=%d test=%d",
        output_dir,
        len(split_rows["train"]),
        before_train,
        len(split_rows["validation"]),
        len(split_rows["test"]),
    )
    if split_rows["train"]:
        sample = split_rows["train"][0]
        logging.info(
            "Sample qkey=%s factor=%s extraction_count=%s",
            sample["question_key"],
            sample["factor"],
            sample["extraction_count"],
        )


if __name__ == "__main__":
    main()
