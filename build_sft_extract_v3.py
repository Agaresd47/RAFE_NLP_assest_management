from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from build_dataset import (
    SYSTEM_MINER,
    assign_fixed_split,
    build_user_prompt_miner,
    load_questions,
    parse_extract_filename,
    parse_raw_filename,
    sanitize_miner_output,
)


DEFAULT_RAW_ROOT = "/gpfs/projects/p32908/data/nlp/MDA_Raw"
DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_QUESTIONS = "/gpfs/projects/p32908/data/nlp/tech_sentiment_questions.json"
DEFAULT_OUTPUT = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v3"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build extract-only SFT v3 dataset.")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--min-extractions", type=int, default=1)
    parser.add_argument("--max-approx-tokens", type=int, default=259935)
    return parser


def build_row(
    *,
    raw_text: str,
    t12_obj: dict,
    questions: dict,
    ticker: str,
    form: str,
    report_date: str,
    source_path: str,
) -> dict:
    prompt = build_user_prompt_miner(
        raw_text=raw_text,
        questions=questions,
        ticker=ticker,
        form=form,
        report_date=report_date,
    )
    assistant = sanitize_miner_output(t12_obj)
    extractions = t12_obj.get("extractions", [])
    metadata = t12_obj.get("metadata") if isinstance(t12_obj.get("metadata"), dict) else {}
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
        "extraction_count": len(extractions) if isinstance(extractions, list) else 0,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "source_path": source_path,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    raw_root = Path(args.raw_root)
    extract_root = Path(args.extract_root)
    questions = load_questions(Path(args.questions))

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

            row = build_row(
                raw_text=raw_text,
                t12_obj=t12_obj,
                questions=questions,
                ticker=key[0],
                form=key[1],
                report_date=key[2],
                source_path=str(t12_path),
            )
            split_rows[split].append(row)
        except Exception as exc:
            logging.error("Failed to process %s: %s", t12_path, exc)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in split_rows.items()})
    dataset.save_to_disk(str(output_dir))
    logging.info(
        "Saved extract SFT v3 dataset to %s | train=%d validation=%d test=%d",
        output_dir,
        len(split_rows["train"]),
        len(split_rows["validation"]),
        len(split_rows["test"]),
    )
    if split_rows["train"]:
        logging.info("Sample ticker=%s extraction_count=%s", split_rows["train"][0]["ticker"], split_rows["train"][0]["extraction_count"])


if __name__ == "__main__":
    main()
