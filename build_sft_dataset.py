from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset, DatasetDict

from task13_dataset_common import (
    SYSTEM_AUDITOR,
    assign_fixed_split,
    build_sft_response,
    build_user_prompt,
    iter_task13_files,
    parse_task13_filename,
)


DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_OUTPUT = "/gpfs/projects/p32908/data/nlp/hf_cot_sft"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build task1.3 CoT SFT datasets from local audit JSONs.")
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rationale-style", choices=["full", "concise"], default="full")
    parser.add_argument("--min-evidence", type=int, default=1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    extract_root = Path(args.extract_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = {"train": [], "validation": [], "test": []}

    for audit_path in iter_task13_files(extract_root):
        parsed = parse_task13_filename(audit_path)
        if not parsed:
            continue
        ticker, form, report_date = parsed
        split = assign_fixed_split(int(report_date[:4]))
        if not split:
            continue

        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError as exc:
            logging.warning("Skipping invalid JSON %s: %s", audit_path, exc)
            continue

        metadata = payload.get("metadata", {}) or {}
        sector = metadata.get("sector")

        for audit in payload.get("factor_audits", []) or []:
            evidence_count = len(audit.get("evidence_used", []) or [])
            if evidence_count < args.min_evidence:
                continue

            factor = audit.get("factor", "unknown")
            user_prompt = build_user_prompt(audit, ticker, form, report_date)
            assistant = build_sft_response(audit, args.rationale_style)
            result = audit.get("audit_result", {}) or {}

            rows[split].append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_AUDITOR},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": assistant},
                    ],
                    "task": "task13_auditor_cot_sft",
                    "ticker": ticker,
                    "form": form,
                    "report_date": report_date,
                    "parsed_report_date": report_date,
                    "year": int(report_date[:4]),
                    "factor": factor,
                    "sector": sector,
                    "evidence_count": evidence_count,
                    "historical_count": len(audit.get("historical_context", []) or []),
                    "teacher_label": result.get("sentiment_label"),
                    "normalized_label": json.loads(assistant)["sentiment_label"],
                    "teacher_confidence": result.get("confidence_score"),
                    "rationale_style": args.rationale_style,
                    "source_path": str(audit_path),
                }
            )

    dataset = DatasetDict({split: Dataset.from_list(items) for split, items in rows.items()})
    dataset.save_to_disk(str(output_dir))
    logging.info(
        "Saved SFT dataset to %s | train=%d validation=%d test=%d",
        output_dir,
        len(rows["train"]),
        len(rows["validation"]),
        len(rows["test"]),
    )


if __name__ == "__main__":
    main()
