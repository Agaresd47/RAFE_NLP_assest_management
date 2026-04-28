from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict

try:
    from .common import (
        build_auditor_filing_dataset_row,
        ensure_dir,
        load_task12_indexes,
        load_task13_index,
        resolve_existing_path,
    )
except ImportError:
    from common import (
        build_auditor_filing_dataset_row,
        ensure_dir,
        load_task12_indexes,
        load_task13_index,
        resolve_existing_path,
    )


DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build filing-level auditor HF dataset from saved miner outputs.")
    parser.add_argument("--miner-output-dir", required=True)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--keep-empty-factors", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    miner_output_dir = resolve_existing_path(args.miner_output_dir)
    output_dir = Path(args.output_dir) if args.output_dir else miner_output_dir / "auditor_hf_dataset"

    filing_aggregates_path = miner_output_dir / "miner_filing_aggregates.json"
    if not filing_aggregates_path.exists():
        raise FileNotFoundError(f"Missing miner filing aggregates: {filing_aggregates_path}")

    filing_aggregates = json.loads(filing_aggregates_path.read_text(encoding="utf-8"))
    task12_index, _ = load_task12_indexes(args.extract_root)
    task13_index = load_task13_index(args.extract_root)

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
        elif filing_item.get("metadata"):
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
            source_miner_output_dir=str(miner_output_dir),
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
        raise ValueError("No auditor filing rows were built from saved miner outputs.")

    ensure_dir(output_dir)
    auditor_dataset = DatasetDict({args.split: Dataset.from_list(auditor_rows)})
    auditor_dataset.save_to_disk(str(output_dir))
    (miner_output_dir / "auditor_rows_preview.json").write_text(
        json.dumps(auditor_preview, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "split": args.split,
        "miner_output_dir": str(miner_output_dir),
        "auditor_dataset_dir": str(output_dir),
        "auditor_row_count": len(auditor_rows),
    }
    (miner_output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
