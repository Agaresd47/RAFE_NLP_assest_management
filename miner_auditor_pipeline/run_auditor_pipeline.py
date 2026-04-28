from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

try:
    from .common import NLP_CODE_DIR, ensure_dir, resolve_existing_path
except ImportError:
    from common import NLP_CODE_DIR, ensure_dir, resolve_existing_path


DEFAULT_MODEL_PATH = "/scratch/xla2767/hold2/models/qwen3_8b_thinking_grpo_merged_v1"
DEFAULT_SGLANG_URL = "http://localhost:30000"
DEFAULT_AUDITOR_DATASET = "/gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/miner_step/auditor_hf_dataset"
DEFAULT_OUTPUT_DIR = "/gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/auditor_step"
DEFAULT_SPLIT = "validation"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run qbatch auditor-only inference on a miner-produced HF dataset.")
    parser.add_argument("--auditor-dataset", default=DEFAULT_AUDITOR_DATASET)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sglang-url", default=DEFAULT_SGLANG_URL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "validation", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--offline-datasets", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    auditor_dataset = resolve_existing_path(args.auditor_dataset)
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    output_csv = out_dir / "auditor_predictions.csv"
    output_json = out_dir / "auditor_predictions.json"

    cmd = [
        sys.executable,
        str(NLP_CODE_DIR / "qbatch.py"),
        "--mode",
        "auditor",
        "--model-path",
        args.model_path,
        "--sglang-url",
        args.sglang_url,
        "--auditor-dataset",
        str(auditor_dataset),
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
        "--output-csv",
        str(output_csv),
        "--output-json",
        str(output_json),
    ]
    if args.max_batches is not None:
        cmd.extend(["--max-batches", str(args.max_batches)])
    if args.offline_datasets:
        cmd.append("--offline-datasets")

    print("[1/2] launching qbatch auditor-only inference...", flush=True)
    subprocess.run(cmd, check=True)

    summary = {
        "auditor_dataset": str(auditor_dataset),
        "split": args.split,
        "output_csv": str(output_csv),
        "output_json": str(output_json),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[2/2] completed", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
