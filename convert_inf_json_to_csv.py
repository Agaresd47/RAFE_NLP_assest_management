import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ROOT = "/projects/p32908"
DEFAULT_OUTPUT_DIR = "/projects/p32908/nlp_result"
DEFAULT_INPUTS = [
    "sft_inf.json",
    "dpo_inf.json",
    "sft_inf_ttc.json",
    "dpo_inf_ttc.json",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert regular/TTC inference JSON files to CSV.")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUTS)
    return parser


def infer_mode(name: str) -> str:
    return "ttc" if "_ttc" in name else "regular"


def infer_model(name: str) -> str:
    if name.startswith("sft_"):
        return "sft"
    if name.startswith("dpo_"):
        return "dpo"
    return "unknown"


def normalize_regular_row(row: dict) -> dict:
    return {
        "Date": row.get("Date"),
        "Symbol": row.get("Symbol"),
        "Form": row.get("Form"),
        "Factor": row.get("Factor"),
        "Signal": row.get("Signal"),
        "Confidence": row.get("Confidence"),
        "Output": row.get("Output"),
    }


def normalize_ttc_row(row: dict) -> dict:
    votes = row.get("Votes") or {}
    if not isinstance(votes, dict):
        votes = {}
    return {
        "Date": row.get("Date"),
        "Symbol": row.get("Symbol"),
        "Form": row.get("Form"),
        "Factor": row.get("Factor"),
        "Signal": row.get("Signal"),
        "Confidence": row.get("Confidence"),
        "Output": row.get("Output"),
        "Votes": json.dumps(votes, ensure_ascii=False, sort_keys=True),
    }


def convert_one(path: Path, output_dir: Path) -> Path:
    rows = json.loads(path.read_text())
    mode = infer_mode(path.name)
    model = infer_model(path.name)

    if mode == "ttc":
        out_rows = [normalize_ttc_row(row) for row in rows]
    else:
        out_rows = [normalize_regular_row(row) for row in rows]

    df = pd.DataFrame(out_rows)
    if not df.empty:
        df = df.sort_values(["Symbol", "Date", "Factor"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / path.with_suffix(".csv").name
    df.to_csv(out_path, index=False)
    print(f"[done] {path.name} -> {out_path.name} rows={len(df)} mode={mode} model={model}")
    return out_path


def main():
    args = build_arg_parser().parse_args()
    root = Path(args.root)
    output_dir = Path(args.output_dir)
    for name in args.inputs:
        path = root / name
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        convert_one(path, output_dir)


if __name__ == "__main__":
    main()
