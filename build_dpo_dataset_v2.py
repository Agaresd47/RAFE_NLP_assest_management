from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from build_dpo_dataset import (
    THINK_END,
    THINK_PATTERN,
    THINK_START,
    _safe_json_loads,
    collect_base_rows,
)
from forward_returns_1m import attach_21d_return_and_excess
from task13_dataset_common import (
    assign_fixed_split,
    bucket_return_label,
    coerce_confidence,
    label_distance,
    normalize_label,
    summarize_reasoning,
)


DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_PRICES_ROOT = "/gpfs/projects/p32908/data/nlp"
DEFAULT_OUTPUT = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo_v2"


def _format_thinking_response(reasoning: str, label: str, confidence: float) -> str:
    payload = {
        "sentiment_label": normalize_label(label),
        "confidence_score": round(coerce_confidence(confidence), 4),
    }
    return THINK_START + str(reasoning).strip() + THINK_END + json.dumps(payload, ensure_ascii=False)


def _parse_thinking_response(text: str) -> dict:
    raw = str(text or "").strip()
    reasoning = ""
    tail = raw

    match = THINK_PATTERN.match(raw)
    if match:
        reasoning = str(match.group("reasoning") or "").strip()
        tail = str(match.group("tail") or "").strip()

    obj = _safe_json_loads(tail)
    if obj is None:
        obj = _safe_json_loads(raw) or {}
        if not reasoning:
            reasoning = str(obj.get("reasoning_chain", "")).strip()

    if not reasoning:
        reasoning = str(obj.get("reasoning_chain", "")).strip()

    return {
        "reasoning_chain": reasoning,
        "sentiment_label": normalize_label(obj.get("sentiment_label")),
        "confidence_score": coerce_confidence(obj.get("confidence_score")),
        "parsed_json": bool(obj),
    }


def _trim_sentences(reasoning: str, max_sentences: int) -> str:
    text = " ".join(str(reasoning or "").split())
    if not text:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(pieces[:max_sentences]).strip()


def _make_verbose_rambling(reasoning: str, factor: str, teacher_label: str) -> str:
    base = " ".join(str(reasoning or "").split())
    if not base:
        base = (
            f"For {factor}, the filing evidence should be interpreted carefully and in context. "
            f"The net directional signal still appears {teacher_label}, but the answer should discuss "
            "current evidence, historical references, caveats, and supporting context before concluding."
        )

    filler = (
        " The discussion should also restate that the answer is grounded in filing evidence, "
        "historical context, and factor-specific interpretation rather than unsupported macro claims. "
        " The evidence should be weighed together with prior context, possible offsets, and cross-checks "
        "before a final conclusion is presented."
    )
    verbose = base + filler + filler
    return verbose


def _build_candidates_v2(row: pd.Series) -> dict[str, str]:
    teacher = _parse_thinking_response(row["teacher_response_full"])
    teacher_reasoning = teacher["reasoning_chain"]
    teacher_label = teacher["sentiment_label"]
    teacher_conf = teacher["confidence_score"]
    return_label = normalize_label(row["return_label"])
    factor = row["factor"]
    evidence_count = int(row["evidence_count"])
    historical_count = int(row["historical_count"])

    concise_reasoning = summarize_reasoning(teacher_reasoning, max_sentences=2) or teacher_reasoning
    balanced_reasoning = _trim_sentences(teacher_reasoning, max_sentences=4) or teacher_reasoning
    verbose_reasoning = _make_verbose_rambling(teacher_reasoning, factor, teacher_label)

    conservative_label = teacher_label if label_distance(teacher_label, return_label) < 2 else "neutral"
    conservative_reasoning = (
        f"For {factor}, the prompt includes {evidence_count} current evidence items and "
        f"{historical_count} historical references. The filing evidence should be read conservatively, "
        f"so the net signal is treated as {conservative_label} unless the documents clearly support a stronger move."
    )

    return_aligned_reasoning = (
        f"For {factor}, the prompt includes {evidence_count} current evidence items and "
        f"{historical_count} historical references. Relative to prior context, the filing evidence "
        f"leans {return_label}. The conclusion should stay grounded in cited filing details and close with a clear final answer."
    )

    return {
        "teacher_balanced": _format_thinking_response(
            reasoning=balanced_reasoning,
            label=teacher_label,
            confidence=min(0.9, teacher_conf),
        ),
        "teacher_concise": _format_thinking_response(
            reasoning=concise_reasoning,
            label=teacher_label,
            confidence=max(0.35, min(0.85, teacher_conf - 0.06)),
        ),
        "verbose_rambling": _format_thinking_response(
            reasoning=verbose_reasoning,
            label=teacher_label,
            confidence=min(0.92, teacher_conf + 0.02),
        ),
        "conservative": _format_thinking_response(
            reasoning=conservative_reasoning,
            label=conservative_label,
            confidence=0.55,
        ),
        "return_aligned_balanced": _format_thinking_response(
            reasoning=return_aligned_reasoning,
            label=return_label,
            confidence=0.62 if return_label != "neutral" else 0.56,
        ),
    }


def choose_pair_v2(row: pd.Series, teacher_tolerance: int) -> tuple[str, str, str]:
    candidates = _build_candidates_v2(row)
    teacher_label = normalize_label(row["teacher_label"])
    return_label = normalize_label(row["return_label"])
    aligned = label_distance(teacher_label, return_label) <= teacher_tolerance

    if aligned:
        chosen = candidates["teacher_balanced"]
        rejected = candidates["verbose_rambling"]
        source = "teacher_balanced_over_verbose_rambling"
    else:
        chosen = candidates["return_aligned_balanced"]
        rejected = candidates["verbose_rambling"]
        source = "return_aligned_balanced_over_verbose_rambling"

    # Keep some anti-overcaution signal even when verbose preference is the main one.
    if label_distance(teacher_label, return_label) >= 2 and row.get("evidence_count", 0) >= 2:
        rejected = candidates["conservative"]
        source = "return_aligned_balanced_over_conservative"

    return chosen, rejected, source


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DPO v2 pairs with anti-rambling preference.")
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--prices-root", default=DEFAULT_PRICES_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rationale-style", choices=["full", "concise"], default="concise")
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument("--score-column", choices=["excess_1m", "ret_1m"], default="excess_1m")
    parser.add_argument("--teacher-tolerance", type=int, default=1)
    parser.add_argument("--drop-neutral-returns", action="store_true", default=True)
    parser.add_argument("--keep-neutral-returns", action="store_false", dest="drop_neutral_returns")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    base_rows = collect_base_rows(Path(args.extract_root), args.rationale_style, args.min_evidence)
    df = pd.DataFrame(base_rows)
    if df.empty:
        raise RuntimeError("No eligible task13 rows found for DPO v2 dataset construction.")

    df = attach_21d_return_and_excess(df, raw_dir=Path(args.prices_root), date_col="parsed_report_date")
    score_series = df[args.score_column].fillna(df["ret_1m"])
    df["return_label"] = score_series.apply(bucket_return_label)
    df = df[df["return_label"].notna()].copy()
    if args.drop_neutral_returns:
        df = df[df["return_label"] != "neutral"].copy()

    if df.empty:
        raise RuntimeError("No rows retained after attaching returns; check date alignment and price CSVs.")

    chosen_list = []
    rejected_list = []
    pref_source = []
    for _, row in df.iterrows():
        chosen, rejected, source = choose_pair_v2(row, args.teacher_tolerance)
        chosen_list.append(chosen)
        rejected_list.append(rejected)
        pref_source.append(source)

    df["chosen"] = chosen_list
    df["rejected"] = rejected_list
    df["preference_source"] = pref_source

    split_rows = {"train": [], "validation": [], "test": []}
    keep_cols = [
        "prompt",
        "chosen",
        "rejected",
        "ticker",
        "form",
        "report_date",
        "parsed_report_date",
        "year",
        "factor",
        "sector",
        "teacher_label",
        "return_label",
        "teacher_confidence",
        "ret_1m",
        "excess_1m",
        "evidence_count",
        "historical_count",
        "preference_source",
        "source_path",
    ]
    for split, part in df.groupby("split"):
        split_rows[split] = part[keep_cols].to_dict(orient="records")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDict({split: Dataset.from_list(items) for split, items in split_rows.items()})
    dataset.save_to_disk(str(output_dir))
    logging.info(
        "Saved DPO v2 dataset to %s | train=%d validation=%d test=%d",
        output_dir,
        len(split_rows["train"]),
        len(split_rows["validation"]),
        len(split_rows["test"]),
    )
    if split_rows["train"]:
        sample = split_rows["train"][0]
        logging.info("Sample preference source: %s", sample["preference_source"])


if __name__ == "__main__":
    main()
