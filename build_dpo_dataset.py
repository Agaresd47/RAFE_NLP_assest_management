from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from forward_returns_1m import attach_21d_return_and_excess
from task13_dataset_common import (
    SYSTEM_AUDITOR,
    assign_fixed_split,
    bucket_return_label,
    build_prompt_text,
    build_sft_response,
    build_user_prompt,
    coerce_confidence,
    iter_task13_files,
    label_distance,
    make_synthetic_response,
    normalize_label,
    parse_task13_filename,
    summarize_reasoning,
)


DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_PRICES_ROOT = "/gpfs/projects/p32908/data/nlp"
DEFAULT_OUTPUT = "/gpfs/projects/p32908/data/nlp/hf_cot_dpo"
THINK_START = "<think>\n"
THINK_END = "\n</think>\n"
THINK_PATTERN = re.compile(r"^<think>\s*(?P<reasoning>.*?)\s*</think>\s*(?P<tail>.*)$", re.DOTALL)


def _safe_json_loads(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


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


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z]{3,}", str(text).lower()))


def _build_candidate_pool(row: pd.Series) -> list[dict]:
    teacher_full = row["teacher_response_full"]
    teacher_obj = _parse_thinking_response(teacher_full)
    teacher_reasoning = teacher_obj["reasoning_chain"]
    teacher_label = teacher_obj["sentiment_label"]
    teacher_conf = teacher_obj["confidence_score"]
    return_label = normalize_label(row["return_label"])
    factor = row["factor"]
    evidence_count = int(row["evidence_count"])
    historical_count = int(row["historical_count"])

    conservative_label = teacher_label
    if label_distance(teacher_label, return_label) >= 2:
        conservative_label = "neutral"

    concise_reasoning = summarize_reasoning(teacher_reasoning, max_sentences=2) or teacher_reasoning
    concise_conf = round(max(0.35, min(teacher_conf - 0.08, 0.95)), 4)

    conservative_reasoning = (
        f"For {factor}, the prompt provides {evidence_count} current evidence items "
        f"and {historical_count} historical references. Relative to the prior baseline, "
        f"the evidence {('looks broadly balanced versus prior context' if conservative_label == 'neutral' else 'leans positive versus prior context' if conservative_label in {'positive', 'very_positive'} else 'leans negative versus prior context')}. "
        "The conclusion should stay grounded in the filing details rather than unsupported macro assumptions."
    )
    conservative = _format_thinking_response(
        reasoning=conservative_reasoning,
        label=conservative_label,
        confidence=0.55,
    )

    return_aligned_reasoning = (
        f"For {factor}, the prompt provides {evidence_count} current evidence items "
        f"and {historical_count} historical references. Relative to the prior baseline, "
        f"the evidence {('points to a strongly improving setup' if return_label == 'very_positive' else 'leans positive versus prior context' if return_label == 'positive' else 'looks broadly balanced versus prior context' if return_label == 'neutral' else 'leans negative versus prior context' if return_label == 'negative' else 'points to a strongly deteriorating setup')}. "
        "The conclusion should stay grounded in the filing details rather than unsupported macro assumptions."
    )
    return_aligned = _format_thinking_response(
        reasoning=return_aligned_reasoning,
        label=return_label,
        confidence=0.64 if return_label != "neutral" else 0.58,
    )

    teacher_full_text = _format_thinking_response(
        reasoning=teacher_reasoning,
        label=teacher_label,
        confidence=teacher_conf,
    )
    teacher_concise_text = _format_thinking_response(
        reasoning=concise_reasoning,
        label=teacher_label,
        confidence=concise_conf,
    )

    return [
        {"name": "teacher_full", "text": teacher_full_text},
        {"name": "teacher_concise", "text": teacher_concise_text},
        {"name": "return_aligned", "text": return_aligned},
        {"name": "conservative", "text": conservative},
    ]


def _score_candidate(candidate: dict, row: pd.Series) -> dict:
    parsed = _parse_thinking_response(candidate["text"])
    if not parsed["parsed_json"]:
        return {
            "candidate_name": candidate["name"],
            "schema_score": 0.0,
            "teacher_quality": 0.0,
            "grounding": 0.0,
            "return_alignment": 0.0,
            "total": 0.0,
        }

    reasoning = parsed["reasoning_chain"]
    candidate_label = parsed["sentiment_label"]
    candidate_conf = parsed["confidence_score"]

    teacher_obj = _parse_thinking_response(row["teacher_response_full"])
    teacher_reasoning = teacher_obj["reasoning_chain"]
    teacher_label = normalize_label(row["teacher_label"])
    return_label = normalize_label(row["return_label"])

    schema_score = 1.0 if reasoning and candidate_label else 0.0
    label_similarity = 1.0 - label_distance(candidate_label, teacher_label) / 4.0
    teacher_overlap = 0.0
    teacher_tokens = _tokenize(teacher_reasoning)
    reasoning_tokens = _tokenize(reasoning)
    if teacher_tokens and reasoning_tokens:
        teacher_overlap = len(teacher_tokens & reasoning_tokens) / max(1, len(teacher_tokens))
    teacher_quality = 0.45 * schema_score + 0.35 * label_similarity + 0.20 * min(1.0, teacher_overlap * 2.0)

    mentions_context = any(word in reasoning.lower() for word in ["evidence", "historical", "filing", "context"])
    has_numbers = bool(re.search(r"\d", reasoning))
    length_score = 1.0 if 80 <= len(reasoning) <= 1200 else 0.65 if len(reasoning) >= 30 else 0.2
    grounding = 0.45 * schema_score + 0.30 * float(mentions_context or has_numbers) + 0.25 * length_score

    return_alignment = 1.0 - label_distance(candidate_label, return_label) / 4.0
    confidence_bonus = 1.0 - abs(candidate_conf - 0.65)
    total = (
        0.40 * teacher_quality
        + 0.30 * grounding
        + 0.25 * return_alignment
        + 0.05 * confidence_bonus
    )

    return {
        "candidate_name": candidate["name"],
        "schema_score": round(schema_score, 4),
        "teacher_quality": round(teacher_quality, 4),
        "grounding": round(grounding, 4),
        "return_alignment": round(return_alignment, 4),
        "total": round(total, 4),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build return-guided DPO pairs for task1.3.")
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--prices-root", default=DEFAULT_PRICES_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--rationale-style", choices=["full", "concise"], default="concise")
    parser.add_argument("--min-evidence", type=int, default=1)
    parser.add_argument(
        "--score-column",
        choices=["excess_1m", "ret_1m"],
        default="excess_1m",
        help="Return metric used to derive the preference label.",
    )
    parser.add_argument(
        "--teacher-tolerance",
        type=int,
        default=1,
        help="Maximum label-distance for treating the teacher answer as aligned with returns.",
    )
    parser.add_argument(
        "--drop-neutral-returns",
        action="store_true",
        default=True,
        help="Drop rows whose return bucket is neutral so the DPO signal stays directional.",
    )
    parser.add_argument(
        "--keep-neutral-returns",
        action="store_false",
        dest="drop_neutral_returns",
        help="Keep neutral-return rows in the preference dataset.",
    )
    return parser


def collect_base_rows(extract_root: Path, rationale_style: str, min_evidence: int) -> list[dict]:
    rows: list[dict] = []
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
        except json.JSONDecodeError:
            continue

        metadata = payload.get("metadata", {}) or {}
        for audit in payload.get("factor_audits", []) or []:
            evidence_count = len(audit.get("evidence_used", []) or [])
            if evidence_count < min_evidence:
                continue

            teacher_full_obj = json.loads(build_sft_response(audit, "full"))
            teacher_concise_obj = json.loads(build_sft_response(audit, "concise"))
            factor = audit.get("factor", "unknown")
            user_prompt = build_user_prompt(audit, ticker, form, report_date)
            rows.append(
                {
                    "split": split,
                    "ticker": ticker,
                    "form": form,
                    "report_date": report_date,
                    "parsed_report_date": report_date,
                    "year": int(report_date[:4]),
                    "factor": factor,
                    "sector": metadata.get("sector"),
                    "prompt": build_prompt_text(SYSTEM_AUDITOR, user_prompt),
                    "teacher_response": json.dumps(
                        teacher_concise_obj if rationale_style == "concise" else teacher_full_obj,
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "teacher_response_full": _format_thinking_response(
                        reasoning=teacher_full_obj["reasoning_chain"],
                        label=teacher_full_obj["sentiment_label"],
                        confidence=teacher_full_obj["confidence_score"],
                    ),
                    "teacher_response_concise": _format_thinking_response(
                        reasoning=teacher_concise_obj["reasoning_chain"],
                        label=teacher_concise_obj["sentiment_label"],
                        confidence=teacher_concise_obj["confidence_score"],
                    ),
                    "teacher_label": teacher_full_obj["sentiment_label"],
                    "teacher_confidence": teacher_full_obj["confidence_score"],
                    "evidence_count": evidence_count,
                    "historical_count": len(audit.get("historical_context", []) or []),
                    "source_path": str(audit_path),
                }
            )
    return rows


def choose_pair(row: pd.Series, teacher_tolerance: int) -> tuple[str, str, str, str]:
    candidates = _build_candidate_pool(row)
    scored = []
    for candidate in candidates:
        score = _score_candidate(candidate, row)
        scored.append({**candidate, **score})

    scored.sort(key=lambda item: (-item["total"], item["candidate_name"]))
    chosen = scored[0]

    rejected = None
    for candidate in reversed(scored):
        if candidate["candidate_name"] != chosen["candidate_name"]:
            rejected = candidate
            break
    if rejected is None:
        rejected = scored[-1]

    pair_source = f"{chosen['candidate_name']}_over_{rejected['candidate_name']}"
    score_blob = json.dumps(
        [
            {
                "candidate_name": item["candidate_name"],
                "teacher_quality": item["teacher_quality"],
                "grounding": item["grounding"],
                "return_alignment": item["return_alignment"],
                "total": item["total"],
            }
            for item in scored
        ],
        ensure_ascii=False,
    )

    return chosen["text"], rejected["text"], pair_source, score_blob


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    base_rows = collect_base_rows(Path(args.extract_root), args.rationale_style, args.min_evidence)
    df = pd.DataFrame(base_rows)
    if df.empty:
        raise RuntimeError("No eligible task13 rows found for DPO dataset construction.")

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
    candidate_scores = []
    for _, row in df.iterrows():
        chosen, rejected, source, score_blob = choose_pair(row, args.teacher_tolerance)
        chosen_list.append(chosen)
        rejected_list.append(rejected)
        pref_source.append(source)
        candidate_scores.append(score_blob)

    df["chosen"] = chosen_list
    df["rejected"] = rejected_list
    df["preference_source"] = pref_source
    df["candidate_scores"] = candidate_scores

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
        "candidate_scores",
        "source_path",
    ]
    for split, part in df.groupby("split"):
        split_rows[split] = part[keep_cols].to_dict(orient="records")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDict({split: Dataset.from_list(items) for split, items in split_rows.items()})
    dataset.save_to_disk(str(output_dir))
    logging.info(
        "Saved DPO dataset to %s | train=%d validation=%d test=%d",
        output_dir,
        len(split_rows["train"]),
        len(split_rows["validation"]),
        len(split_rows["test"]),
    )


if __name__ == "__main__":
    main()
