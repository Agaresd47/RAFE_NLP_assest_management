from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
NLP_CODE_DIR = SCRIPT_DIR.parent
if str(NLP_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(NLP_CODE_DIR))

from build_dataset import assign_fixed_split, parse_raw_filename
from build_sft_extract_v5 import (
    SYSTEM_MINER,
    build_prompt as build_factor_prompt,
    index_task12_files,
    parse_metadata as parse_task12_metadata,
)
from task13_dataset_common import (
    SYSTEM_AUDITOR,
    build_sft_response,
    build_user_prompt,
    iter_task13_files,
    parse_task13_filename,
)


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
    raise FileNotFoundError(f"Could not find path at any known prefix variant: {path_str}")


def safe_json_load(text: str):
    cleaned = str(text or "").replace("<|im_end|>", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_model_output(text: str) -> dict:
    raw = str(text or "")
    clean = raw.replace("<|im_end|>", "").strip()
    clean = re.sub(r"<think>\s*.*?\s*</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()

    parsed = safe_json_load(clean)
    if parsed is None:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = safe_json_load(clean[start : end + 1])
    if not isinstance(parsed, dict):
        parsed = {}

    extractions = parsed.get("extractions", [])
    if not isinstance(extractions, list):
        extractions = []

    return {
        "raw_text": raw,
        "parsed_json": parsed,
        "metadata": parsed.get("metadata") if isinstance(parsed.get("metadata"), dict) else {},
        "extractions": extractions,
        "parse_success": bool(parsed),
    }


def dedupe_extractions(extractions: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in extractions:
        grouped[(str(item.get("factor", "")), str(item.get("question_key", "")))].append(item)

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


def load_task12_indexes(extract_root: str | Path) -> tuple[dict, dict[str, set[str]]]:
    return index_task12_files(resolve_existing_path(str(extract_root)))


def load_task13_index(extract_root: str | Path) -> dict[tuple[str, str, str], dict]:
    extract_path = resolve_existing_path(str(extract_root))
    index: dict[tuple[str, str, str], dict] = {}
    for audit_path in iter_task13_files(extract_path):
        parsed = parse_task13_filename(audit_path)
        if not parsed:
            continue
        try:
            payload = json.loads(audit_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        factor_map = {}
        for audit in payload.get("factor_audits", []) or []:
            factor = str(audit.get("factor") or "").strip()
            if factor:
                factor_map[factor] = audit
        index[parsed] = {
            "path": str(audit_path),
            "payload": payload,
            "factor_map": factor_map,
            "metadata": payload.get("metadata", {}) or {},
        }
    return index


def load_answered_factor_questions(task12_index: dict, filing_key: tuple[str, str, str]) -> tuple[dict[str, list[str]], dict]:
    task12_path = task12_index.get(filing_key)
    if task12_path is None:
        return {}, {}
    obj = json.loads(Path(task12_path).read_text(encoding="utf-8", errors="ignore"))
    factor_map: dict[str, list[str]] = defaultdict(list)
    for item in obj.get("extractions", []) or []:
        factor = str(item.get("factor") or "").strip()
        qkey = str(item.get("question_key") or "").strip()
        if not factor or not qkey:
            continue
        if qkey not in factor_map[factor]:
            factor_map[factor].append(qkey)
    return dict(sorted(factor_map.items())), obj


def load_answered_factors(task12_index: dict, filing_key: tuple[str, str, str]) -> tuple[list[str], dict]:
    factor_map, obj = load_answered_factor_questions(task12_index, filing_key)
    return sorted(factor_map.keys()), obj


def build_all_factor_questions(factor_to_qkeys: dict[str, set[str]]) -> dict[str, list[str]]:
    return {
        factor: sorted(qkeys)
        for factor, qkeys in sorted(factor_to_qkeys.items())
        if qkeys
    }


def apply_filing_limits(
    filing_specs: list[dict],
    *,
    max_tickers: int,
    max_filings: int,
) -> list[dict]:
    selected = filing_specs
    if max_tickers > 0:
        keep_tickers: set[str] = set()
        limited: list[dict] = []
        for spec in selected:
            ticker = spec["ticker"]
            if ticker in keep_tickers or len(keep_tickers) < max_tickers:
                keep_tickers.add(ticker)
                limited.append(spec)
        selected = limited
    if max_filings > 0:
        selected = selected[:max_filings]
    return selected


def build_factor_row(
    *,
    raw_text: str,
    filing_key: tuple[str, str, str],
    factor: str,
    factor_questions: list[str],
    source_path: str,
    metadata_override: dict | None = None,
) -> dict:
    ticker, form, report_date = filing_key
    prompt = build_factor_prompt(
        raw_text=raw_text,
        ticker=ticker,
        form=form,
        report_date=report_date,
        factor=factor,
        factor_questions=factor_questions,
    )
    metadata = {
        "ticker": ticker,
        "filing": form,
        "year": int(report_date[:4]),
        "sector": "info tech",
        "report_date": report_date,
    }
    if metadata_override:
        metadata.update({k: v for k, v in metadata_override.items() if v is not None})
    return {
        "filing_key": filing_key,
        "ticker": ticker,
        "form": form,
        "report_date": report_date,
        "factor": factor,
        "factor_questions": factor_questions,
        "question_count": len(factor_questions),
        "prompt": prompt,
        "messages": [
            {"role": "system", "content": SYSTEM_MINER},
            {"role": "user", "content": prompt},
        ],
        "metadata": metadata,
        "source_path": source_path,
    }


def build_auditor_dataset_row(
    *,
    filing_key: tuple[str, str, str],
    factor: str,
    evidence_used: list[dict],
    sector: str | None,
    historical_context: list[dict],
    teacher_audit: dict | None,
    source_raw_path: str,
    source_task12_file: str | None,
    source_task13_file: str | None,
    source_miner_output_dir: str,
) -> dict:
    ticker, form, report_date = filing_key
    audit_prompt_obj = {
        "factor": factor,
        "evidence_used": evidence_used,
        "historical_context": historical_context,
    }
    user_prompt = build_user_prompt(audit_prompt_obj, ticker, form, report_date)
    assistant = build_sft_response(teacher_audit, "full") if teacher_audit else ""
    teacher_result = (teacher_audit or {}).get("audit_result", {}) or {}
    parsed_teacher = safe_json_load(assistant) if assistant else None
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_AUDITOR},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant},
        ],
        "task": "task13_auditor_cot_sft_from_miner",
        "ticker": ticker,
        "form": form,
        "report_date": report_date,
        "parsed_report_date": report_date,
        "year": int(report_date[:4]),
        "factor": factor,
        "sector": sector,
        "evidence_count": len(evidence_used),
        "historical_count": len(historical_context or []),
        "teacher_label": teacher_result.get("sentiment_label"),
        "normalized_label": parsed_teacher.get("sentiment_label") if isinstance(parsed_teacher, dict) else None,
        "teacher_confidence": teacher_result.get("confidence_score"),
        "rationale_style": "full" if teacher_audit else None,
        "source_path": source_task13_file or source_raw_path,
        "source_raw_path": source_raw_path,
        "source_task12_file": source_task12_file,
        "source_task13_file": source_task13_file,
        "source_miner_output_dir": source_miner_output_dir,
        "teacher_available": bool(teacher_audit),
        "miner_extractions": evidence_used,
        "miner_extraction_count": len(evidence_used),
    }


def build_auditor_filing_prompt(
    *,
    filing_key: tuple[str, str, str],
    factor_blocks: list[dict],
) -> str:
    ticker, form, report_date = filing_key
    lines = [
        "Task: Auditor (1.3)",
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}",
        "",
        "[Current Evidence By Factor]",
    ]

    if factor_blocks:
        for block in factor_blocks:
            lines.append(f"Factor: {block['factor']}")
            evidence_used = block.get("evidence_used", []) or []
            if evidence_used:
                for item in evidence_used:
                    lines.append(
                        f"  q_key : {item.get('question_key', '—')}\n"
                        f"  quote : {item.get('original_quote', '—')}\n"
                        f"  conf  : {item.get('relevance_confidence', '—')}"
                    )
            else:
                lines.append("  No direct evidence extracted.")
            lines.append("")
    else:
        lines.append("No current evidence extracted.")
        lines.append("")

    has_history = any(block.get("historical_context") for block in factor_blocks)
    if has_history:
        lines.append("[Historical Context By Factor]")
        for block in factor_blocks:
            history = block.get("historical_context", []) or []
            if not history:
                continue
            lines.append(f"Factor: {block['factor']}")
            for item in history:
                lines.append(
                    f"  [{item.get('report_date', '—')} {item.get('filing', '—')}] "
                    f"{item.get('fact', '—')}  (conf: {item.get('relevance_confidence', '—')})"
                )
            lines.append("")

    lines.append("Output a single overall sentiment_label and confidence_score for this filing.")
    return "\n".join(lines)


def build_auditor_filing_dataset_row(
    *,
    filing_key: tuple[str, str, str],
    factor_blocks: list[dict],
    sector: str | None,
    source_raw_path: str,
    source_task12_file: str | None,
    source_task13_file: str | None,
    source_miner_output_dir: str,
) -> dict:
    ticker, form, report_date = filing_key
    user_prompt = build_auditor_filing_prompt(
        filing_key=filing_key,
        factor_blocks=factor_blocks,
    )
    evidence_count = sum(len(block.get("evidence_used", []) or []) for block in factor_blocks)
    historical_count = sum(len(block.get("historical_context", []) or []) for block in factor_blocks)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_AUDITOR},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": ""},
        ],
        "task": "task13_auditor_filing_inference_from_miner",
        "ticker": ticker,
        "form": form,
        "report_date": report_date,
        "parsed_report_date": report_date,
        "year": int(report_date[:4]),
        "factor": "__filing__",
        "sector": sector,
        "evidence_count": evidence_count,
        "historical_count": historical_count,
        "teacher_label": None,
        "normalized_label": None,
        "teacher_confidence": None,
        "rationale_style": None,
        "source_path": source_task13_file or source_raw_path,
        "source_raw_path": source_raw_path,
        "source_task12_file": source_task12_file,
        "source_task13_file": source_task13_file,
        "source_miner_output_dir": source_miner_output_dir,
        "teacher_available": False,
        "factor_count": len(factor_blocks),
        "factor_blocks": factor_blocks,
    }


def list_raw_filing_specs(raw_root: str | Path) -> list[dict]:
    raw_path = resolve_existing_path(str(raw_root))
    specs: list[dict] = []
    for path in sorted(raw_path.rglob("*.md")):
        parsed = parse_raw_filename(path)
        if not parsed:
            continue
        ticker, form, report_date = parsed
        split = assign_fixed_split(int(report_date[:4]))
        if split is None:
            continue
        specs.append(
            {
                "filing_key": (ticker, form, report_date),
                "ticker": ticker,
                "form": form,
                "report_date": report_date,
                "split": split,
                "source_path": str(path),
            }
        )
    return specs


def aggregate_filing_extractions(pred_rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str, str], dict] = {}
    for row in pred_rows:
        key = (row["ticker"], row["form"], row["report_date"])
        bucket = buckets.setdefault(
            key,
            {
                "metadata": dict(row["metadata"]),
                "source_raw_path": row["source_path"],
                "factor_rows": [],
                "extractions": [],
            },
        )
        bucket["factor_rows"].append(row)
        bucket["extractions"].extend(row["pred_extractions"])

    merged: list[dict] = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        merged.append(
            {
                "filing_key": key,
                "metadata": bucket["metadata"],
                "source_raw_path": bucket["source_raw_path"],
                "factor_rows": bucket["factor_rows"],
                "extractions": dedupe_extractions(bucket["extractions"]),
            }
        )
    return merged


def metadata_from_task12_obj(task12_obj: dict | None, filing_key: tuple[str, str, str]) -> dict:
    ticker, form, report_date = filing_key
    if task12_obj:
        return parse_task12_metadata(task12_obj, ticker, form, report_date)
    return {
        "ticker": ticker,
        "filing": form,
        "year": int(report_date[:4]),
        "sector": "info tech",
        "report_date": report_date,
    }
