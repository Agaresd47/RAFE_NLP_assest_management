from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


AUDIT_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9._-]+)_(?P<date>\d{2}-\d{2}-\d{4})_(?P<form>10-K|10-Q)_TASK13_AUDIT\.json$",
    re.IGNORECASE,
)

LABEL_ORDER = ["very_negative", "negative", "neutral", "positive", "very_positive"]
LABEL_ALIASES = {
    "very_bad": "very_negative",
    "very_negative": "very_negative",
    "bad": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "good": "positive",
    "positive": "positive",
    "very_good": "very_positive",
    "very_positive": "very_positive",
}
OPPOSITE_LABEL = {
    "very_negative": "very_positive",
    "negative": "positive",
    "neutral": "neutral",
    "positive": "negative",
    "very_positive": "very_negative",
}

SFT_SYSTEM_PROMPT = (
    "You are a senior equity research assistant. "
    "Use the filing evidence and historical context to infer the stock-relevant signal. "
    "Return only valid JSON with keys reasoning_chain, sentiment_label, and confidence_score. "
    "Use sentiment_label from: very_negative, negative, neutral, positive, very_positive."
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_date_to_iso(date_str: str | None) -> str | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_audit_filename(path: Path):
    m = AUDIT_RE.match(path.name)
    if not m:
        return None
    iso = parse_date_to_iso(m.group("date"))
    if not iso:
        return None
    return m.group("ticker").upper(), m.group("form").upper(), iso


def assign_fixed_split(year: int) -> str | None:
    if 2015 <= year <= 2022:
        return "train"
    if year == 2023:
        return "validation"
    if 2024 <= year <= 2026:
        return "test"
    return None


def stable_id(*parts: object) -> str:
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def normalize_sentiment_label(label: str | None) -> str | None:
    if label is None:
        return None
    key = str(label).strip().lower().replace(" ", "_").replace("-", "_")
    return LABEL_ALIASES.get(key, key if key in LABEL_ORDER else None)


def confidence_for_label(label: str | None) -> float:
    label = normalize_sentiment_label(label)
    mapping = {
        "very_negative": 0.90,
        "negative": 0.75,
        "neutral": 0.60,
        "positive": 0.75,
        "very_positive": 0.90,
    }
    return float(mapping.get(label, 0.60))


def format_current_evidence(evidence_used: list[dict]) -> str:
    if not evidence_used:
        return "- None"
    lines: list[str] = []
    for i, e in enumerate(evidence_used, start=1):
        q_key = e.get("question_key") or e.get("question_text") or "unknown"
        quote = e.get("original_quote") or "—"
        conf = e.get("relevance_confidence")
        lines.append(f"{i}. [{q_key}] {quote} (confidence: {conf})")
    return "\n".join(lines)


def format_historical_context(historical_context: list[dict]) -> str:
    if not historical_context:
        return "- None"
    lines: list[str] = []
    for i, h in enumerate(historical_context, start=1):
        report_date = h.get("report_date") or "—"
        filing = h.get("filing") or "—"
        fact = h.get("fact") or "—"
        conf = h.get("relevance_confidence")
        lines.append(f"{i}. [{report_date} | {filing}] {fact} (confidence: {conf})")
    return "\n".join(lines)


def build_auditor_prompt(row: dict) -> str:
    lines = [
        f"Task: Auditor (1.3)",
        f"Ticker: {row['ticker']} | Filing: {row['form']} | Date: {row['report_date']}",
        f"Factor: {row['factor']}",
        "",
        "[Current Evidence]",
        format_current_evidence(row.get("evidence_used", [])),
        "",
        "[Historical Context]",
        format_historical_context(row.get("historical_context", [])),
        "",
        "Return only valid JSON with keys reasoning_chain, sentiment_label, and confidence_score.",
        "Use sentiment_label from: very_negative, negative, neutral, positive, very_positive.",
    ]
    return "\n".join(lines)


def build_sft_assistant_payload(row: dict) -> str:
    payload = {
        "reasoning_chain": str(row.get("reasoning_chain", "")).strip(),
        "sentiment_label": normalize_sentiment_label(row.get("sentiment_label")),
        "confidence_score": float(row.get("confidence_score") or 0.0),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_return_guided_assistant_payload(label: str, confidence: float, *, rationale: str) -> str:
    payload = {
        "reasoning_chain": rationale.strip(),
        "sentiment_label": normalize_sentiment_label(label),
        "confidence_score": float(confidence),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_chat_messages(row: dict, assistant_text: str, *, system_prompt: str = SFT_SYSTEM_PROMPT) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": build_auditor_prompt(row)},
        {"role": "assistant", "content": assistant_text},
    ]


def flatten_audit_rows(extract_root: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(extract_root.rglob("*_TASK13_AUDIT.json")):
        obj = load_json(path)
        metadata = obj.get("metadata", {})
        parsed = parse_audit_filename(path)
        if parsed is None:
            parsed = (
                str(metadata.get("ticker", "")).upper(),
                str(metadata.get("filing", "")).upper(),
                parse_date_to_iso(metadata.get("report_date")),
            )
        ticker, form, report_date = parsed
        if not ticker or not form or not report_date:
            continue

        year = int(metadata.get("year") or report_date[:4])
        for idx, audit in enumerate(obj.get("factor_audits", [])):
            result = audit.get("audit_result", {})
            row = {
                "row_id": stable_id(path.as_posix(), idx, audit.get("factor"), report_date),
                "source_audit_file": path.as_posix(),
                "source_task12_file": obj.get("source_task12_file"),
                "ticker": ticker,
                "form": form,
                "report_date": report_date,
                "year": year,
                "sector": metadata.get("sector"),
                "factor": audit.get("factor", "unknown"),
                "evidence_used": audit.get("evidence_used", []),
                "historical_context": audit.get("historical_context", []),
                "reasoning_chain": str(result.get("reasoning_chain", "")).strip(),
                "sentiment_label": normalize_sentiment_label(result.get("sentiment_label")),
                "confidence_score": float(result.get("confidence_score", 0.0)),
                "num_factors": obj.get("num_factors"),
            }
            rows.append(row)
    return rows


def compute_return_thresholds(values: Iterable[float], *, fallback: tuple[float, float, float, float] = (-0.08, -0.02, 0.02, 0.08)) -> tuple[float, float, float, float]:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size < 5:
        return fallback
    qs = np.nanquantile(arr, [0.2, 0.4, 0.6, 0.8]).astype(float)
    if not np.all(np.isfinite(qs)):
        return fallback
    if any(qs[i] >= qs[i + 1] for i in range(len(qs) - 1)):
        return fallback
    return tuple(float(x) for x in qs)


def bucket_excess_return(excess: float, thresholds: tuple[float, float, float, float]) -> str | None:
    if excess is None or not np.isfinite(excess):
        return None
    t1, t2, t3, t4 = thresholds
    if excess <= t1:
        return "very_negative"
    if excess <= t2:
        return "negative"
    if excess <= t3:
        return "neutral"
    if excess <= t4:
        return "positive"
    return "very_positive"


def build_return_guided_rationale(label: str, *, ticker: str, form: str, report_date: str) -> str:
    pretty = label.replace("_", " ")
    return (
        f"The filing signal for {ticker} {form} on {report_date} is best treated as {pretty}. "
        f"This preference is anchored to the realized 21-day return outcome rather than a purely textual judgment."
    )
