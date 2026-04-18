from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable


SYSTEM_AUDITOR = (
    "You are a Senior Equity Strategist. "
    "Audit the current evidence by comparing it against the historical context. "
    "Analyze the deviation from the baseline and sector context. "
    "Score the sentiment as one of: very_negative, negative, neutral, positive, very_positive. "
    "Provide a reasoning_chain, sentiment_label, and confidence_score. "
    "Output only valid JSON."
)

TASK13_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9._-]+)_(?P<date>\d{2}-\d{2}-\d{4})_(?P<form>10-K|10-Q)_TASK13_AUDIT\.json$",
    re.IGNORECASE,
)

LABEL_ORDER = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
]

LABEL_ALIASES = {
    "very bad": "very_negative",
    "very negative": "very_negative",
    "bad": "negative",
    "neutral": "neutral",
    "good": "positive",
    "very good": "very_positive",
    "very positive": "very_positive",
    "very_negative": "very_negative",
    "negative": "negative",
    "positive": "positive",
    "very_positive": "very_positive",
    "bearish": "negative",
    "very_bearish": "very_negative",
    "bullish": "positive",
    "very_bullish": "very_positive",
}


def parse_date_to_iso(date_str: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_task13_filename(path: Path) -> tuple[str, str, str] | None:
    match = TASK13_RE.match(path.name)
    if not match:
        return None
    report_date = parse_date_to_iso(match.group("date"))
    if not report_date:
        return None
    return match.group("ticker").upper(), match.group("form").upper(), report_date


def assign_fixed_split(year: int) -> str | None:
    if 2015 <= year <= 2022:
        return "train"
    if year == 2023:
        return "validation"
    if 2024 <= year <= 2026:
        return "test"
    return None


def normalize_label(label: str | None) -> str:
    if not label:
        return "neutral"
    key = str(label).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    return LABEL_ALIASES.get(key, "neutral")


def label_distance(left: str, right: str) -> int:
    return abs(LABEL_ORDER.index(normalize_label(left)) - LABEL_ORDER.index(normalize_label(right)))


def bucket_return_label(value: float | None) -> str | None:
    if value is None:
        return None
    if value <= -0.08:
        return "very_negative"
    if value <= -0.02:
        return "negative"
    if value < 0.02:
        return "neutral"
    if value < 0.08:
        return "positive"
    return "very_positive"


def coerce_confidence(value) -> float:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        conf = 0.6
    return max(0.0, min(conf, 1.0))


def summarize_reasoning(reasoning: str, max_sentences: int = 2) -> str:
    text = " ".join(str(reasoning or "").split())
    if not text:
        return ""
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(pieces[:max_sentences]).strip()


def build_user_prompt(audit: dict, ticker: str, form: str, report_date: str) -> str:
    factor = audit.get("factor", "unknown")
    lines = [
        "Task: Auditor (1.3)",
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}",
        f"Factor: {factor}",
        "",
        "[Current Evidence]",
    ]

    evidence = audit.get("evidence_used", []) or []
    if evidence:
        for item in evidence:
            lines.append(
                f"  q_key : {item.get('question_key', '—')}\n"
                f"  quote : {item.get('original_quote', '—')}\n"
                f"  conf  : {item.get('relevance_confidence', '—')}"
            )
    else:
        lines.append("  No direct evidence extracted.")

    history = audit.get("historical_context", []) or []
    if history:
        lines.append("[Historical Context]")
        for item in history:
            lines.append(
                f"  [{item.get('report_date', '—')} {item.get('filing', '—')}] "
                f"{item.get('fact', '—')}  (conf: {item.get('relevance_confidence', '—')})"
            )

    lines.append("\nOutput reasoning_chain, sentiment_label, and confidence_score for this factor.")
    return "\n".join(lines)


def build_sft_response(audit: dict, rationale_style: str) -> str:
    result = audit.get("audit_result", {}) or {}
    payload = {
        "reasoning_chain": summarize_reasoning(result.get("reasoning_chain", ""), 2)
        if rationale_style == "concise"
        else str(result.get("reasoning_chain", "")).strip(),
        "sentiment_label": normalize_label(result.get("sentiment_label")),
        "confidence_score": round(coerce_confidence(result.get("confidence_score")), 4),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_prompt_text(system_prompt: str, user_prompt: str) -> str:
    return f"System: {system_prompt}\n\nUser:\n{user_prompt}\n\nAssistant:\n"


def make_synthetic_response(
    *,
    factor: str,
    label: str,
    evidence_count: int,
    history_count: int,
    confidence: float,
) -> str:
    label = normalize_label(label)
    intensity = {
        "very_negative": "points to a strongly deteriorating setup",
        "negative": "leans negative versus prior context",
        "neutral": "looks broadly balanced versus prior context",
        "positive": "leans positive versus prior context",
        "very_positive": "points to a strongly improving setup",
    }[label]
    reasoning = (
        f"For {factor}, the prompt provides {evidence_count} current evidence items "
        f"and {history_count} historical references. Relative to the prior baseline, "
        f"the evidence {intensity}. The conclusion should stay grounded in the filing "
        f"details rather than unsupported macro assumptions."
    )
    payload = {
        "reasoning_chain": reasoning,
        "sentiment_label": label,
        "confidence_score": round(coerce_confidence(confidence), 4),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def iter_task13_files(extract_root: Path) -> Iterable[Path]:
    yield from sorted(extract_root.rglob("*_TASK13_AUDIT.json"))
