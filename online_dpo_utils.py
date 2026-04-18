from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from task13_dataset_common import LABEL_ORDER, coerce_confidence, label_distance, normalize_label


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
THINK_RE = re.compile(r"^\s*<think>\s*(?P<reasoning>.*?)\s*</think>\s*(?P<tail>.*)$", re.DOTALL)


@dataclass
class CandidateScore:
    candidate_name: str
    prm_score: float
    rm_score: float
    return_score: float
    confidence_score: float
    format_score: float
    total: float
    parsed_json: bool
    sentiment_label: str | None


def extract_prompt_text(example: dict[str, Any], tokenizer=None) -> str:
    prompt = example.get("prompt")
    if prompt:
        return str(prompt).strip()

    messages = example.get("messages")
    if messages:
        msgs = list(messages)
        if msgs and msgs[-1].get("role") == "assistant":
            msgs = msgs[:-1]
        if tokenizer is None:
            raise ValueError("Tokenizer is required to convert messages into prompt text.")
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        ).strip()

    text = example.get("text")
    if text:
        return str(text).strip()

    raise ValueError("Example has no 'prompt', 'messages', or 'text' field.")


def parse_completion_json(text: str) -> dict[str, Any] | None:
    clean = str(text).strip().replace("<tool_call>", "").replace("</tool_call>", "")
    reasoning_from_think = ""
    think_match = THINK_RE.match(clean)
    if think_match:
        reasoning_from_think = str(think_match.group("reasoning") or "").strip()
        clean = str(think_match.group("tail") or "").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        match = JSON_RE.search(clean)
        if not match:
            match = JSON_RE.search(str(text).strip())
            if not match:
                return None
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    if reasoning_from_think and not parsed.get("reasoning_chain"):
        parsed["reasoning_chain"] = reasoning_from_think
    return parsed


def response_format_score(parsed: dict[str, Any] | None) -> float:
    if not parsed:
        return 0.0
    reasoning = str(parsed.get("reasoning_chain", "")).strip()
    sentiment = normalize_label(parsed.get("sentiment_label"))
    confidence = coerce_confidence(parsed.get("confidence_score"))
    if not reasoning or not sentiment:
        return 0.35
    if confidence <= 0.0:
        return 0.45
    return 1.0


def confidence_value(parsed: dict[str, Any] | None) -> float:
    if not parsed:
        return 0.0
    return coerce_confidence(parsed.get("confidence_score"))


def return_alignment_score(candidate_label: str | None, target_label: str | None) -> float:
    candidate_label = normalize_label(candidate_label)
    target_label = normalize_label(target_label)
    if not candidate_label or not target_label:
        return 0.5
    max_distance = max(1, len(LABEL_ORDER) - 1)
    return 1.0 - label_distance(candidate_label, target_label) / max_distance


def heuristic_prm_score(prompt_text: str, completion_text: str, parsed: dict[str, Any] | None) -> float:
    if not parsed:
        return 0.0
    reasoning = str(parsed.get("reasoning_chain", "")).strip()
    sentiment = normalize_label(parsed.get("sentiment_label"))
    confidence = coerce_confidence(parsed.get("confidence_score"))

    prompt_lower = prompt_text.lower()
    reasoning_lower = reasoning.lower()

    evidence_hits = sum(
        1 for token in ("evidence", "historical", "filing", "context", "quote") if token in reasoning_lower
    )
    prompt_overlap = sum(
        1 for token in ("ticker", "factor", "historical context", "current evidence") if token in prompt_lower
    )
    length_score = 1.0 if 80 <= len(reasoning) <= 1200 else 0.7 if len(reasoning) >= 30 else 0.2
    label_score = 0.4 if sentiment else 0.0
    confidence_score = confidence
    format_score = response_format_score(parsed)

    return float(
        0.30 * format_score
        + 0.20 * min(1.0, evidence_hits / 3.0)
        + 0.15 * min(1.0, prompt_overlap / 3.0)
        + 0.20 * length_score
        + 0.15 * label_score
        + 0.15 * confidence_score
    )


def heuristic_rm_score(candidate_label: str | None, target_label: str | None, confidence: float) -> float:
    alignment = return_alignment_score(candidate_label, target_label)
    return float(0.8 * alignment + 0.2 * confidence)


def score_candidate(
    *,
    candidate_name: str,
    prompt_text: str,
    completion_text: str,
    target_label: str | None,
    prm_score: float | None = None,
    rm_score: float | None = None,
) -> CandidateScore:
    parsed = parse_completion_json(completion_text)
    candidate_label = normalize_label(parsed.get("sentiment_label") if parsed else None)
    confidence = confidence_value(parsed)
    format_score = response_format_score(parsed)

    prm_score = float(prm_score) if prm_score is not None else heuristic_prm_score(prompt_text, completion_text, parsed)
    rm_score = float(rm_score) if rm_score is not None else heuristic_rm_score(candidate_label, target_label, confidence)
    return_score = return_alignment_score(candidate_label, target_label)

    total = (
        0.45 * prm_score
        + 0.35 * rm_score
        + 0.15 * confidence
        + 0.05 * format_score
    )

    return CandidateScore(
        candidate_name=candidate_name,
        prm_score=round(prm_score, 4),
        rm_score=round(rm_score, 4),
        return_score=round(return_score, 4),
        confidence_score=round(confidence, 4),
        format_score=round(format_score, 4),
        total=round(total, 4),
        parsed_json=parsed is not None,
        sentiment_label=candidate_label,
    )


def select_pair(scored_candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not scored_candidates:
        raise ValueError("No candidates to select from.")
    ranked = sorted(scored_candidates, key=lambda item: (-item["total"], item["candidate_name"]))
    chosen = ranked[0]
    rejected = ranked[-1]
    if chosen["candidate_name"] == rejected["candidate_name"] and len(ranked) > 1:
        rejected = ranked[1]
    return chosen, rejected


def safe_logit_score(outputs: torch.Tensor) -> float:
    flat = outputs.detach().float().flatten()
    if flat.numel() == 0:
        return 0.0
    if flat.numel() == 1:
        return float(torch.sigmoid(flat[0]).item())
    return float(torch.sigmoid(flat[-1] - flat[0]).item())


def batch_logit_scores(logits: torch.Tensor) -> list[float]:
    logits = logits.detach().float()
    if logits.ndim == 0:
        return [float(torch.sigmoid(logits).item())]
    if logits.ndim == 1:
        if logits.numel() == 1:
            return [float(torch.sigmoid(logits[0]).item())]
        if logits.numel() == 2:
            return [float(torch.sigmoid(logits[1] - logits[0]).item())]
        return [float(torch.sigmoid(x).item()) for x in logits]
    scores: list[float] = []
    for row in logits:
        scores.append(safe_logit_score(row))
    return scores


def normalize_scalar_score(raw_score: float) -> float:
    if math.isnan(raw_score) or math.isinf(raw_score):
        return 0.0
    return max(0.0, min(float(raw_score), 1.0))
