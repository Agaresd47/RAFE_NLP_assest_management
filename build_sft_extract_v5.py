from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict

from build_dataset import assign_fixed_split, parse_extract_filename, parse_raw_filename


DEFAULT_RAW_ROOT = "/gpfs/projects/p32908/data/nlp/MDA_Raw"
DEFAULT_EXTRACT_ROOT = "/gpfs/projects/p32908/data/nlp/Extract"
DEFAULT_OUTPUT = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5"

RAW_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9._-]+)_(?P<form>10-K|10-Q)_(?P<date>\d{4}-\d{2}-\d{2})\.md$",
    re.IGNORECASE,
)
SYSTEM_MINER = (
    "You are a Financial Data Engineer. "
    "Extract original quotes from the provided source text that answer the specific questions in the target factor. "
    "For each answer you find, return the original_quote and a relevance_confidence from 0 to 1. "
    "If the target factor questions are not answered by the filing text, return an empty extractions list. "
    "Output only valid JSON."
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build factor-level extract SFT v5 dataset.")
    parser.add_argument("--raw-root", default=DEFAULT_RAW_ROOT)
    parser.add_argument("--extract-root", default=DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-approx-tokens", type=int, default=259935)
    parser.add_argument("--max-quotes-per-question", type=int, default=3)
    parser.add_argument("--max-total-quotes", type=int, default=6)
    parser.add_argument("--max-train-rows-per-factor", type=int, default=80)
    parser.add_argument("--negative_ratio", type=float, default=0.2)
    return parser


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def stable_bucket(*parts: str) -> float:
    raw = "||".join(parts)
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def question_number(question_key: str) -> int:
    match = re.match(r"Q(\d+)::", str(question_key or ""))
    return int(match.group(1)) if match else 9999


def question_text(question_key: str) -> str:
    key = str(question_key or "")
    return key.split("::", 1)[1].strip() if "::" in key else key


def parse_metadata(t12_obj: dict, ticker: str, form: str, report_date: str) -> dict:
    metadata = t12_obj.get("metadata") if isinstance(t12_obj.get("metadata"), dict) else {}
    return {
        "ticker": metadata.get("ticker", ticker),
        "filing": metadata.get("filing", form),
        "year": int(str(metadata.get("year", report_date[:4]))),
        "sector": metadata.get("sector", "info tech"),
        "report_date": metadata.get("report_date", report_date),
    }


def index_task12_files(extract_root: Path) -> tuple[dict, dict[str, set[str]]]:
    task12_index = {}
    factor_to_qkeys: dict[str, set[str]] = defaultdict(set)

    for path in extract_root.rglob("*.json"):
        parsed = parse_extract_filename(path)
        if not parsed:
            continue
        key, kind = parsed[:3], parsed[3]
        if kind != "TASK12_EXTRACTIONS":
            continue
        task12_index[key] = path
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        for extraction in obj.get("extractions", []):
            factor = extraction.get("factor")
            qkey = extraction.get("question_key")
            if factor and qkey:
                factor_to_qkeys[str(factor)].add(str(qkey))
    return task12_index, factor_to_qkeys


def dedupe_quotes(items: list[dict], max_quotes: int) -> list[dict]:
    seen: set[str] = set()
    ordered = sorted(
        items,
        key=lambda x: float(x.get("relevance_confidence", 0.0)),
        reverse=True,
    )
    output: list[dict] = []
    for item in ordered:
        quote = normalize_whitespace(item.get("original_quote", ""))
        if not quote or quote in seen:
            continue
        seen.add(quote)
        output.append(
            {
                "factor": item.get("factor"),
                "question_key": item.get("question_key"),
                "original_quote": item.get("original_quote"),
                "relevance_confidence": round(float(item.get("relevance_confidence", 0.0)), 4),
            }
        )
        if len(output) >= max_quotes:
            break
    return output


def build_prompt(
    *,
    raw_text: str,
    ticker: str,
    form: str,
    report_date: str,
    factor: str,
    factor_questions: list[str],
) -> str:
    question_lines = [
        f"- {qkey}\n  {question_text(qkey)}"
        for qkey in sorted(factor_questions, key=question_number)
    ]
    return (
        f"Task: Miner (1.2)\n"
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}\n\n"
        "Text:\n"
        f"{raw_text}\n\n"
        "Target Factor Block:\n"
        f"Factor: {factor}\n"
        "Candidate Questions For This Factor:\n"
        f"{chr(10).join(question_lines)}\n\n"
        "Instructions:\n"
        "- Return only valid JSON.\n"
        "- Output only the metadata and extractions schema shown by prior examples.\n"
        "- Extract only original quotes from the filing text.\n"
        "- Multiple questions in this factor may have answers.\n"
        "- If multiple relevant quotes exist, include all distinct supporting quotes you find.\n"
        "- If none of the target factor questions are answered, return an empty extractions list.\n"
        "- Do not answer questions outside this factor.\n"
        "- Do not summarize; copy exact supporting quotes.\n"
    )


def build_row(
    *,
    raw_text: str,
    ticker: str,
    form: str,
    report_date: str,
    factor: str,
    factor_questions: list[str],
    metadata: dict,
    extractions: list[dict],
    source_path: str,
    target_mode: str,
) -> dict:
    prompt = build_prompt(
        raw_text=raw_text,
        ticker=ticker,
        form=form,
        report_date=report_date,
        factor=factor,
        factor_questions=factor_questions,
    )
    assistant = json.dumps(
        {
            "metadata": metadata,
            "extractions": extractions,
        },
        ensure_ascii=False,
        indent=2,
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MINER},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant},
        ],
        "prompt": prompt,
        "response": assistant,
        "ticker": ticker,
        "form": form,
        "report_date": report_date,
        "year": int(report_date[:4]),
        "factor": factor,
        "question_count": len(factor_questions),
        "extraction_count": len(extractions),
        "target_mode": target_mode,
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "source_path": source_path,
    }


def select_positive_extractions(
    grouped_answers: dict[str, list[dict]],
    factor_questions: list[str],
    *,
    max_quotes_per_question: int,
    max_total_quotes: int,
    bucket: float,
) -> tuple[list[dict], str]:
    answered_qkeys = sorted(grouped_answers.keys(), key=question_number)
    if not answered_qkeys:
        return [], "no_answer"

    if len(answered_qkeys) == 1:
        chosen_qkeys = answered_qkeys[:1]
        target_mode = "single_answer"
    elif bucket < 0.625:
        chosen_qkeys = answered_qkeys[: min(3, len(answered_qkeys))]
        target_mode = "multi_answer"
    else:
        chosen_qkeys = answered_qkeys[:1]
        target_mode = "single_answer"

    merged: list[dict] = []
    for qkey in chosen_qkeys:
        merged.extend(dedupe_quotes(grouped_answers[qkey], max_quotes_per_question))

    deduped = dedupe_quotes(merged, max_total_quotes)
    return deduped, target_mode


def build_rows(
    *,
    raw_index: dict,
    task12_index: dict,
    factor_to_qkeys: dict[str, set[str]],
    args,
) -> dict[str, list[dict]]:
    split_rows: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    negative_pool: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}

    for key, t12_path in sorted(task12_index.items()):
        raw_path = raw_index.get(key)
        if raw_path is None:
            continue

        split = assign_fixed_split(int(key[2][:4]))
        if split is None:
            continue

        try:
            raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
            if len(raw_text) // 4 > args.max_approx_tokens:
                continue
            t12_obj = json.loads(t12_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        metadata = parse_metadata(t12_obj, key[0], key[1], key[2])
        answers_by_factor: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        for extraction in t12_obj.get("extractions", []):
            factor = extraction.get("factor")
            qkey = extraction.get("question_key")
            if not factor or not qkey:
                continue
            answers_by_factor[str(factor)][str(qkey)].append(extraction)

        for factor, all_qkeys in factor_to_qkeys.items():
            factor_questions = sorted(all_qkeys, key=question_number)
            if not factor_questions:
                continue
            grouped_answers = answers_by_factor.get(factor, {})
            bucket = stable_bucket(key[0], key[1], key[2], factor)

            if grouped_answers:
                selected, target_mode = select_positive_extractions(
                    grouped_answers,
                    factor_questions,
                    max_quotes_per_question=args.max_quotes_per_question,
                    max_total_quotes=args.max_total_quotes,
                    bucket=bucket,
                )
                if selected:
                    split_rows[split].append(
                        build_row(
                            raw_text=raw_text,
                            ticker=key[0],
                            form=key[1],
                            report_date=key[2],
                            factor=factor,
                            factor_questions=factor_questions,
                            metadata=metadata,
                            extractions=selected,
                            source_path=str(t12_path),
                            target_mode=target_mode,
                        )
                    )
            else:
                negative_pool[split].append(
                    build_row(
                        raw_text=raw_text,
                        ticker=key[0],
                        form=key[1],
                        report_date=key[2],
                        factor=factor,
                        factor_questions=factor_questions,
                        metadata=metadata,
                        extractions=[],
                        source_path=str(t12_path),
                        target_mode="no_answer",
                    )
                )

    return split_rows, negative_pool


def rebalance_train_rows(rows: list[dict], max_rows_per_factor: int) -> list[dict]:
    if max_rows_per_factor <= 0:
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["factor"])].append(row)
    output: list[dict] = []
    for factor, items in grouped.items():
        items = sorted(
            items,
            key=lambda x: hashlib.sha1(
                "||".join(
                    [
                        str(x["ticker"]),
                        str(x["report_date"]),
                        str(x["factor"]),
                        str(x["target_mode"]),
                        str(x["source_path"]),
                    ]
                ).encode("utf-8")
            ).hexdigest(),
        )
        output.extend(items[:max_rows_per_factor])
    return output


def add_negative_rows(split_rows: dict[str, list[dict]], negative_pool: dict[str, list[dict]], negative_ratio: float) -> None:
    for split in split_rows:
        positives = split_rows[split]
        negatives = negative_pool[split]
        if not positives or not negatives or negative_ratio <= 0:
            continue
        target_negatives = min(len(negatives), int(len(positives) * negative_ratio / max(1e-6, 1 - negative_ratio)))
        negatives = sorted(
            negatives,
            key=lambda x: hashlib.sha1(
                "||".join([str(x["ticker"]), str(x["report_date"]), str(x["factor"]), str(x["source_path"])]).encode("utf-8")
            ).hexdigest(),
        )
        split_rows[split].extend(negatives[:target_negatives])


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    raw_root = Path(args.raw_root)
    extract_root = Path(args.extract_root)

    raw_index = {
        parsed: path
        for path in raw_root.rglob("*.md")
        if (parsed := parse_raw_filename(path))
    }
    task12_index, factor_to_qkeys = index_task12_files(extract_root)
    split_rows, negative_pool = build_rows(
        raw_index=raw_index,
        task12_index=task12_index,
        factor_to_qkeys=factor_to_qkeys,
        args=args,
    )

    before_train = len(split_rows["train"])
    split_rows["train"] = rebalance_train_rows(split_rows["train"], args.max_train_rows_per_factor)
    add_negative_rows(split_rows, negative_pool, args.negative_ratio)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = DatasetDict({split: Dataset.from_list(rows) for split, rows in split_rows.items()})
    dataset.save_to_disk(str(output_dir))

    for split in ("train", "validation", "test"):
        mode_counts = Counter(row["target_mode"] for row in split_rows[split])
        logging.info("%s rows=%d mode_counts=%s", split, len(split_rows[split]), dict(mode_counts))

    logging.info(
        "Saved extract SFT v5 dataset to %s | train=%d (before factor rebalance=%d) validation=%d test=%d",
        output_dir,
        len(split_rows["train"]),
        before_train,
        len(split_rows["validation"]),
        len(split_rows["test"]),
    )


if __name__ == "__main__":
    main()
