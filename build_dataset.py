import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path

from datasets import Dataset, DatasetDict


# --- 配置正则 ---
RAW_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9._-]+)_(?P<form>10-K|10-Q)_(?P<date>\d{4}-\d{2}-\d{2})\.md$",
    re.IGNORECASE,
)
EXTRACT_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9._-]+)_(?P<date>\d{2}-\d{2}-\d{4})_(?P<form>10-K|10-Q)"
    r"_(?P<kind>TASK12_EXTRACTIONS|TASK13_AUDIT|FACTORS_DETAILED)\.json$",
    re.IGNORECASE,
)

# --- System Prompts ---
SYSTEM_MINER = (
    "You are a Financial Data Engineer. "
    "Extract original quotes from the provided source text that answer the specific questions in the schema. "
    "For each question, if an answer is found, provide the original_quote and a relevance_confidence from 0 to 1. "
    "If no answer is found, ignore the question entirely. "
    "Output only valid JSON."
)

SYSTEM_AUDITOR = (
    "You are a Senior Equity Strategist. "
    "Audit the current evidence by comparing it against the historical context. "
    "Analyze the deviation from the baseline and sector context. "
    "Score the sentiment as one of: Very Bad, Bad, Neutral, Good, Very Good. "
    "Provide a reasoning_chain, final sentiment label, and confidence score. "
    "Output only valid JSON."
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def load_questions(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_date_to_iso(date_str: str) -> str | None:
    for fmt in ("%Y-%m-%d", "%m-%d-%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def parse_raw_filename(path: Path):
    m = RAW_RE.match(path.name)
    if not m:
        return None
    iso = parse_date_to_iso(m.group("date"))
    return (m.group("ticker").upper(), m.group("form").upper(), iso) if iso else None


def parse_extract_filename(path: Path):
    m = EXTRACT_RE.match(path.name)
    if not m:
        return None
    iso = parse_date_to_iso(m.group("date"))
    return (
        (m.group("ticker").upper(), m.group("form").upper(), iso, m.group("kind").upper())
        if iso else None
    )


def assign_fixed_split(year: int) -> str | None:
    if 2015 <= year <= 2022: return "train"
    if year == 2023:         return "validation"
    if 2024 <= year <= 2026: return "test"
    return None


# ---------------------------------------------------------------------------
# Miner (Task 1.2)
# ---------------------------------------------------------------------------


def build_miner_samples(
    raw_text: str,
    t12_obj: dict,
    questions: dict,
    ticker: str,
    form: str,
    report_date: str,
) -> list[dict]:
    return [{
        "messages": [
            {"role": "system",    "content": SYSTEM_MINER},
            {"role": "user",      "content": build_user_prompt_miner(raw_text, questions, ticker, form, report_date)},
            {"role": "assistant", "content": sanitize_miner_output(t12_obj)},
        ],
        "ticker":      ticker,
        "form":        form,
        "report_date": report_date,
    }]


def build_user_prompt_miner(
    raw_text: str,
    questions: dict,
    ticker: str,
    form: str,
    report_date: str,
) -> str:
    return (
        f"Task: Miner (1.2)\n"
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}\n\n"
        f"Schema:\n{json.dumps(questions, ensure_ascii=False, indent=2)}\n\n"
        f"Text:\n{raw_text}"
    )


def sanitize_miner_output(t12_obj: dict) -> str:
    clean_data = {
        "metadata": t12_obj.get("metadata"),
        "extractions": [
            {
                "factor":               e.get("factor"),
                "question_key":         e.get("question_key"),
                "original_quote":       e.get("original_quote"),
                "relevance_confidence": e.get("relevance_confidence"),
            }
            for e in t12_obj.get("extractions", [])
        ],
    }
    return json.dumps(clean_data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Auditor (Task 1.3)
# ---------------------------------------------------------------------------

def build_user_prompt_auditor_single(
    audit: dict,
    ticker: str,
    form: str,
    report_date: str,
) -> str:
    f_name = audit.get("factor", "unknown")
    lines = [
        f"Task: Auditor (1.3)",
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}",
        f"Factor: {f_name}",
        "",
        "[Current Evidence]",
    ]

    for e in audit.get("evidence_used", []):
        lines.append(
            f"  q_key : {e.get('question_key', '—')}\n"
            f"  quote : {e.get('original_quote', '—')}\n"
            f"  conf  : {e.get('relevance_confidence', '—')}"
        )

    hist = audit.get("historical_context", [])
    if hist:
        lines.append("[Historical Context]")
        for h in hist:
            lines.append(
                f"  [{h.get('report_date', '—')} {h.get('filing', '—')}] "
                f"{h.get('fact', '—')}  (conf: {h.get('relevance_confidence', '—')})"
            )

    lines.append("\nOutput sentiment_label and confidence_score for this factor.")
    return "\n".join(lines)


def sanitize_auditor_output_single(audit: dict, include_cot: bool = True) -> str:
    f_name    = audit.get("factor", "unknown")
    res       = audit.get("audit_result", {})
    reasoning = res.get("reasoning_chain", "")

    core = json.dumps({
        "factor":     f_name,
        "sentiment":  res.get("sentiment_label"),
        "confidence": res.get("confidence_score"),
    }, ensure_ascii=False, indent=2)

    if include_cot and reasoning:
        return f"<think>\n{reasoning}\n</think>\n{core}"

    return core


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root",     default=r"E:\nlpas\stock\MDA_Raw")
    ap.add_argument("--extract_root", default=r"E:\nlpas\stock\Extract")
    ap.add_argument("--questions",    default=r"E:\nlpas\stock\tech_sentiment_questions.json")
    ap.add_argument("--miner_out",    default=r"E:\nlpas\stock\hf_miner_v3")
    ap.add_argument("--auditor_out",  default=r"E:\nlpas\stock\hf_auditor_v3")
    ap.add_argument("--auditor_mode", choices=["cot", "no_cot", "both"], default="both")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    raw_root     = Path(args.raw_root)
    extract_root = Path(args.extract_root)
    questions    = load_questions(Path(args.questions))

    Path(args.miner_out).mkdir(parents=True, exist_ok=True)
    Path(args.auditor_out).mkdir(parents=True, exist_ok=True)

    # --- 索引文件 ---
    raw_index    = {parse_raw_filename(p): p for p in raw_root.rglob("*.md") if parse_raw_filename(p)}
    task12_index: dict = {}
    task13_index: dict = {}

    for p in extract_root.rglob("*.json"):
        parsed = parse_extract_filename(p)
        if not parsed:
            continue
        key, kind = parsed[:3], parsed[3]
        if kind == "TASK12_EXTRACTIONS":
            task12_index[key] = p
        elif kind == "TASK13_AUDIT":
            task13_index[key] = p

    # -----------------------------------------------------------------------
    # 1. Miner 数据集
    # -----------------------------------------------------------------------
    miner_rows: dict[str, list] = {"train": [], "validation": [], "test": []}

    for key, t12_path in sorted(task12_index.items()):
        raw_path = raw_index.get(key)
        if not raw_path:
            logging.warning(f"No raw MD for {key}, skipping.")
            continue

        split = assign_fixed_split(int(key[2][:4]))
        if not split:
            continue

        try:
            raw_text = raw_path.read_text(encoding="utf-8", errors="ignore")
            if len(raw_text) // 4 > 259935:
                logging.warning(f"跳过超长文件 {key}: 约 {len(raw_text)//4} tokens")
                continue
            t12_obj  = json.loads(t12_path.read_text(encoding="utf-8", errors="ignore"))

            samples = build_miner_samples(
                raw_text, t12_obj, questions, *key
            )
            miner_rows[split].extend(samples)  # ← extend 不是 append

        except Exception as e:
            logging.error(f"Error processing {t12_path}: {e}")
            continue

    DatasetDict({s: Dataset.from_list(miner_rows[s]) for s in miner_rows}).save_to_disk(
        Path(args.miner_out)
    )
    logging.info(
        f"Miner saved → train={len(miner_rows['train'])} "
        f"val={len(miner_rows['validation'])} test={len(miner_rows['test'])}"
    )

    # -----------------------------------------------------------------------
    # 2. Auditor 数据集
    # -----------------------------------------------------------------------
    auditor_rows: dict[str, list] = {"train": [], "validation": [], "test": []}

    for key, t13_path in sorted(task13_index.items()):
        if key not in task12_index:
            logging.warning(f"No T12 pair for {key}, skipping auditor.")
            continue

        split = assign_fixed_split(int(key[2][:4]))
        if not split:
            continue

        try:
            t13_obj = json.loads(t13_path.read_text(encoding="utf-8", errors="ignore"))

            for audit in t13_obj.get("factor_audits", []):
                user     = build_user_prompt_auditor_single(audit, *key)
                base_row = {
                    "ticker":      key[0],
                    "form":        key[1],
                    "report_date": key[2],
                    "factor":      audit.get("factor", "unknown"),
                }

                if args.auditor_mode in ("cot", "both"):
                    auditor_rows[split].append({
                        "messages": [
                            {"role": "system",    "content": SYSTEM_AUDITOR},
                            {"role": "user",      "content": user},
                            {"role": "assistant", "content": sanitize_auditor_output_single(audit, True)},
                        ],
                        "cot_visibility": "with_cot",
                        **base_row,
                    })

                if args.auditor_mode in ("no_cot", "both"):
                    auditor_rows[split].append({
                        "messages": [
                            {"role": "system",    "content": SYSTEM_AUDITOR},
                            {"role": "user",      "content": user},
                            {"role": "assistant", "content": sanitize_auditor_output_single(audit, False)},
                        ],
                        "cot_visibility": "no_cot",
                        **base_row,
                    })

        except Exception as e:
            logging.error(f"Error processing {t13_path}: {e}")
            continue

    DatasetDict({s: Dataset.from_list(auditor_rows[s]) for s in auditor_rows}).save_to_disk(
        Path(args.auditor_out)
    )
    logging.info(
        f"Auditor saved → train={len(auditor_rows['train'])} "
        f"val={len(auditor_rows['validation'])} test={len(auditor_rows['test'])}"
    )


if __name__ == "__main__":
    main()