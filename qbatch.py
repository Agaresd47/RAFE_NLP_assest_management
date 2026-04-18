import argparse
import asyncio
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import pandas as pd
import requests
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

'''
Example:
python -m sglang.launch_server \
    --model-path /scratch/xla2767/hold2/models/qwen3_8b_thinking_sft_merged \
    --port 30000 \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --mem-fraction-static 0.85 &

python qbatch.py --model-path /path/to/merged-model --miner-dataset /path/to/miner_ds --auditor-dataset /path/to/auditor_ds
'''

DEFAULT_MODEL_PATH = "/scratch/xla2767/hold2/models/qwen3_8b_thinking_sft_merged"
DEFAULT_SGLANG_URL = "http://localhost:30000"
DEFAULT_OUTPUT_CSV = "/gpfs/projects/p32908/backtest_signals.csv"
DEFAULT_OUTPUT_JSON = "/gpfs/projects/p32908/backtest_signals.json"
DEFAULT_MINER_DATASET = ""
DEFAULT_AUDITOR_DATASET = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
DEFAULT_DATASET_CACHE_DIR = "/gpfs/projects/p32908/hf_cache"
DEFAULT_SPLIT = "test"

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
    "Score the sentiment as one of: very_negative, negative, neutral, positive, very_positive. "
    "Provide sentiment_label and confidence_score. "
    "Output only valid JSON."
)

LABEL_ALIASES = {
    "very bad": "very_negative",
    "bad": "negative",
    "neutral": "neutral",
    "good": "positive",
    "very good": "very_positive",
    "very negative": "very_negative",
    "negative": "negative",
    "positive": "positive",
    "very positive": "very_positive",
    "very_negative": "very_negative",
    "very_positive": "very_positive",
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run batch miner/auditor inference against a merged model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Merged model path used by tokenizer and sglang.")
    parser.add_argument("--sglang-url", default=DEFAULT_SGLANG_URL, help="Base URL for the running sglang server.")
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV, help="Where to write the final CSV.")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Where to write the full JSON output.")
    parser.add_argument(
        "--miner-dataset",
        default=DEFAULT_MINER_DATASET,
        help="HF dataset name or local dataset directory for the miner data. Leave empty to skip miner stage.",
    )
    parser.add_argument(
        "--auditor-dataset",
        default=DEFAULT_AUDITOR_DATASET,
        help="HF dataset name or local dataset directory for the auditor data.",
    )
    parser.add_argument(
        "--dataset-cache-dir",
        default=DEFAULT_DATASET_CACHE_DIR,
        help="Cache directory used when loading remote HF datasets.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split to evaluate, e.g. test or validation.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of batches to run. Omit for full evaluation.",
    )
    parser.add_argument(
        "--offline-datasets",
        action="store_true",
        help="Set HF_DATASETS_OFFLINE=1 before loading datasets.",
    )
    return parser


def load_dataset_source(source, cache_dir=None):
    if source is None or str(source).strip() == "":
        return None
    source_path = Path(source)
    if source_path.exists():
        return load_from_disk(str(source_path))
    return load_dataset(source, cache_dir=cache_dir)


def resolve_split(dataset_obj, split):
    if hasattr(dataset_obj, "keys"):
        if split in dataset_obj:
            return dataset_obj[split]
        available = list(dataset_obj.keys())
        raise KeyError(f"Split '{split}' not found. Available splits: {available}")
    return dataset_obj


def wait_for_server(sglang_url):
    print("Waiting for sglang server...")
    for _ in range(60):
        try:
            requests.get(f"{sglang_url}/health", timeout=2)
            print("Server ready.")
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("sglang server startup timed out")


async def infer_single(session, sglang_url, msgs, max_new_tokens):
    async with session.post(
        f"{sglang_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": msgs,
            "max_tokens": max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": True},
                "separate_reasoning": False,
            },
        },
    ) as resp:
        data = await resp.json()
        choices = data.get("choices") or []
        if not choices:
            return {"content": None, "response": data}
        message = choices[0].get("message") or {}
        return {"content": message.get("content"), "response": data}


async def infer_batch_async(sglang_url, batch_messages, max_new_tokens=512):
    async with aiohttp.ClientSession() as session:
        tasks = [infer_single(session, sglang_url, msgs, max_new_tokens) for msgs in batch_messages]
        return await asyncio.gather(*tasks)


def infer_batch(sglang_url, batch_messages, max_new_tokens=512):
    return asyncio.run(infer_batch_async(sglang_url, batch_messages, max_new_tokens))


def parse_output(raw):
    if raw is None:
        return {}
    clean = (
        str(raw)
        .replace("<tool_call>", "")
        .replace("</tool_call>", "")
        .replace("<|im_end|>", "")
        .strip()
    )
    try:
        parsed = json.loads(clean)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        matches = list(re.finditer(r"\{.*?\}", clean, re.DOTALL))
        for match in reversed(matches):
            try:
                parsed = json.loads(match.group())
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                continue
    return {}


def normalize_sentiment(value):
    if value is None:
        return ""
    key = str(value).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    return LABEL_ALIASES.get(key, str(value).strip())


def normalize_confidence(value):
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return ""
    return max(0.0, min(conf, 1.0))


def extract_think_text(raw, response):
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            reasoning = message.get("reasoning_content")
            if reasoning:
                return str(reasoning).strip()
    if raw is not None:
        match = re.search(r"<think>\s*(.*?)\s*</think>", str(raw), re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def get_content_and_response(item):
    if isinstance(item, dict) and "response" in item and "content" in item:
        return item.get("content"), item.get("response")
    return item, None


def build_user_prompt_auditor_single(audit, ticker, form, report_date):
    f_name = audit.get("factor", "unknown")
    lines = [
        "Task: Auditor (1.3)",
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


def build_direct_auditor_inputs(auditor_split):
    auditor_inputs = []
    for x in auditor_split:
        messages = x.get("messages") or []
        if len(messages) < 2:
            continue
        auditor_inputs.append(
            (
                x.get("ticker", ""),
                x.get("form", ""),
                x.get("report_date", ""),
                x.get("factor", ""),
                messages[1]["content"],
            )
        )
    return auditor_inputs


def main():
    args = build_parser().parse_args()
    if args.offline_datasets:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    print("Starting batch eval.")
    wait_for_server(args.sglang_url)

    print("Loading datasets...")
    miner_ds = load_dataset_source(args.miner_dataset, cache_dir=args.dataset_cache_dir)
    auditor_ds = load_dataset_source(args.auditor_dataset, cache_dir=args.dataset_cache_dir)

    auditor_split = resolve_split(auditor_ds, args.split)
    miner_split = resolve_split(miner_ds, args.split) if miner_ds is not None else None

    direct_auditor_mode = miner_split is None
    if not direct_auditor_mode and "cot_visibility" not in getattr(auditor_split, "column_names", []):
        direct_auditor_mode = True

    if direct_auditor_mode:
        print("\n=== Direct Auditor Mode ===")
        auditor_inputs = build_direct_auditor_inputs(auditor_split)
        if args.max_batches is not None:
            auditor_inputs = auditor_inputs[: args.max_batches * args.batch_size]
        print(f"Built {len(auditor_inputs)} direct auditor prompts.")
    else:
        print("Building historical context index...")
        historical_index = defaultdict(list)
        for x in auditor_split:
            if x.get("cot_visibility") != "no_cot":
                continue
            ticker = x.get("ticker", "")
            form = x.get("form", "")
            date = x.get("report_date", "")
            factor = x.get("factor", "")
            user_content = x["messages"][1]["content"]
            hist_match = re.search(r"\[Historical Context\](.*?)(?:\nOutput|\Z)", user_content, re.DOTALL)
            if hist_match:
                historical_index[(ticker, form, date, factor)] = hist_match.group(1).strip()

        print("\n=== Step 1: Miner inference ===")
        miner_results = []
        for batch_index, i in enumerate(tqdm(range(0, len(miner_split), args.batch_size))):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            batch = miner_split.select(range(i, min(i + args.batch_size, len(miner_split))))
            msgs = [
                [{"role": "system", "content": SYSTEM_MINER}, {"role": "user", "content": x["messages"][1]["content"]}]
                for x in batch
            ]
            outputs = infer_batch(args.sglang_url, msgs, max_new_tokens=2048)

            for x, out in zip(batch, outputs):
                content, response = get_content_and_response(out)
                print("RAW OUTPUT:", str(content)[:100])
                meta = json.loads(x["messages"][2]["content"]).get("metadata", {})
                try:
                    miner_results.append(
                        (
                            meta.get("ticker", ""),
                            meta.get("filing", ""),
                            meta.get("report_date", ""),
                            parse_output(content).get("extractions", []),
                        )
                    )
                except Exception:
                    miner_results.append(("unknown", "unknown", "unknown", []))

        print(f"Miner done, {len(miner_results)} rows.")

        print("\n=== Step 2: Build auditor inputs ===")
        auditor_inputs = []
        for ticker, form, report_date, extractions in miner_results:
            factor_groups = defaultdict(list)
            for e in extractions:
                factor = e.get("factor", "unknown")
                factor_groups[factor].append(
                    {
                        "question_key": e.get("question_key", "—"),
                        "original_quote": e.get("original_quote", "—"),
                        "relevance_confidence": e.get("relevance_confidence", 0),
                    }
                )

            for factor, evidence_list in factor_groups.items():
                hist_raw = historical_index.get((ticker, form, report_date, factor), "")
                audit = {"factor": factor, "evidence_used": evidence_list, "historical_context": []}
                user_prompt = build_user_prompt_auditor_single(audit, ticker, form, report_date)
                if hist_raw:
                    user_prompt = user_prompt.replace(
                        "\nOutput sentiment_label and confidence_score for this factor.",
                        f"\n[Historical Context]\n{hist_raw}\n\nOutput sentiment_label and confidence_score for this factor.",
                    )
                auditor_inputs.append((ticker, form, report_date, factor, user_prompt))

        print(f"Built {len(auditor_inputs)} auditor prompts.")

    print("\n=== Step 3: Auditor inference ===")
    rows = []
    for i in tqdm(range(0, len(auditor_inputs), args.batch_size)):
        batch = auditor_inputs[i : i + args.batch_size]
        msgs = [
            [{"role": "system", "content": SYSTEM_AUDITOR}, {"role": "user", "content": item[4]}]
            for item in batch
        ]
        outputs = infer_batch(args.sglang_url, msgs, max_new_tokens=1024)

        for (ticker, form, date, factor, _), out in zip(batch, outputs):
            content, response = get_content_and_response(out)
            pred = parse_output(content)
            rows.append(
                {
                    "Date": date,
                    "Symbol": ticker,
                    "Form": form,
                    "Factor": factor,
                    "Signal": normalize_sentiment(pred.get("sentiment_label", pred.get("sentiment"))),
                    "Confidence": normalize_confidence(pred.get("confidence_score", pred.get("confidence"))),
                    "Output": content,
                    "Think": extract_think_text(content, response),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    Path(args.output_json).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {args.output_csv}, rows={len(df)}")
    print(f"Saved to {args.output_json}, rows={len(rows)}")
    print(df.head(10))


if __name__ == "__main__":
    main()
