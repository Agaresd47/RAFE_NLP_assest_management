import argparse
import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import aiohttp
from datasets import load_from_disk
from tqdm import tqdm


DEFAULT_SGLANG_URL = "http://localhost:30000"
DEFAULT_DATASET = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
DEFAULT_SPLIT = "test"
DEFAULT_OUTPUT_JSON = "/gpfs/projects/p32908/nlp_result/sglang_ttc_test.json"
DEFAULT_BATCH_SIZE = 16
DEFAULT_TTC_N = 3
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_REQUEST_CONCURRENCY = 32
DEFAULT_MAX_RETRIES = 3

LABEL_ORDER = ["very_negative", "negative", "neutral", "positive", "very_positive"]
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


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Lightweight TTC on top of an SGLang Qwen3 server.")
    parser.add_argument("--sglang-url", default=DEFAULT_SGLANG_URL)
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--ttc-n", type=int, default=DEFAULT_TTC_N)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--request-concurrency", type=int, default=DEFAULT_REQUEST_CONCURRENCY)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    return parser


def normalize_label(value):
    if value is None:
        return ""
    key = str(value).strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    normalized = LABEL_ALIASES.get(key, "")
    return normalized if normalized in LABEL_ORDER else ""


def normalize_confidence(value):
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(conf, 1.0))


def parse_json_output(content):
    if content is None:
        return {}
    text = str(content).replace("<|im_end|>", "").strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        matches = list(re.finditer(r"\{.*?\}", text, re.DOTALL))
        for match in reversed(matches):
            try:
                parsed = json.loads(match.group())
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                continue
    return {}


async def infer_one(session, sglang_url, prompt, *, max_new_tokens, temperature, top_p, semaphore, max_retries):
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "separate_reasoning": False,
        },
    }
    async with semaphore:
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                async with session.post(f"{sglang_url}/v1/chat/completions", json=payload) as resp:
                    data = await resp.json()
                    choices = data.get("choices") or []
                    if not choices:
                        return {"content": None, "reasoning": None, "response": data, "parsed": {}}
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                    reasoning = message.get("reasoning_content")
                    parsed = parse_json_output(content)
                    return {
                        "content": content,
                        "reasoning": reasoning,
                        "response": data,
                        "parsed": parsed,
                    }
            except Exception as exc:
                last_error = repr(exc)
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
        return {
            "content": None,
            "reasoning": None,
            "response": {"error": last_error},
            "parsed": {},
        }


def aggregate_candidates(candidates):
    valid = []
    for cand in candidates:
        parsed = cand.get("parsed") or {}
        label = normalize_label(parsed.get("sentiment_label"))
        conf = normalize_confidence(parsed.get("confidence_score"))
        if label:
            valid.append((label, conf, cand))

    if not valid:
        return {
            "sentiment_label": "",
            "confidence_score": None,
            "selected_index": None,
            "vote_counter": {},
        }

    vote_counter = Counter(label for label, _, _ in valid)
    winner_label = sorted(
        vote_counter.keys(),
        key=lambda x: (vote_counter[x], -abs(LABEL_ORDER.index(x) - LABEL_ORDER.index("neutral"))),
        reverse=True,
    )[0]

    winner_pool = [(idx, conf, cand) for idx, (label, conf, cand) in enumerate(valid) if label == winner_label]
    selected_pos, _, selected_cand = max(
        winner_pool,
        key=lambda item: (-1.0 if item[1] is None else item[1], item[0]),
    )

    conf_values = [conf for label, conf, _ in valid if label == winner_label and conf is not None]
    agg_conf = round(sum(conf_values) / len(conf_values), 4) if conf_values else None

    return {
        "sentiment_label": winner_label,
        "confidence_score": agg_conf,
        "selected_index": selected_pos,
        "vote_counter": dict(vote_counter),
        "selected_candidate": selected_cand,
    }


async def infer_dataset(args):
    ds = load_from_disk(args.dataset_path)[args.split]
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    rows = []
    progress = tqdm(total=len(ds), desc="TTC inference")
    semaphore = asyncio.Semaphore(max(1, args.request_concurrency))
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=300)
    connector = aiohttp.TCPConnector(limit=max(1, args.request_concurrency))
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for start in range(0, len(ds), args.batch_size):
            batch = ds.select(range(start, min(start + args.batch_size, len(ds))))
            tasks = []
            task_meta = []
            for ex in batch:
                prompt = ex["messages"][1]["content"]
                for _ in range(args.ttc_n):
                    tasks.append(
                        infer_one(
                            session,
                            args.sglang_url,
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            semaphore=semaphore,
                            max_retries=args.max_retries,
                        )
                    )
                task_meta.append(ex)

            outputs = await asyncio.gather(*tasks)

            grouped = []
            idx = 0
            for _ in task_meta:
                grouped.append(outputs[idx : idx + args.ttc_n])
                idx += args.ttc_n

            for ex, candidates in zip(task_meta, grouped):
                agg = aggregate_candidates(candidates)
                selected = agg.get("selected_candidate") or {}
                rows.append(
                    {
                        "Date": ex.get("report_date"),
                        "Symbol": ex.get("ticker"),
                        "Form": ex.get("form"),
                        "Factor": ex.get("factor"),
                        "Signal": agg.get("sentiment_label", ""),
                        "Confidence": agg.get("confidence_score"),
                        "Think": selected.get("reasoning"),
                        "Output": selected.get("content"),
                        "Votes": agg.get("vote_counter", {}),
                        "Candidates": [
                            {
                                "content": cand.get("content"),
                                "reasoning": cand.get("reasoning"),
                                "parsed": cand.get("parsed"),
                            }
                            for cand in candidates
                        ],
                    }
                )
            progress.update(len(batch))
    progress.close()
    return rows


def main():
    args = build_arg_parser().parse_args()
    rows = asyncio.run(infer_dataset(args))
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
