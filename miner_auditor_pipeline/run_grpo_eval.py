from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import aiohttp
import requests
from datasets import load_from_disk
from tqdm import tqdm

try:
    from .common import ensure_dir
except ImportError:
    from common import ensure_dir

from task13_dataset_common import LABEL_ORDER, coerce_confidence, label_distance, normalize_label


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR = str(LOCAL_DATA_DIR / "hf_cot_sft")
DEFAULT_SGLANG_URL = "http://localhost:30000"
OUTPUT_DIR = "/gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/grpo_eval"
SPLIT = "test"
MAX_NEW_TOKENS = 768
TEMPERATURE = 0.6
TOP_P = 0.9
BATCH_SIZE = 16
TTC_N = 1

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRPO eval through an SGLang server on local hf_cot_sft.")
    parser.add_argument("--sglang-url", default=DEFAULT_SGLANG_URL)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=TOP_P)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--ttc-n", type=int, default=TTC_N)
    parser.add_argument("--max-filings", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=True)
    return parser


def wait_for_server(sglang_url: str) -> None:
    print("Waiting for sglang server...", flush=True)
    for _ in range(60):
        try:
            requests.get(f"{sglang_url}/health", timeout=2)
            print("Server ready.", flush=True)
            return
        except Exception:
            time.sleep(5)
    raise RuntimeError("sglang server startup timed out")


async def infer_single(
    session,
    sglang_url,
    msgs,
    *,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
):
    async with session.post(
        f"{sglang_url}/v1/chat/completions",
        json={
            "model": "default",
            "messages": msgs,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
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


async def infer_batch_async(
    sglang_url,
    batch_messages,
    *,
    max_new_tokens,
    temperature,
    top_p,
    enable_thinking,
):
    async with aiohttp.ClientSession() as session:
        tasks = [
            infer_single(
                session,
                sglang_url,
                msgs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                enable_thinking=enable_thinking,
            )
            for msgs in batch_messages
        ]
        return await asyncio.gather(*tasks)


def infer_batch(sglang_url, batch_messages, *, max_new_tokens, temperature, top_p, enable_thinking):
    return asyncio.run(
        infer_batch_async(
            sglang_url,
            batch_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
    )


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def parse_model_output(text: str, response=None) -> dict:
    raw = str(text or "")
    thinking = None
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            reasoning = message.get("reasoning_content")
            if reasoning:
                thinking = str(reasoning).strip()
    if not thinking:
        think_match = THINK_RE.search(raw)
        thinking = think_match.group(1).strip() if think_match else None

    parsed = None
    json_text = None
    for match in reversed(list(JSON_BLOCK_RE.finditer(raw))):
        candidate = match.group(0).strip()
        parsed = safe_json_load(candidate)
        if parsed is not None:
            json_text = candidate
            break

    if parsed is None:
        parsed = {}

    sentiment = normalize_label(parsed.get("sentiment_label"))
    confidence = coerce_confidence(parsed.get("confidence_score")) if "confidence_score" in parsed else None
    reasoning = parsed.get("reasoning_chain")
    if not reasoning and thinking:
        reasoning = thinking

    return {
        "raw_text": raw,
        "think_text": thinking,
        "json_text": json_text,
        "parsed_json": parsed if isinstance(parsed, dict) else {},
        "reasoning_chain": reasoning,
        "sentiment_label": sentiment,
        "confidence_score": confidence,
        "parse_success": json_text is not None,
    }


def parse_gt_output(text: str) -> dict:
    raw = str(text or "")
    parsed = safe_json_load(raw)
    if not isinstance(parsed, dict):
        parsed = {}
    return {
        "raw_text": raw,
        "reasoning_chain": parsed.get("reasoning_chain"),
        "sentiment_label": normalize_label(parsed.get("sentiment_label")),
        "confidence_score": coerce_confidence(parsed.get("confidence_score")) if "confidence_score" in parsed else None,
        "parsed_json": parsed,
    }


def aggregate_candidates(candidates: list[dict]) -> dict:
    valid = []
    for cand in candidates:
        label = normalize_label(cand.get("sentiment_label"))
        conf = cand.get("confidence_score")
        if label:
            valid.append((label, conf, cand))

    if not valid:
        return {
            "sentiment_label": "",
            "confidence_score": None,
            "selected_candidate": candidates[0] if candidates else {},
            "vote_counter": {},
            "ttc_n": len(candidates),
        }

    vote_counter = Counter(label for label, _, _ in valid)
    winner_label = sorted(
        vote_counter.keys(),
        key=lambda x: (vote_counter[x], -abs(LABEL_ORDER.index(x) - LABEL_ORDER.index("neutral"))),
        reverse=True,
    )[0]

    winner_pool = [(conf, cand) for label, conf, cand in valid if label == winner_label]
    selected_conf, selected_cand = max(
        winner_pool,
        key=lambda item: (-1.0 if item[0] is None else item[0]),
    )
    conf_values = [conf for label, conf, _ in valid if label == winner_label and conf is not None]
    agg_conf = round(sum(conf_values) / len(conf_values), 4) if conf_values else selected_conf

    return {
        "sentiment_label": winner_label,
        "confidence_score": agg_conf,
        "selected_candidate": selected_cand,
        "vote_counter": dict(vote_counter),
        "ttc_n": len(candidates),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def apply_max_filings(eval_ds, max_filings: int):
    if max_filings <= 0:
        return eval_ds
    selected_indices = []
    seen_filing_keys = set()
    for idx, ex in enumerate(eval_ds):
        filing_key = (ex.get("ticker"), ex.get("form"), ex.get("report_date"))
        if filing_key not in seen_filing_keys:
            if len(seen_filing_keys) >= max_filings:
                continue
            seen_filing_keys.add(filing_key)
        selected_indices.append(idx)
    return eval_ds.select(selected_indices)


def compute_metrics(gt_rows: list[dict], pred_rows: list[dict]) -> dict:
    total = len(gt_rows)
    exact = 0
    parse_success = 0
    sentiment_distances: list[float] = []
    confidence_abs_errors: list[float] = []

    for gt, pred in zip(gt_rows, pred_rows):
        gt_label = normalize_label(gt["sentiment_label"])
        pred_label = normalize_label(pred["sentiment_label"])
        if pred.get("parse_success"):
            parse_success += 1
        if gt_label == pred_label:
            exact += 1
        sentiment_distances.append(float(label_distance(gt_label, pred_label)))
        gt_conf = gt.get("confidence_score")
        pred_conf = pred.get("confidence_score")
        if gt_conf is not None and pred_conf is not None:
            confidence_abs_errors.append(abs(float(gt_conf) - float(pred_conf)))

    max_distance = max(1, len(LABEL_ORDER) - 1)
    return {
        "count": total,
        "parse_success_rate": round(parse_success / total, 6) if total else None,
        "sentiment_accuracy": round(exact / total, 6) if total else None,
        "sentiment_mae_5way": round(mean(sentiment_distances), 6) if sentiment_distances else None,
        "sentiment_mae_normalized": round(mean([d / max_distance for d in sentiment_distances]), 6)
        if sentiment_distances
        else None,
        "confidence_mae": round(mean(confidence_abs_errors), 6) if confidence_abs_errors else None,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    wait_for_server(args.sglang_url)

    print("[1/4] loading dataset...", flush=True)
    ds = load_from_disk(args.data_dir)
    eval_ds = ds[args.split]
    eval_ds = apply_max_filings(eval_ds, args.max_filings)
    if args.limit > 0:
        eval_ds = eval_ds.select(range(min(args.limit, len(eval_ds))))
    print(f"[dataset] split={args.split} count={len(eval_ds)}", flush=True)

    print("[2/4] running GRPO eval via sglang...", flush=True)
    gt_rows: list[dict] = []
    pred_rows: list[dict] = []
    raw_candidate_rows: list[dict] = []

    for start in tqdm(range(0, len(eval_ds), args.batch_size), desc="GRPO eval", unit="batch"):
        batch = eval_ds.select(range(start, min(start + args.batch_size, len(eval_ds))))
        msgs = []
        metas = []
        for ex in batch:
            prompt_messages = ex["messages"][:2]
            for _ in range(max(1, args.ttc_n)):
                msgs.append(prompt_messages)
            metas.append(ex)

        outputs = infer_batch(
            args.sglang_url,
            msgs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
        )

        idx = 0
        for ex in metas:
            candidates = []
            for sample_idx in range(max(1, args.ttc_n)):
                out = outputs[idx]
                parsed = parse_model_output(out.get("content"), out.get("response"))
                candidates.append(parsed)
                raw_candidate_rows.append(
                    {
                        "ticker": ex.get("ticker"),
                        "form": ex.get("form"),
                        "report_date": ex.get("report_date"),
                        "factor": ex.get("factor"),
                        "sample_idx": sample_idx,
                        **parsed,
                    }
                )
                idx += 1

            aggregated = aggregate_candidates(candidates)
            selected = aggregated["selected_candidate"]
            gt = parse_gt_output(ex["messages"][-1]["content"])

            gt_rows.append(
                {
                    "ticker": ex.get("ticker"),
                    "form": ex.get("form"),
                    "report_date": ex.get("report_date"),
                    "factor": ex.get("factor"),
                    **gt,
                }
            )
            pred_rows.append(
                {
                    "ticker": ex.get("ticker"),
                    "form": ex.get("form"),
                    "report_date": ex.get("report_date"),
                    "factor": ex.get("factor"),
                    "ttc_n": aggregated["ttc_n"],
                    "vote_counter": aggregated["vote_counter"],
                    "sentiment_label": aggregated["sentiment_label"],
                    "confidence_score": aggregated["confidence_score"],
                    "raw_text": selected.get("raw_text"),
                    "think_text": selected.get("think_text"),
                    "json_text": selected.get("json_text"),
                    "parsed_json": selected.get("parsed_json"),
                    "reasoning_chain": selected.get("reasoning_chain"),
                    "parse_success": selected.get("parse_success"),
                }
            )

    metrics = compute_metrics(gt_rows, pred_rows)
    metrics["ttc_n"] = max(1, args.ttc_n)
    metrics["enable_thinking"] = bool(args.enable_thinking)

    print("[3/4] writing outputs...", flush=True)
    (out_dir / "grpo_pred_rows.json").write_text(json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "grpo_gt_rows.json").write_text(json.dumps(gt_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "grpo_candidate_rows.json").write_text(
        json.dumps(raw_candidate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[4/4] done", flush=True)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"saved -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
