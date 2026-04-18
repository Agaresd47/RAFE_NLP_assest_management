from __future__ import annotations

import argparse
import logging
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
from datasets import DatasetDict, load_from_disk
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_MODEL = "Skywork/Skywork-Reward-V2-Qwen3-4B"
ADAPTER_DIR = "/scratch/xla2767/hold2/models/cot_reward_model/checkpoint-387"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo"
MAX_LENGTH = 4096
NUM_SAMPLES = 5
MAX_EVAL_EXAMPLES = 64
OUTPUT_DIR = "/scratch/xla2767/hold2/data/nlp/eval_logs"
TOP_K_EXTREMES = 20


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quick eval for the Skywork reward model on task1.3 DPO pairs.")
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--adapter_dir", default=ADAPTER_DIR)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--max_eval_examples", type=int, default=MAX_EVAL_EXAMPLES)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--top_k_extremes", type=int, default=TOP_K_EXTREMES)
    return parser


def join_prompt_and_response(prompt: str, response: str) -> str:
    return f"{prompt.rstrip()}\n{response.strip()}"


def score_text(model, tokenizer, prompt: str, response: str, max_length: int) -> float:
    text = join_prompt_and_response(prompt, response)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(-1)
    return float(logits.item())


def short(text: str, limit: int) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def markdown_block(text: str) -> str:
    return f"```text\n{str(text).strip()}\n```"


def main() -> None:
    args = build_arg_parser().parse_args()
    has_cuda = torch.cuda.is_available()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"skywork_eval_validation_{args.max_eval_examples}_{timestamp}.md"

    print("[1/4] loading model...", flush=True)
    model_kwargs = {
        "trust_remote_code": True,
        "num_labels": 1,
        "attn_implementation": "flash_attention_2" if has_cuda else "eager",
    }
    if has_cuda:
        model_kwargs["torch_dtype"] = torch.bfloat16
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        **model_kwargs,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    if has_cuda:
        model = model.to("cuda")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/4] loading dataset...", flush=True)
    ds = load_from_disk(args.data_dir)
    eval_ds = ds["validation"] if isinstance(ds, DatasetDict) else ds
    total = min(args.max_eval_examples, len(eval_ds))

    print(f"[3/4] scoring {total} validation examples...", flush=True)
    wins = 0
    margins = []
    preview_rows: list[dict] = []
    all_rows: list[dict] = []
    by_source: dict[str, list[float]] = defaultdict(list)
    for i in range(total):
        ex = eval_ds[i]
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen_score = score_text(model, tokenizer, prompt, chosen, args.max_length)
        rejected_score = score_text(model, tokenizer, prompt, rejected, args.max_length)
        margin = chosen_score - rejected_score
        wins += int(margin > 0)
        margins.append(margin)
        pref_source = ex.get("preference_source") or "unknown"
        by_source[str(pref_source)].append(margin)
        all_rows.append(
            {
                "index": i,
                "ticker": ex.get("ticker"),
                "factor": ex.get("factor"),
                "return_label": ex.get("return_label"),
                "preference_source": pref_source,
                "chosen_score": chosen_score,
                "rejected_score": rejected_score,
                "margin": margin,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

        if i < args.num_samples:
            preview_rows.append(
                {
                    "index": i,
                    "ticker": ex.get("ticker"),
                    "factor": ex.get("factor"),
                    "return_label": ex.get("return_label"),
                    "preference_source": pref_source,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "margin": margin,
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                }
            )
            print("=" * 100)
            print(f"[CASE {i}] ticker={ex.get('ticker')} factor={ex.get('factor')} return={ex.get('return_label')}")
            print(f"chosen_score={chosen_score:.4f} rejected_score={rejected_score:.4f} margin={margin:.4f}")
            print("-" * 100)
            print("[PROMPT]")
            print(prompt[:2000])
            print("-" * 100)
            print("[CHOSEN]")
            print(chosen[:3000])
            print("-" * 100)
            print("[REJECTED]")
            print(rejected[:3000])
            print()

    avg_margin = sum(margins) / len(margins) if margins else 0.0
    accuracy = wins / len(margins) if margins else 0.0
    print("[4/4] summary", flush=True)
    print(f"pairs={len(margins)} pairwise_accuracy={accuracy:.4f} avg_margin={avg_margin:.4f}", flush=True)

    md_lines = [
        "# Skywork Reward Eval",
        "",
        f"- base_model: `{args.base_model}`",
        f"- adapter_dir: `{args.adapter_dir}`",
        f"- data_dir: `{args.data_dir}`",
        f"- evaluated_pairs: `{len(margins)}`",
        f"- preview_cases: `{len(preview_rows)}`",
        f"- pairwise_accuracy: `{accuracy:.4f}`",
        f"- avg_margin: `{avg_margin:.4f}`",
        f"- generated_at: `{timestamp}`",
        "",
    ]

    for row in preview_rows:
        md_lines.extend(
            [
                f"## Case {row['index']}",
                "",
                f"- ticker: `{row['ticker']}`",
                f"- factor: `{row['factor']}`",
                f"- return_label: `{row['return_label']}`",
                f"- preference_source: `{row['preference_source']}`",
                f"- chosen_score: `{row['chosen_score']:.4f}`",
                f"- rejected_score: `{row['rejected_score']:.4f}`",
                f"- margin: `{row['margin']:.4f}`",
                "",
                "### Prompt",
                "",
                markdown_block(short(row["prompt"], 4000)),
                "",
                "### Chosen",
                "",
                markdown_block(short(row["chosen"], 5000)),
                "",
                "### Rejected",
                "",
                markdown_block(short(row["rejected"], 5000)),
                "",
            ]
        )

    md_lines.extend(
        [
            "## Preference Source Stats",
            "",
            "| preference_source | count | pairwise_accuracy | avg_margin |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source, source_margins in sorted(by_source.items()):
        source_acc = sum(m > 0 for m in source_margins) / len(source_margins)
        source_avg = sum(source_margins) / len(source_margins)
        md_lines.append(
            f"| {source} | {len(source_margins)} | {source_acc:.4f} | {source_avg:.4f} |"
        )
    md_lines.append("")

    def append_case_section(title: str, rows: list[dict]) -> None:
        md_lines.extend([f"## {title}", ""])
        for row in rows:
            md_lines.extend(
                [
                    f"### Case {row['index']}",
                    "",
                    f"- ticker: `{row['ticker']}`",
                    f"- factor: `{row['factor']}`",
                    f"- return_label: `{row['return_label']}`",
                    f"- preference_source: `{row['preference_source']}`",
                    f"- chosen_score: `{row['chosen_score']:.4f}`",
                    f"- rejected_score: `{row['rejected_score']:.4f}`",
                    f"- margin: `{row['margin']:.4f}`",
                    "",
                    "#### Prompt",
                    "",
                    markdown_block(short(row["prompt"], 4000)),
                    "",
                    "#### Chosen",
                    "",
                    markdown_block(short(row["chosen"], 5000)),
                    "",
                    "#### Rejected",
                    "",
                    markdown_block(short(row["rejected"], 5000)),
                    "",
                ]
            )

    top_k = max(1, int(args.top_k_extremes))
    smallest_rows = sorted(all_rows, key=lambda row: row["margin"])[:top_k]
    largest_rows = sorted(all_rows, key=lambda row: row["margin"], reverse=True)[:top_k]
    append_case_section(f"Smallest Margins Top {len(smallest_rows)}", smallest_rows)
    append_case_section(f"Largest Margins Top {len(largest_rows)}", largest_rows)

    report_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"saved markdown report to {report_path}", flush=True)


if __name__ == "__main__":
    main()
