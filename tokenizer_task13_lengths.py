from __future__ import annotations

import argparse
import json
from statistics import mean, median

import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer


DEFAULT_MODEL_PATH = "Qwen/Qwen3.5-9B-Base"
DEFAULT_SFT_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
DEFAULT_DPO_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_dpo"
THINK_START = "<think>\n"
THINK_END = "\n</think>\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect token-length distributions for task13 datasets.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft_path", default=DEFAULT_SFT_PATH)
    parser.add_argument("--dpo_path", default=DEFAULT_DPO_PATH)
    parser.add_argument("--split", default="train")
    return parser


def summarize(name: str, lengths: list[int]) -> None:
    arr = np.asarray(lengths, dtype=np.int32)
    print(f"\n{name}")
    print("=" * len(name))
    print(f"count      : {len(arr)}")
    print(f"min        : {int(arr.min())}")
    print(f"max        : {int(arr.max())}")
    print(f"mean       : {mean(lengths):.1f}")
    print(f"median     : {median(lengths):.1f}")
    print(f"p90        : {np.percentile(arr, 90):.1f}")
    print(f"p95        : {np.percentile(arr, 95):.1f}")
    print(f"p99        : {np.percentile(arr, 99):.1f}")
    print(f"> 2k       : {int((arr > 2048).sum())}")
    print(f"> 4k       : {int((arr > 4096).sum())}")
    print(f"> 8k       : {int((arr > 8192).sum())}")


def safe_json_load(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def build_sft_assistant_target(example) -> str:
    assistant_raw = example["messages"][-1]["content"]
    obj = safe_json_load(assistant_raw)
    if obj is None:
        reasoning_chain = str(assistant_raw).strip()
        sentiment_label = "neutral"
        confidence_score = 0.5
    else:
        reasoning_chain = str(obj.get("reasoning_chain", "")).strip()
        sentiment_label = str(obj.get("sentiment_label", "neutral"))
        confidence_score = obj.get("confidence_score", 0.5)
        try:
            confidence_score = float(confidence_score)
        except Exception:
            confidence_score = 0.5

    final_json = {
        "sentiment_label": sentiment_label,
        "confidence_score": confidence_score,
    }
    return THINK_START + reasoning_chain + THINK_END + json.dumps(final_json, ensure_ascii=False)


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    sft = load_from_disk(args.sft_path)[args.split]
    dpo = load_from_disk(args.dpo_path)[args.split]

    sft_lengths = []
    sft_output_lengths = []
    for example in sft:
        output_text = build_sft_assistant_target(example)
        prompt_messages = example["messages"][:-1]
        full_messages = list(prompt_messages) + [{"role": "assistant", "content": output_text}]
        text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=True,
        )
        sft_lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        sft_output_lengths.append(len(tokenizer.encode(output_text, add_special_tokens=False)))

    dpo_prompt_lengths = []
    dpo_chosen_lengths = []
    dpo_rejected_lengths = []
    dpo_pair_lengths = []
    dpo_rejected_pair_lengths = []
    for example in dpo:
        prompt_len = len(tokenizer.encode(example["prompt"], add_special_tokens=False))
        chosen_len = len(tokenizer.encode(example["chosen"], add_special_tokens=False))
        rejected_len = len(tokenizer.encode(example["rejected"], add_special_tokens=False))
        chosen_pair_len = len(
            tokenizer.encode(
                f"{example['prompt'].rstrip()}\n{example['chosen'].strip()}",
                add_special_tokens=False,
            )
        )
        rejected_pair_len = len(
            tokenizer.encode(
                f"{example['prompt'].rstrip()}\n{example['rejected'].strip()}",
                add_special_tokens=False,
            )
        )
        dpo_prompt_lengths.append(prompt_len)
        dpo_chosen_lengths.append(chosen_len)
        dpo_rejected_lengths.append(rejected_len)
        dpo_pair_lengths.append(chosen_pair_len)
        dpo_rejected_pair_lengths.append(rejected_pair_len)

    summarize(f"SFT {args.split} lengths", sft_lengths)
    summarize(f"SFT {args.split} assistant output lengths", sft_output_lengths)
    summarize(f"DPO {args.split} prompt lengths", dpo_prompt_lengths)
    summarize(f"DPO {args.split} chosen lengths", dpo_chosen_lengths)
    summarize(f"DPO {args.split} rejected lengths", dpo_rejected_lengths)
    summarize(f"DPO {args.split} prompt+chosen lengths", dpo_pair_lengths)
    summarize(f"DPO {args.split} prompt+rejected lengths", dpo_rejected_pair_lengths)

    print("\nexamples")
    print("========")
    print(
        json.dumps(
            {
                "sft_first_len": sft_lengths[0] if sft_lengths else None,
                "sft_first_output_len": sft_output_lengths[0] if sft_output_lengths else None,
                "dpo_first_prompt_len": dpo_prompt_lengths[0] if dpo_prompt_lengths else None,
                "dpo_first_chosen_len": dpo_chosen_lengths[0] if dpo_chosen_lengths else None,
                "dpo_first_rejected_len": dpo_rejected_lengths[0] if dpo_rejected_lengths else None,
                "dpo_first_pair_len": dpo_pair_lengths[0] if dpo_pair_lengths else None,
                "dpo_first_rejected_pair_len": dpo_rejected_pair_lengths[0] if dpo_rejected_pair_lengths else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
