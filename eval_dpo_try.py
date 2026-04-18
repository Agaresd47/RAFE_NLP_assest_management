import unsloth
import argparse
import logging
import warnings
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from train_common import load_lora_adapter_weights, load_unsloth_model, resolve_latest_adapter_path


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/scratch/xla2767/hold2/models/cot_dpo_adapter"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
MAX_SEQ_LEN = 8192
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 768
SPLIT = "validation"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=BASE_MODEL)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    return parser


def resolve_model_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.exists():
        parent = path.parent
        if path.name.startswith("checkpoint-") and parent.exists():
            checkpoints = sorted(
                [p for p in parent.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                key=lambda p: int(p.name.split("-", 1)[1]),
            )
            if checkpoints:
                return str(checkpoints[-1])
        raise FileNotFoundError(f"Model path does not exist: {path}")

    if path.is_dir() and (path / "adapter_config.json").exists() and not (path / "config.json").exists():
        checkpoints = sorted(
            [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
            key=lambda p: int(p.name.split("-", 1)[1]),
        )
        if checkpoints:
            return str(checkpoints[-1])
    return str(path)


def make_prompt(messages, tokenizer):
    prompt_messages = messages[:2]
    return tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def main():
    args = build_arg_parser().parse_args()
    adapter_path = resolve_latest_adapter_path(args.adapter_path)
    if not adapter_path:
        raise FileNotFoundError(f"Could not find a DPO adapter under: {args.adapter_path}")

    print("[1/3] loading model...")
    model, _ = load_unsloth_model(
        model_name=args.base_model,
        max_seq_length=args.max_seq_len,
        lora_r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    loaded = load_lora_adapter_weights(model, adapter_path)
    if not loaded:
        raise RuntimeError(f"Failed to load DPO adapter weights from: {adapter_path}")
    FastLanguageModel.for_inference(model)
    print(f"[model] base={args.base_model}", flush=True)
    print(f"[adapter] using {adapter_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "left"

    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    tokenizer.pad_token = tokenizer.eos_token

    print("[2/3] loading dataset...")
    ds = load_from_disk(args.data_dir)
    eval_ds = ds[args.split]

    print("[3/3] running inference...\n")

    for i in range(min(args.num_samples, len(eval_ds))):
        ex = eval_ds[i]
        messages = ex["messages"]

        prompt = make_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_len:]
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=False)
        gt = messages[-1]["content"]

        print("=" * 100)
        print(f"[CASE {i}] ticker={ex.get('ticker')} factor={ex.get('factor')}")
        print("-" * 100)
        print("[USER INPUT]")
        print(messages[1]["content"][:2000])
        print("-" * 100)
        print("[MODEL OUTPUT]")
        print(pred_text[:4000])
        print("-" * 100)
        print("[GROUND TRUTH]")
        print(gt[:3000])
        print()


if __name__ == "__main__":
    main()
