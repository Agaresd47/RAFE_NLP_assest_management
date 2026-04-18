import unsloth
import argparse
import logging
import warnings

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer


warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

MODEL_PATH = "/scratch/xla2767/hold2/models/qwen3_8b_thinking_sft_merged_v2"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
MAX_SEQ_LEN = 8192
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 768
SPLIT = "validation"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    return parser


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

    print("[1/3] loading merged model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[model] using {args.model_path}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len).to(
            model.device
        )

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
