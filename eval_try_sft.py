import unsloth
import os
import json
import warnings
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
from train_common import load_lora_adapter_weights, resolve_latest_adapter_path, load_unsloth_model

# 压掉那条 transformers warning/logging 兼容问题
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_PATH = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5"
SPLIT = "validation"
MAX_SEQ_LEN = 32768
NUM_SAMPLES = 5
MAX_NEW_TOKENS = 4096
REPETITION_PENALTY = 1.02
#NO_REPEAT_NGRAM_SIZE = 0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


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
    # 推理时只喂 system + user，不喂 assistant GT
    prompt_messages = messages[:2]

    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return text


def safe_json_load(s):
    s = str(s or "").replace("<|im_end|>", "").strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def count_extractions(text: str) -> int:
    obj = safe_json_load(text)
    if not isinstance(obj, dict):
        return 0
    extractions = obj.get("extractions", [])
    return len(extractions) if isinstance(extractions, list) else 0


def main():
    print("[1/3] loading model...")
    model, _ = load_unsloth_model(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        lora_r=16,
        lora_alpha=16,
        target_modules=TARGET_MODULES,
    )
    adapter_path = resolve_latest_adapter_path(ADAPTER_PATH)
    if not adapter_path:
        raise FileNotFoundError(f"No adapter checkpoint found under: {ADAPTER_PATH}")
    loaded = load_lora_adapter_weights(model, adapter_path)
    print(f"[adapter] loaded={loaded} path={adapter_path}", flush=True)
    FastLanguageModel.for_inference(model)
    print(f"[model] using base={BASE_MODEL}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
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
    print("tokenizer.eos_token =", repr(tokenizer.eos_token), flush=True)
    print("tokenizer.pad_token =", repr(tokenizer.pad_token), flush=True)

    print("[2/3] loading dataset...")
    ds = load_from_disk(DATA_DIR)
    eval_ds = ds[SPLIT]

    print("[3/3] running inference...\n")

    for i in range(min(NUM_SAMPLES, len(eval_ds))):
        ex = eval_ds[i]
        messages = ex["messages"]

        prompt = make_prompt(messages, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                #repetition_penalty=REPETITION_PENALTY,
                #no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0][prompt_len:]
        pred_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        gt = messages[-1]["content"]
        gt_count = count_extractions(gt)
        pred_count = count_extractions(pred_text)

        print("=" * 100)
        print(
            f"[CASE {i}] ticker={ex.get('ticker')} form={ex.get('form')} "
            f"date={ex.get('report_date')} extraction_count={ex.get('extraction_count')}"
        )
        print("-" * 100)
        print("[USER INPUT]")
        print(messages[1]["content"][:200])
        print("-" * 100)
        print("[MODEL OUTPUT]")
        print(pred_text[:4000])
        print("-" * 100)
        print("[GROUND TRUTH]")
        print(gt[:3000])
        print("-" * 100)
        print(f"[COUNTS] pred_extractions={pred_count} gt_extractions={gt_count}")
        print()


if __name__ == "__main__":
    main()
