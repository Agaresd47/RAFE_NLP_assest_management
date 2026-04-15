import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import torch
import json
import os
from unsloth import FastLanguageModel
from peft import PeftModel
from datasets import load_dataset

os.environ["HF_DATASETS_OFFLINE"] = "1"

BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "/scratch/xla2767/models/nlp_combined_new/checkpoint-500"
MAX_SEQ_LEN = 32768

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

print("加载模型...", flush=True)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name       = BASE_MODEL,
    max_seq_length   = MAX_SEQ_LEN,
    dtype            = torch.bfloat16,
    load_in_4bit     = True,
    local_files_only = True,
)
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()
FastLanguageModel.for_inference(model)

def infer(messages, max_new_tokens=2048):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.8,
            do_sample=True,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

print("加载数据集...", flush=True)
miner_ds   = load_dataset("Vandy-NLPasset-26-G1/hf_miner_v2",
                          cache_dir="/projects/p32908/hf_cache")
auditor_ds = load_dataset("Vandy-NLPasset-26-G1/hf_auditor_v2",
                          cache_dir="/projects/p32908/hf_cache")

miner_test   = miner_ds["test"][13]
auditor_test = auditor_ds["test"].filter(
    lambda x: x.get("cot_visibility") == "no_cot"
)[0]

print("\n" + "="*60, flush=True)
print("TEST 1: Miner", flush=True)
print("="*60, flush=True)

miner_input = [
    {"role": "system", "content": SYSTEM_MINER},
    {"role": "user",   "content": miner_test["messages"][1]["content"]},
]
print("--- Ground Truth ---")
print(miner_test["messages"][2]["content"][:500])
print("\n--- Model Output ---")
print(infer(miner_input), flush=True)

print("\n" + "="*60, flush=True)
print("TEST 2: Auditor", flush=True)
print("="*60, flush=True)

auditor_input = [
    {"role": "system", "content": SYSTEM_AUDITOR},
    {"role": "user",   "content": auditor_test["messages"][1]["content"]},
]
print("--- Ground Truth ---")
print(auditor_test["messages"][2]["content"])
print("\n--- Model Output ---")
print(infer(auditor_input), flush=True)

print("\n推理测试完成", flush=True)