from __future__ import annotations

import traceback

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from unsloth import FastLanguageModel


MODEL_NAME = "Qwen/Qwen3.5-9B"
DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"


def print_header(title: str) -> None:
    print("\n" + "=" * 20 + f" {title} " + "=" * 20, flush=True)


def try_block(label: str, fn):
    print_header(label)
    try:
        fn()
    except Exception as exc:
        print(f"{label} FAILED: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()


def load_sample_messages():
    ds = load_from_disk(DATASET_PATH)["train"]
    sample = ds[0]
    print("sample keys:", sorted(sample.keys()), flush=True)
    print("sample factor:", sample.get("factor"), flush=True)
    print("sample date:", sample.get("report_date"), flush=True)
    return sample["messages"]


def main() -> None:
    messages = load_sample_messages()

    def unsloth_probe():
        model, tok_or_proc = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=4096,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        print("unsloth returned:", type(tok_or_proc), flush=True)
        print("model class:", type(model), flush=True)
        print("model config class:", model.config.__class__.__name__, flush=True)
        print("has vision_config:", hasattr(model.config, "vision_config"), flush=True)
        print("model_type:", getattr(model.config, "model_type", None), flush=True)

    def auto_tokenizer_probe():
        tok = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=False,
        )
        print("AutoTokenizer:", type(tok), flush=True)
        print("chat_template exists:", hasattr(tok, "chat_template"), flush=True)
        print("chat_template is not None:", getattr(tok, "chat_template", None) is not None, flush=True)
        print("eos_token:", getattr(tok, "eos_token", None), getattr(tok, "eos_token_id", None), flush=True)
        print("pad_token:", getattr(tok, "pad_token", None), getattr(tok, "pad_token_id", None), flush=True)
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            print("tokenizer template ok")
            print(text[:800], flush=True)
        except Exception as exc:
            print("tokenizer template failed:", repr(exc), flush=True)
        try:
            ids = tok("hello world", return_tensors="pt")
            print("tokenizer encode ok:", ids["input_ids"].shape, flush=True)
            print("decode:", tok.decode(ids["input_ids"][0]), flush=True)
        except Exception as exc:
            print("tokenizer encode/decode failed:", repr(exc), flush=True)

    def auto_processor_probe():
        proc = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
        print("AutoProcessor:", type(proc), flush=True)
        print("chat_template exists:", hasattr(proc, "chat_template"), flush=True)
        print("chat_template is not None:", getattr(proc, "chat_template", None) is not None, flush=True)
        try:
            out = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            print("processor template ok")
            print(out[:800] if isinstance(out, str) else out, flush=True)
        except Exception as exc:
            print("processor template failed:", repr(exc), flush=True)

    def transformers_forward_probe():
        tok = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_fast=False,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            print("using tokenizer chat template", flush=True)
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m['role'].capitalize()}: {m['content']}")
            text = "\n\n".join(parts)
            print("using manual text fallback", flush=True)
        batch = tok(text, return_tensors="pt", truncation=True, max_length=1024)
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        print("forward ok, logits shape:", tuple(out.logits.shape), flush=True)

    try_block("UNSLOTH_PROBE", unsloth_probe)
    try_block("AUTO_TOKENIZER_PROBE", auto_tokenizer_probe)
    try_block("AUTO_PROCESSOR_PROBE", auto_processor_probe)
    try_block("TRANSFORMERS_FORWARD_PROBE", transformers_forward_probe)


if __name__ == "__main__":
    main()
