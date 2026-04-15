import json
from pathlib import Path
from transformers import AutoTokenizer

if __name__ == "__main__":
    base_dir = Path(__file__).parent

    SYSTEM_MINER = (
        "You are a Financial Data Engineer. "
        "Extract original quotes from the provided source text that answer the specific questions in the schema. "
        "For each question, if an answer is found, provide the original_quote and a relevance_confidence from 0 to 1. "
        "If no answer is found, ignore the question entirely. "
        "Output only valid JSON."
    )

    print("加载 tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507-FP8")

    with open(base_dir / "sc.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)

    sys_tokens    = len(tokenizer(SYSTEM_MINER)["input_ids"])
    schema_tokens = len(tokenizer(schema_str)["input_ids"])
    total         = sys_tokens + schema_tokens

    print(f"system prompt : {sys_tokens} tokens")
    print(f"schema (full) : {schema_tokens} tokens")
    print(f"合计固定开销  : {total} tokens")
    print(f"262144 context 剩余可用: {262144 - total} tokens")