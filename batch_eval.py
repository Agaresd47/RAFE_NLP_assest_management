import json
import os
import re
import torch
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
from tqdm import tqdm

# ── 配置 ──────────────────────────────────────────────
BASE_MODEL  = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = "/scratch/xla2767/models/nlp_combined"
MAX_SEQ_LEN = 32768
BATCH_SIZE  = 4
OUTPUT_CSV  = "/projects/p32908/backtest_signals.csv"
os.environ["HF_DATASETS_OFFLINE"] = "1"

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

# ── 加载模型 ───────────────────────────────────────────
print("加载模型...")
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

# ── 工具函数 ───────────────────────────────────────────
def parse_output(raw):
    clean = raw.replace("<tool_call>", "").replace("</tool_call>", "").strip()
    try:
        return json.loads(clean)
    except:
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return {}

def infer_batch(batch_messages, max_new_tokens=512):
    texts = [
        tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        for msgs in batch_messages
    ]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=MAX_SEQ_LEN
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return [
        tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        for out in outputs
    ]

def build_user_prompt_auditor_single(audit, ticker, form, report_date):
    f_name = audit.get("factor", "unknown")
    lines = [
        f"Task: Auditor (1.3)",
        f"Ticker: {ticker} | Filing: {form} | Date: {report_date}",
        f"Factor: {f_name}",
        "",
        "[Current Evidence]",
    ]
    for e in audit.get("evidence_used", []):
        lines.append(
            f"  q_key : {e.get('question_key', '—')}\n"
            f"  quote : {e.get('original_quote', '—')}\n"
            f"  conf  : {e.get('relevance_confidence', '—')}"
        )
    hist = audit.get("historical_context", [])
    if hist:
        lines.append("[Historical Context]")
        for h in hist:
            lines.append(
                f"  [{h.get('report_date', '—')} {h.get('filing', '—')}] "
                f"{h.get('fact', '—')}  (conf: {h.get('relevance_confidence', '—')})"
            )
    lines.append("\nOutput sentiment_label and confidence_score for this factor.")
    return "\n".join(lines)

# ── 加载数据集 ─────────────────────────────────────────
print("加载数据集...")
miner_ds   = load_dataset("Vandy-NLPasset-26-G1/hf_miner_v2",
                          cache_dir="/projects/p32908/hf_cache")
auditor_ds = load_dataset("Vandy-NLPasset-26-G1/hf_auditor_v2",
                          cache_dir="/projects/p32908/hf_cache")

miner_test   = miner_ds["test"]
auditor_test = auditor_ds["test"].filter(
    lambda x: x.get("cot_visibility") == "no_cot"
)

# 建立 historical context 索引
# key: (ticker, filing, date, factor) → historical_context list
print("建立 historical context 索引...")
historical_index = defaultdict(list)
for x in auditor_ds["test"]:
    if x.get("cot_visibility") != "no_cot":
        continue
    ticker = x.get("ticker", "")
    form   = x.get("form", "")
    date   = x.get("report_date", "")
    factor = x.get("factor", "")
    # 从 user content 里提取 historical context
    user_content = x["messages"][1]["content"]
    hist_match = re.search(r'\[Historical Context\](.*?)(?:\nOutput|\Z)', 
                           user_content, re.DOTALL)
    if hist_match:
        historical_index[(ticker, form, date, factor)] = hist_match.group(1).strip()

# ── Step 1: 跑 Miner ──────────────────────────────────
print("\n=== Step 1: Miner 推理 ===")
miner_results = []  # list of (ticker, form, date, extractions)

for i in tqdm(range(0, len(miner_test), BATCH_SIZE)):
    if i >= 4 * BATCH_SIZE:  # 只跑30个batch
        break
    batch = miner_test.select(range(i, min(i+BATCH_SIZE, len(miner_test))))
    msgs = [
        [{"role": "system", "content": SYSTEM_MINER},
         {"role": "user",   "content": x["messages"][1]["content"]}]
        for x in batch
    ]
    outputs = infer_batch(msgs, max_new_tokens=2048)

    for x, out in zip(batch, outputs):
        print("RAW OUTPUT:", out)
        meta = json.loads(x["messages"][2]["content"]).get("metadata", {})
        ticker      = meta.get("ticker", "")
        form        = meta.get("filing", "")
        report_date = meta.get("report_date", "")

        pred = parse_output(out)
        extractions = pred.get("extractions", [])
        miner_results.append((ticker, form, report_date, extractions))

print(f"Miner 完成，共 {len(miner_results)} 条")

# ── Step 2: 按 factor 分组，构建 Auditor input ─────────
print("\n=== Step 2: 构建 Auditor 输入 ===")
auditor_inputs = []  # list of (ticker, form, date, factor, user_prompt)

for ticker, form, report_date, extractions in miner_results:
    # 按 factor 分组
    factor_groups = defaultdict(list)
    for e in extractions:
        factor = e.get("factor", "unknown")
        factor_groups[factor].append({
            "question_key":        e.get("question_key", "—"),
            "original_quote":      e.get("original_quote", "—"),
            "relevance_confidence": e.get("relevance_confidence", 0),
        })

    for factor, evidence_list in factor_groups.items():
        # 取 historical context
        hist_raw = historical_index.get((ticker, form, report_date, factor), "")

        audit = {
            "factor":           factor,
            "evidence_used":    evidence_list,
            "historical_context": [],  # 用原始文本塞进去
        }

        user_prompt = build_user_prompt_auditor_single(
            audit, ticker, form, report_date
        )

        # 如果有 historical，直接拼接到 prompt 里
        if hist_raw:
            user_prompt = user_prompt.replace(
                "\nOutput sentiment_label and confidence_score for this factor.",
                f"\n[Historical Context]\n{hist_raw}\n\nOutput sentiment_label and confidence_score for this factor."
            )

        auditor_inputs.append((ticker, form, report_date, factor, user_prompt))

print(f"共 {len(auditor_inputs)} 个 factor 需要 Auditor 推理")

# ── Step 3: 跑 Auditor ────────────────────────────────
print("\n=== Step 3: Auditor 推理 ===")
rows = []

for i in tqdm(range(0, len(auditor_inputs), BATCH_SIZE)):
    batch = auditor_inputs[i:i+BATCH_SIZE]
    msgs = [
        [{"role": "system", "content": SYSTEM_AUDITOR},
         {"role": "user",   "content": item[4]}]
        for item in batch
    ]
    outputs = infer_batch(msgs, max_new_tokens=256)

    for (ticker, form, date, factor, _), out in zip(batch, outputs):
        pred = parse_output(out)
        rows.append({
            "Date":       date,
            "Symbol":     ticker,
            "Form":       form,
            "Factor":     factor,
            "Signal":     pred.get("sentiment", ""),
            "Confidence": pred.get("confidence", ""),
        })

# ── 保存 CSV ───────────────────────────────────────────
df = pd.DataFrame(rows)
df = df.sort_values(["Symbol", "Date"]).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n保存到 {OUTPUT_CSV}，共 {len(df)} 行")
print(df.head(10))