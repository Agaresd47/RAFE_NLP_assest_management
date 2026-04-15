import json
import os
import re
import requests
import time
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import asyncio
import aiohttp

'''
python -m sglang.launch_server \
    --model-path /scratch/xla2767/models/try_delete_latter \
    --port 30000 \
    --dtype bfloat16 \
    --mem-fraction-static 0.85 &

python qbatch.py
'''

# ── 配置 ──────────────────────────────────────────────
MODEL_PATH  = "/scratch/xla2767/models/try_delete_latter"
SGLANG_URL  = "http://localhost:30000"
MAX_SEQ_LEN = 32768
BATCH_SIZE  = 32
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
print("启动")
# ── 加载 tokenizer（只用来格式化 prompt）──────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ── sglang 工具函数 ────────────────────────────────────
def wait_for_server():
    print("等待 sglang server 启动...")
    for _ in range(60):
        try:
            requests.get(f"{SGLANG_URL}/health", timeout=2)
            print("Server ready!")
            return
        except:
            time.sleep(5)
    raise RuntimeError("Server 启动超时")

async def infer_single(session, msgs, max_new_tokens):
    async with session.post(
        f"{SGLANG_URL}/v1/chat/completions",
        json={
            "model": "default",
            "messages": msgs,
            "max_tokens": max_new_tokens,
            "temperature": 0.1,
            "top_p": 0.9,
        }
    ) as resp:
        data = await resp.json()
        return data["choices"][0]["message"]["content"]

async def infer_batch_async(batch_messages, max_new_tokens=512):
    async with aiohttp.ClientSession() as session:
        tasks = [infer_single(session, msgs, max_new_tokens) for msgs in batch_messages]
        return await asyncio.gather(*tasks)

def infer_batch(batch_messages, max_new_tokens=512):
    return asyncio.run(infer_batch_async(batch_messages, max_new_tokens))

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

# ── 等待 server ────────────────────────────────────────

wait_for_server()

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

print("建立 historical context 索引...")
historical_index = defaultdict(list)
for x in auditor_ds["test"]:
    if x.get("cot_visibility") != "no_cot":
        continue
    ticker = x.get("ticker", "")
    form   = x.get("form", "")
    date   = x.get("report_date", "")
    factor = x.get("factor", "")
    user_content = x["messages"][1]["content"]
    hist_match = re.search(r'\[Historical Context\](.*?)(?:\nOutput|\Z)',
                           user_content, re.DOTALL)
    if hist_match:
        historical_index[(ticker, form, date, factor)] = hist_match.group(1).strip()

# ── Step 1: Miner ──────────────────────────────────────
print("\n=== Step 1: Miner 推理 ===")
miner_results = []
print("Only 30 batch for test !!!!!!!!!!!")
for i in tqdm(range(0, len(miner_test), BATCH_SIZE)):
    batch = miner_test.select(range(i, min(i+BATCH_SIZE, len(miner_test))))
    #'''
    if i >= 2 * BATCH_SIZE:  # 只跑30个batch
        break
    #'''
    msgs = [
        [{"role": "system", "content": SYSTEM_MINER},
         {"role": "user",   "content": x["messages"][1]["content"]}]
        for x in batch
    ]
    outputs = infer_batch(msgs, max_new_tokens=2048)

    for x, out in zip(batch, outputs):
        print("RAW OUTPUT:", out[:100]) 
        meta = json.loads(x["messages"][2]["content"]).get("metadata", {})
        try: 
            miner_results.append((
                meta.get("ticker", ""),
                meta.get("filing", ""),
                meta.get("report_date", ""),
                parse_output(out).get("extractions", [])
            ))
        except:
            miner_results.append(("unknown", "unknown", "unknown", []))

print(f"Miner 完成，共 {len(miner_results)} 条")

# ── Step 2: 构建 Auditor 输入 ──────────────────────────
print("\n=== Step 2: 构建 Auditor 输入 ===")
auditor_inputs = []

for ticker, form, report_date, extractions in miner_results:
    factor_groups = defaultdict(list)
    for e in extractions:
        factor = e.get("factor", "unknown")
        factor_groups[factor].append({
            "question_key":         e.get("question_key", "—"),
            "original_quote":       e.get("original_quote", "—"),
            "relevance_confidence": e.get("relevance_confidence", 0),
        })

    for factor, evidence_list in factor_groups.items():
        hist_raw = historical_index.get((ticker, form, report_date, factor), "")
        audit = {"factor": factor, "evidence_used": evidence_list, "historical_context": []}
        user_prompt = build_user_prompt_auditor_single(audit, ticker, form, report_date)
        if hist_raw:
            user_prompt = user_prompt.replace(
                "\nOutput sentiment_label and confidence_score for this factor.",
                f"\n[Historical Context]\n{hist_raw}\n\nOutput sentiment_label and confidence_score for this factor."
            )
        auditor_inputs.append((ticker, form, report_date, factor, user_prompt))

print(f"共 {len(auditor_inputs)} 个 factor 需要 Auditor 推理")

# ── Step 3: Auditor ────────────────────────────────────
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