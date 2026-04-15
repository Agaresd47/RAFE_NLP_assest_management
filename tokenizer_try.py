from datasets import load_dataset
from transformers import AutoTokenizer
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np

def init_tokenizer():
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507-FP8")

def get_length_worker(text):
    tokens = tokenizer(text, return_tensors="pt")
    return tokens["input_ids"].shape[1]

if __name__ == "__main__":
    print("=" * 50)
    print("[1/4] 加载数据集...")
    ds = load_dataset("Vandy-NLPasset-26-G1/hf_miner_v2")
    texts = [example["messages"][0]["content"] for example in ds["train"]]
    print(f"      train 样本数: {len(texts)}")

    print("\n[2/4] 初始化 4 个 worker 进程...")
    with Pool(processes=4, initializer=init_tokenizer) as pool:
        print("      worker 就绪，开始计算 token 长度...\n")
        lengths = list(tqdm(
            pool.imap(get_length_worker, texts),
            total=len(texts),
            desc="[3/4] Tokenizing",
            unit="sample",
        ))

    print("\n[4/4] 统计结果")
    print("=" * 50)
    print(f"  样本总数 : {len(lengths)}")
    print(f"  最长     : {max(lengths)}")
    print(f"  最短     : {min(lengths)}")
    print(f"  平均     : {np.mean(lengths):.0f}")
    print(f"  中位数   : {np.median(lengths):.0f}")
    print(f"  90 分位  : {np.percentile(lengths, 90):.0f}")
    print(f"  95 分位  : {np.percentile(lengths, 95):.0f}")
    print(f"  99 分位  : {np.percentile(lengths, 99):.0f}")
    print("-" * 50)
    print(f"  超过 2K  : {sum(l > 2048  for l in lengths):>6} / {len(lengths)}")
    print(f"  超过 4K  : {sum(l > 4096  for l in lengths):>6} / {len(lengths)}")
    print(f"  超过 8K  : {sum(l > 8192  for l in lengths):>6} / {len(lengths)}")
    print(f"  超过 16K : {sum(l > 16384 for l in lengths):>6} / {len(lengths)}")
    print("=" * 50)