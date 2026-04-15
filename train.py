import unsloth
from unsloth import FastLanguageModel
import argparse
import os
from pathlib import Path
import json 

import torch
from datasets import concatenate_datasets, load_dataset
from trl import SFTTrainer, SFTConfig
import datasets
from transformers import EarlyStoppingCallback
datasets.disable_caching()
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
MAX_SEQ_LEN = 32768        


def build_text(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

def is_good_sample(example):
    try:
        gt = json.loads(example["messages"][2]["content"])
        extractions = gt.get("extractions", [])
        return len(extractions) >= 2
    except:
        return False
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",    default="Qwen/Qwen3-4B-Instruct-2507")  # 去掉 FP8
    parser.add_argument("--miner_path",    default="/projects/p32908/hf_cache/datasets/Vandy-NLPasset-26-G1___hf_miner_v2")
    parser.add_argument("--auditor_path",  default="/projects/p32908/hf_cache/datasets/Vandy-NLPasset-26-G1___hf_auditor_v2")
    parser.add_argument("--output_dir", default="/scratch/xla2767/models/nlp_combined_new")
    parser.add_argument("--num_epochs",    type=float, default=20.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--save_steps",    type=int,   default=5)
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # --- 用 unsloth 加载模型 + tokenizer ---
    print("[1/4] 加载模型 (unsloth)...", flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.model_path,
        max_seq_length = MAX_SEQ_LEN,       # 覆盖你的 110K 数据
        dtype          = torch.bfloat16,
        load_in_4bit   = True,         # QLoRA，显存大幅减少
    )
    tokenizer.padding_side = "right"

    # --- unsloth LoRA ---
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        lora_alpha     = 16,
        lora_dropout   = 0.0,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],  # 加上 MLP 层效果更好
        bias           = "none",
        use_gradient_checkpointing = "unsloth",  # unsloth 专属，比原版省更多显存
        random_state   = 42,
    )
    model.print_trainable_parameters()

    # --- 数据集 ---
    print("[2/4] 加载数据集...", flush=True)
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    miner_ds   = load_dataset("Vandy-NLPasset-26-G1/hf_miner_v2",
                          cache_dir="/projects/p32908/hf_cache")
    auditor_ds = load_dataset("Vandy-NLPasset-26-G1/hf_auditor_v2",
                            cache_dir="/projects/p32908/hf_cache")

    miner_train   = miner_ds["train"].filter(is_good_sample)
    auditor_train = auditor_ds["train"].filter(
        lambda x: x.get("cot_visibility") == "no_cot"
    )

    train_ds = concatenate_datasets([miner_train, auditor_train]).shuffle(seed=42)
    val_ds   = concatenate_datasets([
        miner_ds["validation"].filter(is_good_sample),   # ← 改这里
        auditor_ds["validation"].filter(lambda x: x.get("cot_visibility") == "no_cot"),
    ])
    print(f"train={len(train_ds)}  val={len(val_ds)}", flush=True)

    print("[3/4] 格式化数据集...", flush=True)
    train_ds = train_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="Formatting train",
        keep_in_memory=True,
    )
    val_ds = val_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="Formatting val",
        keep_in_memory=True,
    )

    # ↓ 加这一块，格式化之后立即过滤
    def add_length(example):
        return {"length": len(tokenizer.encode(example["text"], add_special_tokens=False))}

    train_ds = train_ds.map(add_length, desc="计算train长度", num_proc=4)
    val_ds   = val_ds.map(add_length,   desc="计算val长度",   num_proc=4)

    before_train = len(train_ds)
    before_val   = len(val_ds)
    train_ds = train_ds.filter(lambda x: x["length"] <= MAX_SEQ_LEN)
    val_ds   = val_ds.filter(lambda x: x["length"] <= MAX_SEQ_LEN)
    print(f"过滤后 train: {before_train} → {len(train_ds)}", flush=True)
    print(f"过滤后 val:   {before_val}   → {len(val_ds)}",   flush=True)

    cfg = SFTConfig(
        output_dir                  = args.output_dir,
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LEN,   
        packing                     = True,
        num_train_epochs            = args.num_epochs,
        learning_rate               = args.learning_rate,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 16,
        bf16                        = True,
        fp16                        = False,
        optim                       = "adamw_8bit",
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        weight_decay                = 0.01,
        logging_steps               = 5,
        eval_strategy               = "steps",
        eval_steps                  = 5,
        save_steps                  = args.save_steps,
        save_total_limit            = 20,
        report_to                   = "none",
        dataloader_num_workers      = 2,
        dataset_num_proc            = 4,
        packing_strategy            = "bfd",   
        load_best_model_at_end = True,
        metric_for_best_model  = "eval_loss",
        greater_is_better      = False,
    )

    trainer = SFTTrainer(
        model             = model,
        processing_class  = tokenizer,   # 新版参数名，不是 tokenizer
        train_dataset     = train_ds,
        eval_dataset      = val_ds,
        args              = cfg,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    #trainer.train(resume_from_checkpoint="/scratch/xla2767/models/nlp_combined/checkpoint-220")
    trainer.train(resume_from_checkpoint=True)
    #trainer.train()

    # --- 保存 LoRA adapter ---
    trainer.model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"模型已保存到 {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()