import unsloth
from unsloth import FastLanguageModel

import os
import json
from pathlib import Path

import torch
import datasets
from datasets import load_from_disk
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoTokenizer, EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from peft import set_peft_model_state_dict

datasets.disable_caching()

# =========================
# 写死配置
# =========================
MODEL_PATH = "Qwen/Qwen3-8B"
DATA_DIR = "/scratch/xla2767/hold2/data/nlp/hf_cot_sft"
OUTPUT_DIR = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_thinking_sft_out"

MAX_SEQ_LEN = 4096
NUM_EPOCHS = 3.0
LEARNING_RATE = 2e-4
SAVE_STEPS = 50

THINK_START = "<think>\n"
THINK_END = "\n</think>\n"
AUTO_CONTINUE_MODE = "warm_start_latest"


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    root = Path(output_dir)
    if not root.exists():
        return None

    checkpoints: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((step, child))

    if not checkpoints:
        return None
    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def load_adapter_weights(model, adapter_dir: str | Path) -> bool:
    path = Path(adapter_dir)
    if not path.exists():
        return False

    safetensors_path = path / "adapter_model.safetensors"
    if safetensors_path.exists():
        state_dict = load_safetensors_file(str(safetensors_path))
        set_peft_model_state_dict(model, state_dict)
        return True

    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
        set_peft_model_state_dict(model, state_dict)
        return True

    return False


def safe_json_load(s):
    try:
        return json.loads(s)
    except Exception:
        return None


def build_assistant_target(example):
    """
    把最后一条 assistant JSON 转成:
    <think>
    reasoning_chain
    </think>
    {"sentiment_label": "...", "confidence_score": ...}
    """
    msgs = example["messages"]
    assistant_raw = msgs[-1]["content"]

    obj = safe_json_load(assistant_raw)
    if obj is None:
        reasoning_chain = str(assistant_raw).strip()
        sentiment_label = "neutral"
        confidence_score = 0.5
    else:
        reasoning_chain = str(obj.get("reasoning_chain", "")).strip()
        sentiment_label = str(obj.get("sentiment_label", "neutral"))
        confidence_score = obj.get("confidence_score", 0.5)
        try:
            confidence_score = float(confidence_score)
        except Exception:
            confidence_score = 0.5

    final_json = {
        "sentiment_label": sentiment_label,
        "confidence_score": confidence_score,
    }

    assistant_text = (
        THINK_START
        + reasoning_chain
        + THINK_END
        + json.dumps(final_json, ensure_ascii=False)
    )
    return assistant_text


def build_text(example, tokenizer):
    msgs = example["messages"]

    new_messages = []
    for m in msgs[:-1]:
        new_messages.append(
            {
                "role": m["role"],
                "content": m["content"],
            }
        )

    assistant_text = build_assistant_target(example)
    new_messages.append(
        {
            "role": "assistant",
            "content": assistant_text,
        }
    )

    text = tokenizer.apply_chat_template(
        new_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    return {"text": text}


def has_messages(example):
    return "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) >= 2


def add_length(example, tokenizer):
    ids = tokenizer.encode(example["text"], add_special_tokens=False)
    return {"length": len(ids)}


def get_splits(ds):
    if isinstance(ds, datasets.DatasetDict):
        if "train" not in ds:
            raise ValueError("本地数据集中没有 train split")
        train_ds = ds["train"]

        if "validation" in ds:
            eval_ds = ds["validation"]
        elif "eval" in ds:
            eval_ds = ds["eval"]
        elif "test" in ds:
            eval_ds = ds["test"]
        else:
            split_ds = train_ds.train_test_split(test_size=0.02, seed=42)
            train_ds = split_ds["train"]
            eval_ds = split_ds["test"]

        return train_ds, eval_ds

    elif isinstance(ds, datasets.Dataset):
        split_ds = ds.train_test_split(test_size=0.02, seed=42)
        return split_ds["train"], split_ds["test"]

    else:
        raise ValueError(f"不支持的数据集类型: {type(ds)}")


def main():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[GPU] {gpu_name}", flush=True)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("[1/5] 加载模型 (unsloth)...", flush=True)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LEN,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    print("[1.5/5] 加载 tokenizer (HF 原版)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "right"

    print("before fix:")
    print("tokenizer class =", type(tokenizer))
    print("eos_token =", repr(tokenizer.eos_token), "eos_token_id =", tokenizer.eos_token_id)
    print("pad_token =", repr(tokenizer.pad_token), "pad_token_id =", tokenizer.pad_token_id)
    print("bos_token =", repr(tokenizer.bos_token), "bos_token_id =", tokenizer.bos_token_id)

    vocab = tokenizer.get_vocab()

    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    else:
        raise ValueError("找不到可用 eos token")

    tokenizer.pad_token = tokenizer.eos_token

    # 关键：把 init_kwargs 和 special token map 一起改掉
    tokenizer.init_kwargs["eos_token"] = tokenizer.eos_token
    tokenizer.init_kwargs["pad_token"] = tokenizer.pad_token

    if hasattr(tokenizer, "_special_tokens_map"):
        tokenizer._special_tokens_map["eos_token"] = tokenizer.eos_token
        tokenizer._special_tokens_map["pad_token"] = tokenizer.pad_token

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print("after fix:")
    print("eos_token =", repr(tokenizer.eos_token), "eos_token_id =", tokenizer.eos_token_id)
    print("pad_token =", repr(tokenizer.pad_token), "pad_token_id =", tokenizer.pad_token_id)
    print("special_tokens_map =", tokenizer.special_tokens_map)
    print("init_kwargs =", tokenizer.init_kwargs)
    print("final eos in vocab =", tokenizer.eos_token in tokenizer.get_vocab())

    print("[2/5] 注入 LoRA...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    latest_checkpoint = find_latest_checkpoint(OUTPUT_DIR)
    if AUTO_CONTINUE_MODE == "warm_start_latest":
        if latest_checkpoint and load_adapter_weights(model, latest_checkpoint):
            print(f"[warm_start] 已加载最新 checkpoint 权重: {latest_checkpoint}", flush=True)
            print("[warm_start] 将使用新的 trainer state 继续训练，可安全修改 batch 配置", flush=True)
        else:
            print("[warm_start] 没找到可加载的旧 checkpoint，按冷启动训练", flush=True)
    else:
        if latest_checkpoint:
            print(f"[resume] 将严格恢复 trainer state: {latest_checkpoint}", flush=True)
        else:
            print("[resume] 没找到 checkpoint，从头开始训练", flush=True)

    print("[3/5] 加载本地数据集...", flush=True)
    ds = load_from_disk(DATA_DIR)
    print(ds, flush=True)

    train_ds, eval_ds = get_splits(ds)
    print(f"raw train={len(train_ds)} eval={len(eval_ds)}", flush=True)

    train_ds = train_ds.filter(has_messages)
    eval_ds = eval_ds.filter(has_messages)
    print(f"after has_messages train={len(train_ds)} eval={len(eval_ds)}", flush=True)

    if len(train_ds) == 0:
        raise ValueError("train_ds 为空，请检查数据结构是否包含 messages 字段")

    print("[样本预览] 原始第一条:", flush=True)
    print(train_ds[0], flush=True)

    print("[4/5] 格式化数据集...", flush=True)
    train_ds = train_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="Formatting train",
        keep_in_memory=True,
    )
    eval_ds = eval_ds.map(
        lambda x: build_text(x, tokenizer),
        desc="Formatting eval",
        keep_in_memory=True,
    )

    print("[样本预览] 格式化后第一条 text 前 2000 字符:", flush=True)
    print(train_ds[0]["text"][:2000], flush=True)

    train_ds = train_ds.map(
        lambda x: add_length(x, tokenizer),
        desc="Length train",
        num_proc=4,
    )
    eval_ds = eval_ds.map(
        lambda x: add_length(x, tokenizer),
        desc="Length eval",
        num_proc=4,
    )

    before_train = len(train_ds)
    before_eval = len(eval_ds)

    train_ds = train_ds.filter(lambda x: x["length"] <= MAX_SEQ_LEN)
    eval_ds = eval_ds.filter(lambda x: x["length"] <= MAX_SEQ_LEN)

    print(f"after length filter train: {before_train} -> {len(train_ds)}", flush=True)
    print(f"after length filter eval:  {before_eval} -> {len(eval_ds)}", flush=True)

    if len(train_ds) == 0:
        raise ValueError("长度过滤后 train_ds 为空。可以先把 MAX_SEQ_LEN 调大。")

    print("[5/5] 初始化 trainer...", flush=True)
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=True,  
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        gradient_accumulation_steps=6,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_steps=20,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=8,
        dataset_num_proc=8,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
    )

    print("tokenizer.eos_token =", repr(tokenizer.eos_token))
    print("tokenizer.pad_token =", repr(tokenizer.pad_token))
    print("tokenizer.special_tokens_map =", tokenizer.special_tokens_map)
    print("tokenizer.init_kwargs =", tokenizer.init_kwargs)
    print("model.config.eos_token_id =", getattr(model.config, "eos_token_id", None))
    print("model.config.pad_token_id =", getattr(model.config, "pad_token_id", None))

    if hasattr(model, "generation_config") and model.generation_config is not None:
        print("generation_config.eos_token_id =", getattr(model.generation_config, "eos_token_id", None))
        print("generation_config.pad_token_id =", getattr(model.generation_config, "pad_token_id", None))

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if AUTO_CONTINUE_MODE == "warm_start_latest":
        trainer.train()
    else:
        trainer.train(resume_from_checkpoint=latest_checkpoint)

    trainer.model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"模型已保存到 {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
