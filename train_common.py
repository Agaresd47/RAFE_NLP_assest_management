from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Optional, Tuple

import datasets
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from peft import set_peft_model_state_dict
from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoTokenizer
from unsloth import FastLanguageModel


datasets.disable_caching()


def configure_runtime() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def load_local_hf_dataset(dataset_path: str | Path):
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")
    return load_from_disk(str(path))


def get_text_tokenizer(tokenizer):
    if hasattr(tokenizer, "eos_token") or hasattr(tokenizer, "eos_token_id"):
        return tokenizer
    for attr in ("text_tokenizer", "tokenizer"):
        inner = getattr(tokenizer, attr, None)
        if inner is not None and (hasattr(inner, "eos_token") or hasattr(inner, "eos_token_id")):
            return inner
    return tokenizer


def configure_qwen_tokenizer_and_model(tokenizer, model) -> None:
    tokenizer.padding_side = "right"

    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        tokenizer.eos_token = "<|im_end|>"
    elif "<|endoftext|>" in vocab:
        tokenizer.eos_token = "<|endoftext|>"
    else:
        raise ValueError("Could not find a compatible eos token in tokenizer vocab.")

    tokenizer.pad_token = tokenizer.eos_token
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


def pick_train_eval_splits(
    dataset_obj,
    *,
    train_split: str = "train",
    eval_split: str = "validation",
    eval_ratio: float = 0.02,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    if isinstance(dataset_obj, DatasetDict):
        split_names = list(dataset_obj.keys())
        train_split_name = None
        if train_split in dataset_obj:
            train_split_name = train_split
            train_ds = dataset_obj[train_split]
        elif split_names:
            train_split_name = split_names[0]
            train_ds = dataset_obj[split_names[0]]
        else:
            raise ValueError("DatasetDict has no splits")

        for candidate in (eval_split, "validation", "test"):
            if candidate in dataset_obj and candidate != train_split_name and len(dataset_obj[candidate]) > 0:
                return train_ds, dataset_obj[candidate]

        if eval_ratio and len(train_ds) > 1:
            split = train_ds.train_test_split(test_size=eval_ratio, seed=seed)
            return split["train"], split["test"]
        return train_ds, None

    if isinstance(dataset_obj, Dataset):
        if eval_ratio and len(dataset_obj) > 1:
            split = dataset_obj.train_test_split(test_size=eval_ratio, seed=seed)
            return split["train"], split["test"]
        return dataset_obj, None

    raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)!r}")


def normalize_text_dataset(dataset: Dataset, tokenizer, *, max_seq_len: int, num_proc: int = 4) -> Dataset:
    text_tokenizer = get_text_tokenizer(tokenizer)

    missing_text = "text" not in dataset.column_names
    missing_messages = "messages" not in dataset.column_names
    if missing_text and missing_messages:
        raise ValueError("SFT dataset needs either a 'text' or 'messages' column")

    def to_text(example):
        if example.get("text"):
            text = str(example["text"])
        else:
            try:
                text = tokenizer.apply_chat_template(
                    example["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                parts: list[str] = []
                for message in example["messages"]:
                    role = str(message.get("role", "user")).strip().capitalize()
                    content = str(message.get("content", "")).strip()
                    parts.append(f"{role}: {content}")
                text = "\n\n".join(parts)
        return {"text": text}

    def add_length(example):
        if hasattr(text_tokenizer, "encode"):
            length = len(text_tokenizer.encode(example["text"], add_special_tokens=False))
        else:
            encoded = text_tokenizer(example["text"], add_special_tokens=False, truncation=False)
            length = len(encoded["input_ids"])
        return {"length": length}

    formatted = dataset.map(to_text, desc="format text", num_proc=num_proc)
    formatted = formatted.map(add_length, desc="compute length", num_proc=num_proc)
    formatted = formatted.filter(lambda x: x["length"] <= max_seq_len)
    return formatted


def normalize_dpo_dataset(dataset: Dataset, *, num_proc: int = 4) -> Dataset:
    required = {"prompt", "chosen", "rejected"}
    missing = sorted(required.difference(dataset.column_names))
    if missing:
        raise ValueError(
            "DPO dataset needs prompt/chosen/rejected columns; missing: "
            + ", ".join(missing)
        )

    def clean(example):
        return {
            "prompt": str(example["prompt"]).strip(),
            "chosen": str(example["chosen"]).strip(),
            "rejected": str(example["rejected"]).strip(),
        }

    cleaned = dataset.map(clean, desc="normalize dpo fields", num_proc=num_proc)
    cleaned = cleaned.filter(
        lambda x: bool(x["prompt"]) and bool(x["chosen"]) and bool(x["rejected"])
    )
    return cleaned


def load_unsloth_model(
    *,
    model_name: str,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    target_modules: list[str],
    lora_dropout: float = 0.0,
    dtype: torch.dtype = torch.bfloat16,
    load_in_4bit: bool = True,
):
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    text_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    configure_qwen_tokenizer_and_model(text_tokenizer, model)
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    configure_qwen_tokenizer_and_model(text_tokenizer, model)
    return model, text_tokenizer


def load_lora_adapter_weights(model, adapter_path: str | Path) -> bool:
    path = Path(adapter_path)
    if not path.exists():
        return False

    target_dtype = None
    for param in model.parameters():
        if torch.is_floating_point(param):
            target_dtype = param.dtype
            break

    def _cast_state_dict(state_dict):
        if target_dtype is None:
            return state_dict
        casted = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value) and torch.is_floating_point(value):
                casted[key] = value.to(dtype=target_dtype)
            else:
                casted[key] = value
        return casted

    safetensors_path = path / "adapter_model.safetensors"
    if safetensors_path.exists():
        state_dict = load_safetensors_file(str(safetensors_path))
        state_dict = _cast_state_dict(state_dict)
        set_peft_model_state_dict(model, state_dict)
        return True

    bin_path = path / "adapter_model.bin"
    if bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu")
        state_dict = _cast_state_dict(state_dict)
        set_peft_model_state_dict(model, state_dict)
        return True

    if hasattr(model, "load_adapter"):
        try:
            model.load_adapter(str(path), adapter_name="default")
            return True
        except Exception:
            pass

    return False


def resolve_latest_adapter_path(adapter_path: str | Path) -> str | None:
    path = Path(adapter_path)
    if not path.exists():
        return None

    latest_checkpoint = find_latest_checkpoint(path)
    if latest_checkpoint:
        latest_path = Path(latest_checkpoint)
        if (latest_path / "adapter_model.safetensors").exists() or (latest_path / "adapter_model.bin").exists():
            return str(latest_path)

    if (path / "adapter_model.safetensors").exists() or (path / "adapter_model.bin").exists():
        return str(path)
    return None


def trainer_init_kwargs(trainer_cls, tokenizer):
    params = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in params:
        return {"processing_class": tokenizer}
    if "tokenizer" in params:
        return {"tokenizer": tokenizer}
    return {}


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def find_latest_checkpoint(path: str | Path) -> Optional[str]:
    root = Path(path)
    if not root.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        candidates.append((step, child))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return str(candidates[-1][1])
