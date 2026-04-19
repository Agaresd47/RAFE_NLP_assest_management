import argparse
import logging
import warnings
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_common import load_lora_adapter_weights, load_unsloth_model, resolve_latest_adapter_path


warnings.filterwarnings(
    "ignore",
    message=r".*attention mask API under `transformers\.modeling_attn_mask_utils`.*",
    category=FutureWarning,
)

_ORIGINAL_LOGGER_WARNING = logging.Logger.warning


def _patched_logger_warning(self, msg, *args, **kwargs):
    if (
        args == (FutureWarning,)
        and isinstance(msg, str)
        and "attention mask API under `transformers.modeling_attn_mask_utils`" in msg
    ):
        args = ()
    return _ORIGINAL_LOGGER_WARNING(self, msg, *args, **kwargs)


logging.Logger.warning = _patched_logger_warning


DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ADAPTER_PATH = "/scratch/xla2767/hold2/models/cot_grpo_adapter_v2"
DEFAULT_MERGED_OUTPUT = "/scratch/xla2767/hold2/models/qwen3_8b_thinking_grpo_merged_v1"


def build_parser():
    parser = argparse.ArgumentParser(description="Merge a QLoRA adapter into a base model.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model path or HF cache directory.")
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help="Adapter safetensors file or a directory containing adapter_model.safetensors.",
    )
    parser.add_argument("--merged-output", default=DEFAULT_MERGED_OUTPUT, help="Directory for the merged model.")
    parser.add_argument(
        "--verify-model-path",
        default=None,
        help="Directory to reload for the smoke-test verification. Defaults to --merged-output.",
    )
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-merge smoke-test reload/generate step.",
    )
    return parser


def resolve_adapter_path(adapter_path):
    path = Path(adapter_path)
    if path.is_dir():
        candidate = path / "adapter_model.safetensors"
        if candidate.exists():
            return str(candidate)
    return str(path)


def main():
    args = build_parser().parse_args()
    target_modules = [name.strip() for name in args.target_modules.split(",") if name.strip()]
    merged_output = Path(args.merged_output)
    verify_model_path = Path(args.verify_model_path) if args.verify_model_path else merged_output
    resolved_adapter_path = resolve_latest_adapter_path(args.adapter_path) or resolve_adapter_path(args.adapter_path)

    model, tokenizer = load_unsloth_model(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )

    loaded = load_lora_adapter_weights(model, resolved_adapter_path)
    if not loaded:
        raise FileNotFoundError(f"Could not load adapter weights from: {resolved_adapter_path}")

    merged_output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_merged(
        str(merged_output),
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merge complete: {merged_output}")
    print(f"Adapter source: {resolved_adapter_path}", flush=True)

    if args.skip_verify:
        print("Skipping verification step.", flush=True)
        return

    test_model = AutoModelForCausalLM.from_pretrained(
        str(verify_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    test_tokenizer = AutoTokenizer.from_pretrained(str(verify_model_path))
    test_model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 1+1?"},
    ]
    text = test_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = test_tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = test_model.generate(**inputs, max_new_tokens=50, temperature=0.1, do_sample=True)

    response = test_tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Verification output: {response}")


if __name__ == "__main__":
    main()
