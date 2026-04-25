import train_sft_extract_v3 as base
from train_common import resolve_latest_adapter_path


V4_ADAPTER_PATH = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v4_out"

base.DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
base.DEFAULT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5"
base.DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out"
base.PER_DEVICE_TRAIN_BATCH_SIZE = 1
base.PER_DEVICE_EVAL_BATCH_SIZE = 1
base.GRADIENT_ACCUMULATION_STEPS = 16


_original_resolve_latest_adapter_path = base.resolve_latest_adapter_path


def _resolve_latest_adapter_path_with_v4_default(path):
    resolved = _original_resolve_latest_adapter_path(path)
    if resolved:
        return resolved
    return resolve_latest_adapter_path(V4_ADAPTER_PATH)


base.resolve_latest_adapter_path = _resolve_latest_adapter_path_with_v4_default


if __name__ == "__main__":
    base.main()
