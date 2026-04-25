import train_sft_extract_v3 as base


base.DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
base.DEFAULT_DATASET_PATH = "/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v4"
base.DEFAULT_OUTPUT_DIR = "/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v4_out"
base.PER_DEVICE_TRAIN_BATCH_SIZE = 1
base.PER_DEVICE_EVAL_BATCH_SIZE = 1
base.GRADIENT_ACCUMULATION_STEPS = 16


if __name__ == "__main__":
    base.main()
