#!/usr/bin/env bash
set -euo pipefail

ORG="Vandy-NLPasset-26-G1"
SRC_ROOT="/projects/p32908/data/nlp/finished_model"

REPOS=(
  "qwen3_8b_thinking_sft_out"
  "cot_grpo_adapter_v2"
  "skywork_reward_v2_qwen3_4b"
)

echo "[check] huggingface auth"
hf auth whoami

for name in "${REPOS[@]}"; do
  src_dir="${SRC_ROOT}/${name}"
  if [[ ! -d "${src_dir}" ]]; then
    echo "[skip] missing directory: ${src_dir}"
    continue
  fi

  repo_id="${ORG}/${name}"
  echo "[upload] ${src_dir} -> ${repo_id}"

  python - "${repo_id}" "${src_dir}" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id = sys.argv[1]
folder_path = sys.argv[2]

api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=folder_path,
)
print(f"[done] {folder_path} -> {repo_id}")
PY
done

echo "[done] all uploads finished"
