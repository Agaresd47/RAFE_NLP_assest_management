
  python -m sglang.launch_server \
    --model-path /scratch/xla2767/hold2/models/qwen3_8b_thinking_grpo_merged_v1 \
    --port 30000 \
    --dtype bfloat16 \
    --reasoning-parser qwen3 \
    --mem-fraction-static 0.85


python /projects/p32908/nlp_code/miner_auditor_pipeline/run_grpo_eval.py \
    --sglang-url http://localhost:30000 \
    --ttc-n 1 \
    --split test \
    --max-filings 10