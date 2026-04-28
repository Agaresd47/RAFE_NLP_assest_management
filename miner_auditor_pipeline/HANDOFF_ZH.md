# Miner -> Auditor 本地流水线交接文档

## 范围

这个目录只负责一条独立的新流水线：

1. 用本地 `unsloth + QLoRA adapter` 跑 `task1.2 / miner / extract`
2. 把 miner 输出转换成 auditor 可直接消费的 HF dataset
3. 再通过现有 `sglang + qbatch.py` 路径跑 `task1.3 / auditor`
4. 通过 `sglang` 对本地 `hf_cot_sft` 跑 GRPO eval，并支持 TTC 可选

这个目录刻意只做 `miner -> auditor`，不扩展到别的链路。
现在这个目录也会同时保留本地数据副本，model 仍然放在目录外部。

## 文件说明

- `common.py`
  放共享逻辑，包括路径解析、miner 输出解析、factor 聚合、auditor dataset row 构造。
- `data/`
  放这个目录直接使用的数据副本。
- `run_miner_pipeline.py`
  第一步。跑本地 miner inference，同时输出 JSON 和 auditor 可用的 HF dataset。
- `run_auditor_pipeline.py`
  第二步。包装现有 `qbatch.py --mode auditor`。
- `run_grpo_eval.py`
  通过 `sglang` 对本地 `hf_cot_sft` 跑 GRPO eval，并支持 TTC 可选。
- `run_miner_pipeline.sh`
  第一步 shell 入口。
- `run_auditor_pipeline.sh`
  第二步 shell 入口。
- `run_grpo_eval.sh`
  GRPO eval 的 shell 入口。

## 设计选择

- miner 推理只走 `base model + adapter`
  不依赖 merged miner 目录。
- miner prompt 格式复用现有 `build_sft_extract_v5.py` 的 factor-level 逻辑。
- auditor prompt 格式复用现有 `task13_dataset_common.py`。
- auditor 推理这里不重写。
  第二步直接复用仓库里已有的 `qbatch.py` auditor-only 路径。
- 整体实现优先“最稳能跑通”，不是优先最省时或最优雅。

## 第一步：Miner

### 做什么

`run_miner_pipeline.py` 会：

1. 从下面两种输入源里选 filing：
   - raw MDA
   - 已有 factor-level miner HF dataset
2. 支持这些控制项：
   - `train | validation | test` split
   - 限制 ticker 数量
   - 限制 filing 数量
   - 选择跑全部 factor，或者只跑 teacher 命中过的 factor
3. 用本地 `unsloth` 跑 factor-level miner inference
4. 输出 factor-level 和 filing-level 的 miner JSON
5. 生成一个新的 HF dataset，给第二步 auditor 直接使用

### 关键参数

- `--input-source {factor_dataset,raw}`
- `--factor-dataset`
- `--raw-root`
- `--extract-root`
- `--split {train,validation,test}`
- `--question-mode {all,teacher_answered}`
- `--max-tickers`
- `--max-filings`
- `--output-dir`

`--question-mode` 的真实语义：

- `all`
  跑全局 schema 里的全部 factor。
- `teacher_answered`
  只保留这个 filing 里 teacher 至少命中过一次的 factor，但每个保留 factor 仍然喂这个 factor 的全量 candidate questions，不会缩成只喂 teacher 实际答到的那几个 question。

### 输出产物

miner 输出目录下会有：

- `selected_filings.json`
  记录本次选中了哪些 filing。
- `miner_pred_rows.json`
  factor-level 的 miner 预测结果。
- `miner_filing_aggregates.json`
  filing-level 聚合后的 miner 结果。
- `auditor_rows_preview.json`
  生成出的 auditor rows 预览。
- `summary.json`
  第一步摘要。
- `auditor_hf_dataset/`
  最关键的产物，第二步直接读取这个目录。

## 第二步：Auditor

### 做什么

`run_auditor_pipeline.py` 会：

1. 读取第一步生成的 HF dataset
2. 调用仓库现有 `qbatch.py --mode auditor`
3. 输出 auditor 预测 CSV 和 JSON

### `sglang` server 需要单独先启动

这条流水线默认假设：

- `sglang` server 在另一个 shell / tmux pane 里单独启动
- 第二步只负责连这个 server 跑 auditor

当前可直接参考的 GRPO serving 模型路径是：

`/scratch/xla2767/hold2/models/qwen3_8b_thinking_grpo_merged_v1`

启动示例：

```bash
python -m sglang.launch_server \
  --model-path /scratch/xla2767/hold2/models/qwen3_8b_thinking_grpo_merged_v1 \
  --port 30000 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.85
```

如果以后 serving 模型目录变了，只需要替换 `--model-path`。

### 关键参数

- `--auditor-dataset`
- `--model-path`
- `--sglang-url`
- `--split {train,validation,test}`
- `--batch-size`
- `--max-batches`
- `--output-dir`

### 输出产物

auditor 输出目录下会有：

- `auditor_predictions.csv`
- `auditor_predictions.json`
- `summary.json`

## 最小运行命令

### 第一步：Miner

```bash
bash /projects/p32908/nlp_code/miner_auditor_pipeline/run_miner_pipeline.sh \
  --input-source factor_dataset \
  --factor-dataset /scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5 \
  --split test \
  --question-mode teacher_answered \
  --max-filings 10000 \
  --output-dir /gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/miner_step \
  --batch-size 3
```

### 第二步：Auditor

先单独启动 `sglang` server，例如：

```bash
python -m sglang.launch_server \
  --model-path /scratch/xla2767/hold2/models/qwen3_8b_thinking_grpo_merged_v1 \
  --port 30000 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --mem-fraction-static 0.85
```

再执行：

```bash
bash /projects/p32908/nlp_code/miner_auditor_pipeline/run_auditor_pipeline.sh \
  --auditor-dataset /gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/miner_step/auditor_hf_dataset \
  --split validation \
  --sglang-url http://localhost:30000 \
  --output-dir /gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/auditor_step
```

## 第二步输入数据格式约定

这里生成的 `auditor_hf_dataset` 是按当前 auditor 工具链的习惯组织的：

- 含有 `messages`
- `messages[1]["content"]` 就是 auditor user prompt
- 同时保留常见元数据字段，例如：
  - `ticker`
  - `form`
  - `report_date`
  - `factor`
  - `evidence_count`
  - `historical_count`

这已经满足当前 `qbatch.py --mode auditor` 的读取方式，因为它现在就是直接从 `messages[1]["content"]` 取 prompt。

## 注意事项和约束

- miner 刻意固定为本地 `unsloth + adapter`
  这是遵循当前仓库里“miner 本地更稳”的判断。
- auditor 刻意继续走 `sglang`
- 这个目录里的 GRPO eval 默认直接读本地复制过来的 `data/hf_cot_sft`
- 如果能找到对应的 `TASK13_AUDIT`，生成的 auditor dataset 会顺带保留 teacher 相关字段，但第二步推理本身并不依赖 teacher target。
- 如果某个 factor 的 miner 没抽到任何 evidence，默认会被丢掉；如需保留空 factor，可加 `--keep-empty-factors`。
- 当前实现是稳定优先，不是速度优先。

## 本地数据

这个目录现在预期会包含：

- `data/hf_extract_sft_v5`
- `data/hf_cot_sft`

这个目录里的脚本默认都会优先读这些本地副本，而不是继续默认指向 `/scratch/...`。

## GRPO Eval

### 做什么

`run_grpo_eval.py` 会通过单独启动的 `sglang` server，对本地 `hf_cot_sft` 跑 GRPO eval。

### SGLang + TTC 行为

- `--ttc-n 1`
  不做 TTC。每条样本只请求一次 completion。
- `--ttc-n >1`
  开启 TTC。每条样本请求多次 completion，然后按 label 投票聚合。
- `--enable-thinking` / `--disable-thinking`
  控制请求里是否开启 Qwen thinking 模式。

### 最小命令

```bash
python /projects/p32908/nlp_code/miner_auditor_pipeline/run_grpo_eval.py \
  --sglang-url http://localhost:30000 \
  --ttc-n 1 \
  --split test \
  --max-filings 10
```

### 默认数据和模型约定

- dataset：
  `data/hf_cot_sft`
- model weights：
  由外部单独启动的 `sglang` server 提供
- outputs：
  `/gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/grpo_eval`

## 已完成验证

- 用 `py_compile` 做过 Python 语法检查
- 用 `--help` 做过参数入口检查
- 做过一个轻量 smoke test，确认在 `teacher_answered` 模式下能从 validation filing 重建 factor rows
- 已检查过一轮 GRPO eval 输出目录：
  `/gpfs/projects/p32908/nlp_result/miner_auditor_pipeline/grpo_eval`
- 当前那轮已检查到的 GRPO eval 指标：
  - `count=40`
  - `parse_success_rate=0.875`
  - `sentiment_accuracy=0.65`
  - `sentiment_mae_5way=0.425`
  - `confidence_mae=0.081714`

## 还没有做的端到端验证

- 没有实际跑完整 GPU miner inference
- 没有实际连 live `sglang` server 跑 auditor inference

这两步还需要在目标运行环境里实际执行。
