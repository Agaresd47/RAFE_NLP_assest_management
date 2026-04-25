# Current Pipeline Runbook

## 这份文档是干嘛的

这份文档记录的是：

- 当前推荐的端到端运行顺序
- 每一步依赖什么输入
- 每一步产出什么结果
- 当前哪些地方已经能跑，哪些地方还没有完全统一

这不是 worklog，也不是长期愿景文档。

它的作用是让新的 session 进入仓库后，能快速知道“现在这套东西实际上应该怎么跑”。

## 当前推荐主链路

当前推荐按下面顺序理解和执行：

1. 准备原始 raw filing 和 extraction / audit 标注数据。
2. 运行 `build_dataset.py`，构建 Miner / Auditor 数据集。
3. 运行 `train.py`，做 SFT / QLoRA 训练。
4. 运行 `qlora_merge.py`，把 adapter merge 回基座模型。
5. 启动 `sglang` server。
6. 运行 `qbatch.py`，做 batch inference / eval，导出 CSV。

## 当前新的推荐主线

如果目标已经切到“预测股票而不是复刻旧 Miner/Auditor 联合流程”，当前更推荐的新主线是：

1. 运行 `build_sft_dataset.py`，从 `TASK13_AUDIT` JSON 构建 `task1.3` CoT SFT 数据集。
2. 运行 `train_sft.py`，对本地 `hf_cot_sft` 数据集做 QLoRA SFT。
3. 运行 `build_dpo_dataset.py`，给样本贴上 21 个交易日 forward return / excess return，并构造 `prompt/chosen/rejected`。
4. 运行 `train_dpo.py`，从 SFT adapter 继续做 return-guided DPO。
5. 运行 `qlora_merge.py`，按需要把最终 adapter merge 回 base model。
6. 如需批量推理，再启动 `sglang` server 并运行 `qbatch.py`。

当前默认产物目录是：

- SFT 数据集：`/scratch/xla2767/hold2/data/nlp/hf_cot_sft`
- DPO 数据集：`/scratch/xla2767/hold2/data/nlp/hf_cot_dpo`

当前常用训练产物目录是：

- SFT adapter：`/scratch/xla2767/hold2/data/nlp/qwen3_8b_thinking_sft_out`
- DPO adapter：`/scratch/xla2767/hold2/models/cot_dpo_adapter`
- GRPO adapter：`/scratch/xla2767/hold2/models/cot_grpo_adapter`
- SFT merged：`/scratch/xla2767/hold2/models/qwen3_8b_thinking_sft_merged_v2`
- DPO merged：`/scratch/xla2767/hold2/models/qwen3_8b_thinking_dpo_merged_v1`

当前额外补齐的离线全量 eval 脚本有：

- `eval_sft_full.py`
- `eval_dpo_full.py`

默认输出目录：

- SFT: `/gpfs/projects/p32908/nlp_result/sft_test`
- DPO: `/gpfs/projects/p32908/nlp_result/dpo_test`

当前默认训练入口是：

- `train_sft.py`
- `train_dpo.py`

这条新主线的设计原则是：

- 只围绕 `task1.3` / Auditor 做训练。
- SFT 阶段先学会稳定的 thinking-style 输出格式。
- DPO 阶段再用 `forward_returns_1m.py` 的 21 天收益信号做偏好学习。
- 旧 `train.py` / `build_dataset.py` 保留，但不再作为这条主线的默认入口。

当前新的输出格式约定是：

- 训练和偏好学习阶段都优先使用
  - `<think> reasoning </think>`
  - `{"sentiment_label": "...", "confidence_score": ...}`

不要再默认把 auditor 输出理解成单个 assistant JSON。

## 当前 extract 支线

如果当前目标不是 `task1.3 / Auditor`，而是 `task1.2 / Miner / extract`，当前更务实的路线不是“一次性整套 schema 输出”，而是：

1. 先把训练样本拆成 single-question extract。
2. 训练时只让模型学：
   - 单个 `factor + question_key`
   - 对应的 quote 列表 JSON
3. 推理时再循环所有 question 并聚合结果。

如果继续往前推进，当前还有一条更新的 `v5` 路线：

1. 输入不再是单个 question。
2. 改成单个 factor block。
3. 一个 factor block 下面带多个 candidate questions。
4. 允许返回空 `extractions`，让模型学会“这个 factor 在当前 filing 里没答案”。
5. 默认从 `v4` adapter warm start，而不是从 base 冷启动。

当前这条 extract 支线的关键脚本是：

- dataset builder:
  - `build_sft_extract_v4.py`
- trainer:
  - `train_sft_extract_v4.py`
- factor-level v5 builder:
  - `build_sft_extract_v5.py`
- factor-level v5 trainer:
  - `train_sft_extract_v5.py`
- single-question sanity check:
  - `eval_try_sft.py`
- full-schema aggregate inference:
  - `infer_extract_aggregate.py`

当前 extract 支线默认目录是：

- dataset:
  - `/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v4`
- adapter:
  - `/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v4_out`
- v5 dataset:
  - `/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5`
- v5 adapter:
  - `/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out`

当前 extract 支线的几个重要约束是：

- 训练侧当前默认是 `Qwen3-8B`
- packing 关闭
- 只对 assistant JSON 算 loss
- 同一样本内 quote 已做去重
- 训练样本已经按 `factor + question_key` 拆开
- 高频 question 已做一定程度重平衡
- `v5` 默认从 `v4` adapter warm start

当前最重要的工程判断是：

- 不要默认要求模型一次性从整篇 filing 输出全 schema 大 JSON
- 更稳定的做法是：
  - 单 question 推理
  - 再由聚合脚本合并成最终结果

当前 `infer_extract_aggregate.py` 还支持：

- `--raw-file`
- 或者 `--ticker --form --date`

后者会自动去 `MDA_Raw/<ticker>/<form>/` 下定位原文件。

## 远端计算节点协作约定

这个项目的训练、merge、推理这类重活，可能会放到用户临时开出的远端计算节点上执行。

这里要记录的不是某一台机器的名字，而是协作约定：

- 节点名不是长期配置，用户每次 session 可能都会开新的节点。
- 新的 Codex session 读完这些文档后，不应假设固定主机名仍然有效。
- 新的 Codex session 应主动确认：用户是否已经开好了可用的远端 session / 节点。
- 对这个问题的默认预期不是“还没开”，而是先问一句是否已经开好。

当前已确认的真实情况是：

- 用户手动交互式 SSH 登录远端节点是可行的。
- 远端节点上可以看到项目目录与运行环境。
- 因此后续如果 Codex 侧远程执行异常，优先怀疑的是当前非交互 SSH 执行方式、远端 shell 初始化或会话启动细节，而不是直接判断“节点不可用”。
- 当前对 Codex 已验证可用的进入方式，是先在本地创建 `tmux` session，再在该交互式会话里 `ssh` 到远端节点。
- 如果后续需要在远端节点上连续执行多条命令，优先沿用这条路径。

## Step 1: 构建数据集

脚本：

- `build_dataset.py`

目标：

- 从 raw markdown filing 和已有的 `TASK12_EXTRACTIONS` / `TASK13_AUDIT` JSON 构建训练数据。
- 产出统一的 chat `messages` 格式数据集。

当前脚本能力：

- 解析 raw 文件名和 extraction 文件名
- 配对 raw 与 task12 / task13
- 根据年份切分 train / validation / test
- 构建 miner 数据集
- 构建 auditor 数据集
- 将数据集 `save_to_disk`

当前注意点：

- 默认路径还是旧的 Windows 路径，需要显式传参或后续统一。
- 这个脚本本身承担的是“数据集构建器”职责，不是训练器。

如果使用新主线：

- `build_dpo_dataset.py` 最好显式传 `--output-dir`
- 否则容易把新版数据写到 `gpfs`，而训练脚本继续从 `/scratch` 读旧目录

## Step 2: 训练

脚本：

- `train.py`

目标：

- 用 Unsloth + QLoRA 对 Miner 和 Auditor 任务做联合 SFT 训练。

当前脚本能力：

- 加载 Qwen base model
- 配置 LoRA
- 加载 miner / auditor 数据
- 拼接 train / validation
- 用 chat template 格式化文本
- 过滤超长样本
- 运行 SFTTrainer
- 保存 adapter 和 tokenizer

当前注意点：

- `train.py` 里虽然定义了本地路径参数，但实际加载数据时还是使用固定 dataset name。
- 这意味着“本地 `build_dataset.py` 输出”与“训练入口”还没真正统一。

如果切到 `train_sft.py` / `train_dpo.py` 这条新主线，还要知道两个当前已经落地的行为：

- `train_sft.py`
  - 会把 assistant JSON 重写成 thinking-block 目标再训练
  - 默认从输出目录最新 checkpoint warm start 权重
  - 但不会恢复旧 trainer state
- `train_dpo.py`
  - 当前也已经对齐到同一个 Qwen tokenizer/eos/pad 契约
  - 默认会优先读取 SFT 输出目录下最新 checkpoint 作为 DPO 初始化 adapter

如果继续看 `train_grpo.py`，当前应再记住几件事：

- GRPO 的 train/eval 不是普通 loss-only forward，而是 rollout 风格评估
- `num_generations`、completion 长度和 eval 数据量会直接决定速度
- 训练内 eval sample 只能当粗检，不能替代独立推理评估
- 当前脚本已经默认把内部 eval 截到较小子集，避免 full validation 直接拖死

## Step 3: Merge

脚本：

- `qlora_merge.py`

目标：

- 从训练产出的 checkpoint 里加载 adapter 权重。
- 将 LoRA adapter merge 回 base model。
- 保存 merged 模型，供后续推理和评测使用。

当前脚本能力：

- 可以加载 base model
- 可以读取 adapter safetensors
- 可以保存 merged 模型
- 包含一个简单推理验证片段

当前注意点：

- 现在已经参数化了 base model、adapter 路径、merge 输出目录和验证目录；旧默认值仍保留作 fallback。
- 如果你切到新主线，优先把 `--base-model`、`--adapter-path`、`--merged-output` 对齐到同一条产物链。

## Step 4: 启动推理服务

当前 `qbatch.py` 依赖外部 `sglang` server。

当前示例启动方式在 `qbatch.py` 顶部注释里已经给出。

核心含义是：

- 用 merged 模型启动本地服务
- 让 `qbatch.py` 通过 OpenAI 风格接口批量请求该服务

当前注意点：

- server 模型路径必须和 merge 产物路径一致。
- 如果 merge 输出路径改了，server 启动命令和 `qbatch.py` 里的 `MODEL_PATH` 也要一起改。
- 对 `Qwen3 + --reasoning-parser qwen3`：
  - 最终答案在 `message.content`
  - reasoning 在 `message.reasoning_content`
  - 不要默认再从 `content` 里强行找 `<think>...</think>`

## Step 5: Batch Eval

脚本：

- `qbatch.py`

目标：

- 在 test 集上先跑 Miner
- 再根据 Miner 输出构造 Auditor 输入
- 最终输出 signal CSV

当前脚本能力：

- 等待 `sglang` server 启动
- 加载 miner / auditor test 数据
- 跑 Miner batch inference
- 构造 Auditor prompt
- 跑 Auditor batch inference
- 导出 CSV

当前注意点：

- 现在可以通过参数切换 merged model、HF dataset 名称或本地 `load_from_disk` 数据集、输出 CSV 和评测 split。
- `--max-batches` 默认不再硬截断，旧 smoke test 行为可以显式传 `--max-batches 2` 复现。
- 如果你要离线跑本地数据，传 `--offline-datasets` 即可，不再把离线模式写死在脚本里。
- 当前新主线默认更适合走：
  - `Direct Auditor Mode`
  - 本地 `hf_cot_sft`
  - merged model + `sglang`
- 当前导出不再只盯 CSV：
  - JSON 更适合保留 `Think` 和最终 `Output`
  - CSV 更像方便快速扫结果的副产物

## 当前推荐理解方式

现在不要把这套代码理解成一套完全收口的工程化 pipeline。

更准确的理解是：

- 主链路已经明确存在
- 各阶段职责也已经明确
- 但不同阶段之间的接口、路径、命名、默认参数还没有统一

所以当前推荐工作方式是：

1. 先确认本次改动要影响哪个阶段。
2. 再确认该阶段的输入输出和上下游是否一致。
3. 改动时优先收口接口，而不是继续堆新的脚本。

## 新 session 进入时建议先检查什么

建议先检查下面几件事：

1. `build_dataset.py` 的输入路径和输出路径现在指向哪里。
2. `train.py` 实际是从本地磁盘数据集加载，还是从 Hugging Face dataset name 加载。
3. `qlora_merge.py` 读取的是哪个 checkpoint，写出到哪个模型目录。
4. `qbatch.py` 依赖的 `MODEL_PATH` 和 `sglang` server 路径是否一致。
5. 当前 eval 是想跑 smoke test 还是 full test。
6. 用户是否已经开好了本次 session 要用的远端计算节点。
7. 当前这次 session 是否应通过 `tmux + 交互式 ssh` 的方式进入远端节点。
8. 当前这次 session 的 eval 是不是需要真正的高吞吐批量推理。
   - 如果是，优先考虑 merge 后单独起卡跑 serving / `sglang`
   - 不要默认拿单条 HF/Unsloth `generate` 做全量评估
