# RAFE NLP Project Readme

## 这个项目到底要干嘛

这个仓库的目标，是把一套面向金融文档的训练流程搭起来，并逐步收口成稳定可复现的训练与评测 pipeline。

核心任务链路是：

1. 从原始 filing 文本和已有标注结果中构建训练数据。
2. 先做 `task1.3 / Auditor` 的 CoT SFT，让模型学会输出格式、基本推理方式和 teacher 风格。
3. 再做 return-guided preference optimization，让模型在投资目标上进一步收口。
4. 将训练得到的 LoRA adapter merge 回基座模型。
5. 用 merged 模型跑 batch inference / eval，输出可用于后续分析的结果文件。

## 当前任务定义

当前代码里的任务拆分是：

- Miner:
  从原始文本中根据 schema 抽取与问题相关的原句和置信度。
- Auditor:
  基于当前证据和历史上下文，对单个 factor 给出情绪判断和置信度。

这两个任务都使用 chat `messages` 格式作为统一数据结构。

但当前新的推荐训练主线，已经不再把 Miner 和 Auditor 绑成同一个训练目标。

更准确的理解是：

- 旧主线仍然保留：`build_dataset.py -> train.py -> qlora_merge.py -> qbatch.py`
- 新主线已经落地：`build_sft_dataset.py -> train_sft.py -> build_dpo_dataset.py -> train_dpo.py`
- 新主线当前只围绕 `task1.3 / Auditor`

## 当前阶段的判断

这个仓库现在的重点，不是重新发明任务定义，而是把已经存在的主链路收口，并把训练阶段拆成更合理的几步。

当前判断：

- 框架已经搭起来了，不是空壳。
- 已经存在旧主链路，也已经落地新的 `task13` 主线。
- 现在的主要问题不是“有没有功能”，而是“训练策略是否一致、数据入口是否统一、路径是否统一、脚本是否能稳定端到端复现”。
- 当前 `task13` 主线已经进一步收口到统一的 thinking-output 契约：
  - SFT 训练目标不再是单个 assistant JSON
  - 而是 `"<think> ... </think>" + 最终 JSON`
  - 最终 JSON 只保留：
    - `sentiment_label`
    - `confidence_score`
- 新版 DPO 数据也已经对齐到同样的输出格式，不再继续使用旧的纯 JSON `reasoning_chain` 监督格式。
- 当前更推荐的训练策略是三步走：
  1. `SFT`
  2. 用当前离线规则数据做一轮“笨 DPO”
  3. 再进入更强的 `PRM/RM` 驱动的“聪明 DPO”

## 当前默认资源约定

当前更稳定的默认约定是：

- 代码：`/gpfs/projects/p32908/nlp_code`
- HF dataset / adapter / eval 产物：优先 `/scratch/xla2767/hold2`

当前实际默认常用目录是：

- SFT dataset: `/scratch/xla2767/hold2/data/nlp/hf_cot_sft`
- DPO dataset: `/scratch/xla2767/hold2/data/nlp/hf_cot_dpo`
- SFT adapter: `/scratch/xla2767/hold2/data/nlp/qwen3_8b_thinking_sft_out`
- DPO adapter: `/scratch/xla2767/hold2/models/cot_dpo_adapter`
- GRPO adapter: `/scratch/xla2767/hold2/models/cot_grpo_adapter`
- SFT merged model: `/scratch/xla2767/hold2/models/qwen3_8b_thinking_sft_merged_v2`
- DPO merged model: `/scratch/xla2767/hold2/models/qwen3_8b_thinking_dpo_merged_v1`
- 全量 eval 输出：
  - SFT: `/gpfs/projects/p32908/nlp_result/sft_test`
  - DPO: `/gpfs/projects/p32908/nlp_result/dpo_test`

## 未来 session 进入仓库时应先确认的事

每次新 session 接手时，优先确认：

1. 训练数据入口到底是本地 `save_to_disk` 数据集，还是 Hugging Face dataset name。
2. 训练输出目录、merge 输出目录、eval 模型目录是否一致。
3. 当前改动是在修“框架缺失”，还是在做“接口收口 / 参数统一 / 路径统一”。
4. 当前是在跑旧 Miner/Auditor 联合主线，还是跑新的 `task13` 主线。
5. 大文件产物是默认放 `gpfs`，还是默认放 `/scratch/xla2767/hold2`。
6. 当前 `train_sft.py` / `train_dpo.py` 是否都在使用同一个 Qwen tokenizer 配置：
   - `eos_token = <|im_end|>`
   - `pad_token = <|im_end|>`
7. 当前 DPO 是否在自动读取最新的 SFT checkpoint，而不是读取过期的根目录 adapter。
8. 当前大规模全量 eval 是否仍在走单条 HF/Unsloth `generate`。
   - 如果只是抽样 sanity check，可以继续用本地脚本。
   - 如果是全 test set 批量推理，更适合 merge 后单独起一张卡走 `sglang` 或其它高吞吐 serving。
9. 当前 `qbatch.py` 是否已经切到“Direct Auditor Mode”。
   - 新主线默认可以直接读取本地 `hf_cot_sft`
   - 不再强依赖旧的 miner -> auditor 双阶段数据集

## 判断标准

后续改动默认应服务于下面这个目标：

- 让这套流程从“能跑的实验脚本组合”变成“可复用、可解释、可重复执行”的项目框架。
