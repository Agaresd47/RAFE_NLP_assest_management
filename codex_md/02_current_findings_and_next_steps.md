# Current Findings And Next Steps

## 当前框架判断

当前结论：

- 仓库已经具备基础的 SFT 训练和 eval 框架。
- 旧主线仍然存在，但新主线也已经落地。
- 当前更准确的状态不是“从零开始做 RL”，而是已经有一条可运行的 `task13` SFT + return-guided DPO 主线。
- 它目前仍然更像“能跑通的 v0 实验框架”，还不是完全收口的工程化 pipeline。

旧主链路是：

- `build_dataset.py`: 构建 Miner / Auditor 数据集。
- `train.py`: 用 Unsloth + QLoRA 做 SFT 训练。
- `qlora_merge.py`: 将 adapter merge 回基座模型。
- `qbatch.py`: 用 merged 模型做 batch inference / eval，并导出结果。

当前新的推荐主链路是：

- `build_sft_dataset.py`: 从 `TASK13_AUDIT` JSON 构建 `task13` CoT SFT 数据集。
- `train_sft.py`: 用本地 `save_to_disk` HF dataset 做 SFT。
- `build_dpo_dataset.py`: 贴上 21 个交易日收益标签，并构建 DPO pair。
- `train_dpo.py`: 从 SFT adapter 继续做 DPO。
- `qlora_merge.py`: 参数化 merge。
- `qbatch.py`: 参数化 batch eval。

## 已发现的关键问题

### 1. 旧主线和新主线并存，必须明确当前跑哪一条

当前仓库里已经不是只有一条训练路径。

- 旧主线偏向 `Miner + Auditor` 联合训练。
- 新主线偏向 `task13 / Auditor` 单任务训练。

如果 session 进入后不先确认当前跑哪一条，很容易误把旧脚本默认值当成新方案。

### 2. 旧主线的数据入口没有统一

`build_dataset.py` 会把数据集 `save_to_disk` 到本地目录。

但 `train.py` 实际加载时，用的是固定的 Hugging Face dataset name，而不是命令行参数里的本地路径。

这说明逻辑链路已经存在，但接口还没真正接上。

### 3. 新主线的数据入口已经统一，但运行根目录约定还没完全统一

当前新主线已经默认走本地 `save_to_disk` HF dataset：

- `hf_cot_sft`
- `hf_cot_dpo`

但还需要明确：

- 代码基座继续放 `gpfs`
- 大文件产物默认放 `gpfs` 还是 `/scratch/xla2767/hold2`

当前更合理的约定是：

- 代码：`/gpfs/projects/p32908/nlp_code`
- 大文件产物：优先 `/scratch/xla2767/hold2`

当前已经确认的新主线常用默认目录是：

- SFT dataset: `/scratch/xla2767/hold2/data/nlp/hf_cot_sft`
- DPO dataset: `/scratch/xla2767/hold2/data/nlp/hf_cot_dpo`
- SFT output: `/scratch/xla2767/hold2/data/nlp/qwen3_8b_thinking_sft_out`
- DPO output: `/scratch/xla2767/hold2/models/cot_dpo_adapter`

### 4. 路径约定没有统一

不同脚本里混用了不同根路径和输出目录，比如：

- `/projects/...`
- `/gpfs/projects/...`
- `/scratch/...`
- 历史 Windows 路径 `E:\...`

这说明代码来源和运行环境已经发生过迁移，但没有完成统一收口。

### 5. merge 环节已经参数化，但训练策略还需要继续升级

`qlora_merge.py` 已经不再是完全写死路径的脚本。

但训练策略本身现在分成三步才更合理：

1. `SFT`
2. 当前离线规则版的“笨 DPO”
3. 更强的 `PRM / RM` 驱动的“聪明 DPO”

所以当前的重点已经不是“有没有 DPO”，而是“现在这版 DPO 是什么强度、下一步要怎么升级”。

### 6. eval 链路仍然是通路打通版，不是完整评测框架

`qbatch.py` 目前已经可以完成 Miner -> Auditor -> CSV 输出的主流程。

但它仍然存在一些明显的实验态特征：

- 依赖外部启动的 `sglang` server
- 模型路径写死
- 只跑部分测试数据
- 注释与实际执行上限不完全一致

所以它证明了 eval 主流程已经存在，但还没有完全收口。

最近这条链路又有两个新的稳定事实：

- `qbatch.py` 已经可以默认切到新主线的 “Direct Auditor Mode”
  - 直接读取本地 `hf_cot_sft`
  - 直接把 `messages[1]["content"]` 送给 merged model / `sglang`
  - 不再强依赖旧的 miner -> auditor 双阶段数据
- `sglang + Qwen3 reasoning parser` 这条服务化路线已经验证可用，而且批量推理速度明显快于本地单条 HF/Unsloth `generate`

### 7. 当前 DPO 已经不是单模板 reject，而是多候选打分版

当前 `build_dpo_dataset.py` 的真实实现已经是：

- 先为每条样本生成多个候选：
  - `teacher_full`
  - `teacher_concise`
  - `return_aligned`
  - `conservative`
- 再按下面三类分数离线排序：
  - `teacher_quality`
  - `grounding`
  - `return_alignment`
- 选最高分做 `chosen`
- 选最低分做 `rejected`

这意味着当前 DPO 已经能表达“PRM 权重大、RM 次之”的方向。

但它仍然是轻量级离线规则版，不是最终形态。

另外还有两个现在已经稳定下来的真实约束：

- DPO `chosen/rejected` 已经统一成和新版 SFT 相同的 thinking-output 格式：
  - `"<think> reasoning </think>" + final JSON`
- 当前真正有信息量的难 pair 主要不是 `*_over_conservative`
  - reward model 评估显示：
    - `return_aligned_over_conservative`
    - `teacher_concise_over_conservative`
    - `teacher_full_over_conservative`
    这几类都很容易
  - 真正难的是 `return_aligned_over_teacher_full`
  - 所以后续如果要继续做 DPO / RM 分析，优先关注这类 pair

### 8. 当前 DPO 本体基本正常，GRPO sample callback 不能直接当模型结论

最近补了独立推理脚本后，已经确认：

- `eval_dpo_try.py` 走“base model + latest DPO adapter”的独立推理路径时，
  DPO 输出整体是正常的：
  - 能稳定输出 `<think> ... </think>` + 最终 JSON
  - 没有大面积出现 `Assistant:` 复读 / 吐泡泡
- 因此，之前在 `train_grpo.py` 的 `eval sample` 里看到的脏输出，
  不能直接解释成 “DPO 已经炼烂了”
- 更合理的判断是：
  - `train_grpo.py` 里的 sample 展示生成路径和正式 eval / inference 路径并不完全一致
  - 那块输出只能做粗检，不能当最终质量结论

当前还观察到一个具体偏移：

- DPO 在某些“有缓释措施的风险因子”上，倾向于比 teacher/GT 更乐观
- 典型例子是：
  - `AAPL / structural_risks_tech.supply_chain_concentration`

这说明：

- DPO 没坏
- 但当前偏好构造可能会系统性奖励“存在 mitigation 叙事”的答案
- 后续如果继续沿 return / RL 方向推进，要小心把这种“偏乐观”继续放大

### 9. merge 路线已经修到和当前训练契约一致

最近确认过一个很关键的问题：

- 旧版 `qlora_merge.py` merge 出来的模型可能表现异常
- 同一个 adapter 用 Unsloth 直接推理时正常，但旧 merge 产物会更容易“胡说八道”

当前已经把 merge 路线修到和训练时一致：

- 复用 `train_common.py` 里的：
  - `load_unsloth_model(...)`
  - `load_lora_adapter_weights(...)`
  - `resolve_latest_adapter_path(...)`
- 这样 merge 时也能继承当前 Qwen tokenizer/eos/pad 配置和 adapter dtype 对齐逻辑

当前已经验证：

- `qwen3_8b_thinking_sft_merged_v2` 的行为比旧版 merged 明显正常

### 10. `sglang` 返回结构和本地 `generate` 不同，必须分开处理

最近已经确认：

- 对 `Qwen3 + --reasoning-parser qwen3`
- `sglang` 的 OpenAI 风格接口会把：
  - 最终答案放到 `message.content`
  - reasoning / CoT 放到 `message.reasoning_content`

这意味着：

- 不能再假设 `<think>...</think>` 一定在 `content` 里
- 批量推理脚本如果要保留 CoT，必须显式读取 `reasoning_content`
- `content = null` 并不一定表示模型完全失败，可能只是把 token 全花在 reasoning 上、还没来得及收口到最终 JSON

当前 `qbatch.py` 已经调整成：

- `Output` 读取 `content`
- `Think` 优先读取 `reasoning_content`
- 默认同时导出 JSON，便于后处理

### 11. 远端计算资源是 session 级上下文，不是固定配置

当前项目的重任务执行可能依赖用户临时开出的远端计算节点。

这里真正稳定的信息不是某一个节点名，而是下面这些约束：

- 节点名和目标主机可能每次 session 都会变化。
- 新 session 不应把上一次的主机名当成项目固定配置。
- 新 session 进入后，应先确认用户是否已经开好了本次要用的远端 session / 节点。
- 用户手动交互式 SSH 登录可能正常，但 Codex 当前的非交互远程执行不一定等价，需要单独验证。
- 当前已验证的更可靠方式是：先在本地起一个 `tmux` session，再从该交互式终端里 `ssh` 到远端节点。
- 因此后续如果需要在远端节点上持续执行命令，不要默认依赖一次性的非交互 `ssh host cmd`。

## 当前优先级判断

当前最重要的，不是继续扩脚本，而是把三步训练策略和运行路径约定收口。

优先级应是：

1. 统一数据入口。
2. 统一训练、merge、eval 的模型与目录约定。
3. 明确旧主线和新主线的职责边界。
4. 去掉明显一次性实验脚本风格的硬编码。
5. 把端到端运行方式沉淀成清晰接口。
6. 把训练内 sample / callback 和真正独立 eval 分开看，不要混用结论。

## 下一步改动应优先做什么

建议后续优先把下面几件事做好：

### A. 先跑通并验证三步训练策略

当前推荐策略是：

1. `SFT`
2. 用当前规则版 HF 做一轮“笨 DPO”
3. 再升级到 `PRM + RM + confidence` 驱动的“聪明 DPO”

当前最值得继续沉淀的是：

- 第二步不要训太久，避免过拟合规则候选。
- 第三步里 `PRM` 更偏 teacher / grounding，`RM` 更偏 stock return。
- 真实升级点不是“换个 trainer 名字”，而是把离线规则分数逐步替换成真实 `PRM/RM` 分数。

### B. 明确基座模型、PRM、RM 的角色

目标是明确：

- base model 是什么。
- `PRM` 是不是 step-level / rationale-quality 模型。
- `RM` 是否单独存在，还是由 stock return bucket / excess return 先做代理。

### C. 把大文件默认迁到 `/scratch/xla2767/hold2`

目标是明确：

- 代码继续放 `gpfs`
- env / model / hf dataset / cache 优先放 `/scratch/xla2767/hold2`
- 新 session 默认从这个运行资源根目录起训练和推理

### D. 明确训练脚本当前已经收口到哪些行为

当前已经确认：

- `train_sft.py`
  - 默认做 Qwen thinking-style SFT
  - 输出格式是 `<think> + final JSON`
  - 默认会从输出目录下最新 `checkpoint-*` warm start 权重
  - 这个 warm start 会加载旧 adapter，但不会恢复旧 trainer state
- `train_dpo.py`
  - 当前已经对齐到 `Qwen/Qwen3-8B`
  - 已同步新版 tokenizer / eos / pad 配置
  - 默认优先读取 SFT 输出目录里的最新 checkpoint
  - 如果没有 checkpoint，再回退到根目录 adapter
  - 当前走的是 `Unsloth + 4bit + LoRA + TRL DPOTrainer`
- `train_grpo.py`
  - 当前已经补了 batch 对齐日志、`generation_batch_size` 自动对齐、以及小 eval 子集截断
  - 但 GRPO 的 train/eval 仍然是高成本 rollout 路线
  - 训练内 eval 更适合做稳定性监控，不适合替代独立推理评估

## 当前已经落地的重要事实

- `hf_cot_sft` 已经构建完成
- `hf_cot_dpo` 已经构建完成
- 新版 DPO 已经是多候选打分版
- `/scratch/xla2767/hold2/data/nlp/hf_cot_dpo` 已确认是 thinking-block 新版数据
- `sample.md` 已经提供了 2 个 SFT 和 2 个 DPO 真实样本
- `hf_dataset_build_notes.md` 已经记录了过滤规则、删掉了什么、以及当前 DPO 的构造方式
- `eval_sft_full.py` 已补齐，可导出：
  - `gt.json`
  - `pred.json`
  - `metrics.json`
- `eval_dpo_full.py` 已补齐，可导出：
  - `gt.json`
  - `pred.json`
  - `metrics.json`
- 当前默认全量 eval 输出目录：
  - SFT: `/gpfs/projects/p32908/nlp_result/sft_test`
  - DPO: `/gpfs/projects/p32908/nlp_result/dpo_test`
- 当前全量 eval 仍然走 HF/Unsloth 单条 `generate`
  - 适合一次性揭锅
  - 不适合作为长期高吞吐批量评估方案

## 当前推荐的模型角色理解

如果后续选择：

- base: `Qwen/Qwen3-8B`
- reward model: `Skywork/Skywork-Reward-V2-Qwen3-4B`

那么当前更合理的理解是：

- `Qwen3-8B` 做当前 SFT / DPO 主模型
- `Skywork Reward` 更像 `RM / judge`
- 如果要做真正的 `PRM`，仍然更推荐单独准备一个偏 rationale / grounding / teacher-quality 的打分模型

也就是说，`base + RM` 这两件已经基本齐了；如果要把“PRM 权重大”落实成真正的独立模块，最好再补一个单独 PRM。

## 这份文档的用途

这份文档不是 worklog。

它的用途是让新的 Codex session 读完后，立刻知道：

- 这个项目已经做到哪一步了。
- 当前判断是什么。
- 下一步应该优先改哪里。
