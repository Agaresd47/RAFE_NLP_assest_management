# Extract Track Current

## 这份文档记录什么

这份文档只记录当前 `task1.2 / Miner / extract` 这条线里最关键、最长期有效的判断。

不记录：

- 逐次试错过程
- 单次报错细节
- 一次性命令回显

## 当前结论

当前 `extract` 线已经验证出一件很重要的事：

- 问题不只是模型大小。
- 更关键的是训练样本的组织方式。

旧版做法是：

- 一整篇 filing
- 整套 schema
- 一次性输出一个包含很多 `extractions` 的大 JSON

这条路对 `Qwen3-4B` 很容易塌成：

- 记高频 factor
- 记高频 question
- 记高频 quote
- 输出固定模板

即使格式学到了，也很容易：

- 反复输出同一个 factor / question
- 只输出 1 条 quote
- 或者在同一句 quote 上复读

## 重新检查 `hf_extract_sft_v3` 后的判断

`/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v3` 本身没有明显脏格式问题：

- assistant 目标都是合法 JSON
- 没有混入 auditor / thinking / DPO 格式
- `messages / prompt / response` 结构是干净的

但它有三个会直接诱导模式塌缩的问题：

1. 标签分布明显偏斜
   - 高密度集中在少数 factor / qkey
   - 例如 `m_and_a_and_corpdev.integration_execution_risk / Q26`
   - 以及 `financial_quality_and_balance_sheet.liquidity_and_runway / Q59`

2. 单条样本目标太大
   - 平均每条样本有十几个 extraction
   - 最大可到百级别
   - 对 `4B` 很不友好

3. teacher 目标里同一条 quote 在同一条样本内会重复出现
   - 这不是“GPT 乱说”
   - 而是 teacher 会把同一句原文标给多个 extraction 项
   - 这会直接教会学生模型“找到一条高概率 quote 后重复输出”

这里所谓的“重复 quote”，意思是：

- teacher 给出的 `extractions` 中
- 同一条 `original_quote`
- 会在同一条训练样本里出现多次

不是说 GPT 在自然语言层面胡乱复读整段解释。

## `v4` 的核心变化

为了解决上面的结构性问题，当前已经引入 `v4`：

- builder: `/projects/p32908/nlp_code/build_sft_extract_v4.py`
- trainer: `/projects/p32908/nlp_code/train_sft_extract_v4.py`

`v4` 的关键设计是：

1. 按 `factor + question_key` 拆样本
   - 一条训练样本只对应一个目标问题
   - 不再让模型一次性学整套 schema

2. 同一样本内对 `original_quote` 去重
   - 先按 `relevance_confidence` 排序
   - 相同 quote 只保留一条

3. 对训练集做 question 级频次裁剪
   - 防止高频 `Q26 / Q59 / Q63` 继续主导训练

4. 保持训练时 `assistant-only loss`
   - system + user 全部 mask
   - 只对 assistant JSON 算 loss

5. packing 关闭
   - 当前这条 extract 线不再做 packing

## `v4` 的真实含义

`v4` 训练出来的不是：

- “整篇 filing + 全 schema 一次性输出完整大 JSON”

而是：

- “single-question quote extractor”

也就是说，它更像一个：

- 输入：单个 factor / question
- 输出：该 question 对应 quote 列表

这个变化非常关键，因为它解释了为什么：

- 训练效果明显比 `v3` 好
- 但推理时如果还指望“一次生成全 schema”
- 就会重新碰到 recall 和模板塌缩问题

## 当前对 `8B + v4` 的判断

切到 `Qwen3-8B` 后，当前真实观察是：

- 输出格式已经明显稳定
- factor / question_key 能大体对齐
- 原文 quote 选择能力显著强于 `4B`
- 不再稳定塌到固定错误 factor

但当前仍然存在两个现实约束：

1. 常常只输出 1 条最稳的 quote
   - GT 里可能有 2 到 3 条
   - 这更像 recall 不足，不像方向错误

2. 如果直接让它一次性做完整 schema
   - 仍然不如单 question 稳

所以当前更合理的工程形态是：

- 训练时拆成 single-question
- 推理时循环所有问题并聚合结果

## `v5` 的核心变化

在 `v4` 之后，当前又补了一条 `v5` 路线：

- builder: `/projects/p32908/nlp_code/build_sft_extract_v5.py`
- trainer: `/projects/p32908/nlp_code/train_sft_extract_v5.py`

`v5` 不再是：

- 单个 `question`

而是：

- 单个 `factor`
- 对应一组该 factor 下的 candidate questions

也就是说，`v5` 的输入结构已经变成：

- 一个 factor block
- 多个该 factor 下的问题
- 模型在这个 factor 范围内决定：
  - 哪些问题有答案
  - 哪些问题没有答案
  - 是否需要返回多条 distinct quotes

`v5` 的关键设计目标是：

1. 强化多条 quote recall
2. 允许无答案输出空 `extractions`
3. 把 factor-specific 提示放到 prompt 末尾
   - 这样前面大段 `Text:` 更适合做 KV cache 复用
4. 保留 `v4` 已经学到的 JSON 外壳能力

当前 `v5` 还有一个重要工程选择：

- 默认不是从 base 冷启动
- 而是从 `v4` adapter warm start

更具体地说：

- dataset:
  - `/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5`
- output:
  - `/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out`
- warm start source:
  - `/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v4_out`

这样做的原因是：

- `v4` 已经学会了 JSON 外壳和基本 quote 抽取
- `v5` 更像能力强化，不像从头重新学格式

## 当前推荐推理方式

当前已经补了一条更务实的推理路线：

- 单 question 抽取：`/projects/p32908/nlp_code/eval_try_sft.py`
- 全 schema 聚合：`/projects/p32908/nlp_code/infer_extract_aggregate.py`

`infer_extract_aggregate.py` 的作用是：

- 读取一篇 raw filing
- 从当前 extract dataset 中抽出全部唯一的 `factor/question_key`
- 对每个 question 单独调用当前 extract 模型
- 收集全部 `extractions`
- 去重后合并成最终大 JSON
- 允许单题返回空结果
- 聚合后默认只保留 confidence 前 20%

这条路线的核心思想是：

- 模型端保持 single-question 稳定性
- 工程端负责 full-schema 聚合

不要再默认要求模型一次性学完整 schema 输出。

## 关于 `sglang` 的现实判断

当前这条 extract 线还额外验证出一个工程判断：

- `sglang` 不是“模型扔进去就不用管”的黑盒组件。

尤其对 `task1.2 / Miner / extract` 这种任务，更要谨慎：

- prompt 很长
- 输出是嵌套 JSON
- `original_quote` 也可能很长
- 任务本质更像“长上下文定位 + 原句复制”

这类任务比 `task1.3 / Auditor` 或 `DPO` 风格输出更容易暴露服务端链路问题，例如：

- tokenizer / chat template patch 的兼容性分叉
- decode 参数过松导致 repetition / “吐泡泡”
- JSON 提取逻辑对长输出不够稳
- server 端实际挂载模型与调用侧假设不一致

所以当前更务实的结论是：

- `Auditor / DPO` 可以继续优先使用 `sglang`
- `Miner / extract` 如果目标是稳定产出，优先使用本地 `unsloth + adapter` 的离线全量推理

不要把 `sglang` 默认当成“接上就稳定”的基础设施。

当前这个脚本还支持两种 raw filing 入口：

1. 直接传 `--raw-file`
2. 传 `--ticker --form --date`

后者会自动去 `MDA_Raw/<ticker>/<form>/` 下定位真实原文件。

## 当前默认路径

当前 extract 线常用路径是：

- v4 dataset:
  - `/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v4`
- v4 adapter:
  - `/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v4_out`
- v5 dataset:
  - `/scratch/xla2767/hold2/data/nlp/hf_extract_sft_v5`
- v5 adapter:
  - `/scratch/xla2767/hold2/data/nlp/qwen3_8b_extract_sft_v5_out`

当前常用脚本是：

- v4 builder:
  - `/projects/p32908/nlp_code/build_sft_extract_v4.py`
- v4 trainer:
  - `/projects/p32908/nlp_code/train_sft_extract_v4.py`
- v5 builder:
  - `/projects/p32908/nlp_code/build_sft_extract_v5.py`
- v5 trainer:
  - `/projects/p32908/nlp_code/train_sft_extract_v5.py`
- single-question eval:
  - `/projects/p32908/nlp_code/eval_try_sft.py`
- full-schema aggregate inference:
  - `/projects/p32908/nlp_code/infer_extract_aggregate.py`

## 当前最重要的项目知识

后续新 session 接手时，先记住下面几点：

1. `extract` 线当前更像 retrieval-style quote extraction，不像 summarization。
2. `4B` 在旧版大样本范式下很容易塌成模板记忆。
3. `v4` 的核心不是“换个脚本名”，而是把任务拆成 single-question。
4. `v5` 的核心不是回到“大一统 schema 输出”，而是“单 factor + 多问题 block + 允许无答案”。
5. teacher 数据本身存在重复 quote 和高频 qkey 偏斜，所以 builder 侧去重和重平衡是必要的。
6. 当前更推荐的交付形态不是“一次生成全 schema”，而是“逐 question 生成，再聚合”。
