# HF Dataset Build Notes

这份文档记录当前 `task1.3` 新主线 HF dataset 的构建过程、过滤规则、删掉了什么、以及需要特别注意的点。

## 当前主线对应脚本

- SFT builder: `/gpfs/projects/p32908/nlp_code/build_sft_dataset.py`
- DPO builder: `/gpfs/projects/p32908/nlp_code/build_dpo_dataset.py`
- 共享工具: `/gpfs/projects/p32908/nlp_code/task13_dataset_common.py`
- 收益贴标: `/gpfs/projects/p32908/nlp_code/forward_returns_1m.py`

默认产物：

- SFT: 当前常用目录是 `/scratch/xla2767/hold2/data/nlp/hf_cot_sft`
- DPO: 当前常用目录是 `/scratch/xla2767/hold2/data/nlp/hf_cot_dpo`

如果脚本没有显式传 `--output-dir`，仍然要检查它是不是写到了历史的 `gpfs` 路径。  
当前实际训练入口更推荐统一到 `/scratch/xla2767/hold2`。

## SFT 是怎么构的

SFT 只围绕 `TASK13_AUDIT` JSON，也就是 `task1.3 / Auditor`。

步骤是：

1. 遍历 `Extract` 目录下所有 `*_TASK13_AUDIT.json`。
2. 从文件名恢复 `ticker / form / report_date`，并把日期统一成 ISO 格式。
3. 按年份固定切分：
   - `2015-2022 -> train`
   - `2023 -> validation`
   - `2024-2026 -> test`
4. 对每个 `factor_audits` 条目构造一条样本。
5. 把当前证据和历史上下文整理成统一 user prompt。
6. 把 `audit_result` 里的 `reasoning_chain / sentiment_label / confidence_score` 整成 assistant JSON。
7. 额外保留顶层字段，方便后面审计和与收益表对齐：
   - `ticker`
   - `form`
   - `report_date`
   - `parsed_report_date`
   - `factor`
   - `evidence_count`
   - `historical_count`
   - `teacher_label`
   - `teacher_confidence`

要注意区分：

- builder 落盘的 SFT dataset 仍然保存原始 assistant JSON
- 但 `train_sft.py` 训练时会把最后一条 assistant 目标重写成：
  - `<think>`
  - `reasoning_chain`
  - `</think>`
  - `{"sentiment_label": "...", "confidence_score": ...}`

也就是说：

- dataset 原始监督格式和训练时实际 tokenized 目标格式，已经不再完全等价
- 后续新的 DPO / eval / online rollout 都应对齐到 thinking-block 版本，而不是继续假设 assistant 只输出单个 JSON

## DPO 是怎么构的

DPO 的起点不是 raw 文本，而是已经可以做 SFT 的 `task13` 审计样本。

步骤是：

1. 先按和 SFT 一样的方式，把 `TASK13_AUDIT` 展平成逐 factor 的样本。
2. 对每条样本保留 `ticker + parsed_report_date`，这是后面和价格表对齐的关键键。
3. 调 `attach_21d_return_and_excess(...)` 贴上：
   - `ret_1m`
   - `excess_1m`
4. 用收益桶把连续收益离散成标签：
   - `very_negative`
   - `negative`
   - `neutral`
   - `positive`
   - `very_positive`
5. 对每条样本先生成多个候选回答：
   - `teacher_full`
   - `teacher_concise`
   - `return_aligned`
   - `conservative`
6. 对每个候选分别打分：
   - `teacher_quality`
   - `grounding`
   - `return_alignment`
7. 从候选里选总分最高的做 `chosen`，最低的做 `rejected`
8. 写出 DPO 所需的：
   - `prompt`
   - `chosen`
   - `rejected`
9. 同时保留辅助审计字段：
   - `return_label`
   - `preference_source`
   - `candidate_scores`
   - `teacher_label`
   - `ret_1m`
   - `excess_1m`
   - `evidence_count`
   - `historical_count`

当前新版还有一个关键变化：

- `chosen` / `rejected` 已经统一改成 thinking-block 输出
- 不再继续写出旧版的纯 JSON `reasoning_chain`
- 当前真实格式是：
  - `<think> reasoning </think>`
  - `{"sentiment_label": "...", "confidence_score": ...}`

## 这版明确删掉了什么

### SFT 侧

- `factor_audits` 为空的文件，不会产生任何样本。
- `evidence_used` 为空的条目会被跳过。
  当前默认是 `--min-evidence 1`，所以空证据样本不会进入 HF dataset。
- 文件名无法解析出 `ticker / form / report_date` 的文件会被跳过。
- JSON 解析失败的文件会被跳过。
- 不在固定年份范围内的样本会被跳过。

### DPO 侧

- 会先继承 SFT 侧的过滤，也就是空证据样本先被挡掉。
- 无法贴上 `21d return` 的样本会被跳过。
  常见原因是：
  - 找不到 ticker
  - 找不到日期
  - 日期之后没有足够交易日
- `return_label` 贴不上时会被跳过。
- 当前默认还会额外删除 `neutral` return bucket。
  这是通过 `--drop-neutral-returns` 控制的，默认打开。

这一步的目的很直接：DPO 更需要方向性偏好，`neutral` 回报样本信号太弱，容易把 pairwise preference 搅浑。

## 你刚刚关心的“空样本有没有剔除”

结论是：`有`，而且是显式剔除，不是碰运气。

关键点：

- `build_sft_dataset.py` 里有 `--min-evidence`，默认 `1`
- 所以 `evidence_count < 1` 的 auditor 样本不会进 SFT HF dataset
- DPO builder 也是从这个过滤后的审计样本继续构建

也就是说，新版 HF 不会像“把空 prompt 也先塞进去，训练时再想办法过滤”那样处理；它是在 builder 阶段就先裁掉一批低价值样本。

## 这版有哪些小巧思

### 1. 日期字段双写

同时保留了：

- `report_date`
- `parsed_report_date`

原因不是重复，而是为了兼容现有代码里两种命名口径。  
这样 `forward_returns_1m.py`、训练脚本、后续导出脚本不会因为字段名轻微不一致就断链。

### 2. 训练标签先标准化

`task13_dataset_common.py` 会先把历史标签统一到：

- `very_negative`
- `negative`
- `neutral`
- `positive`
- `very_positive`

这样旧标签写法，比如 `Very Bad / Bad / Good / Very Good`，在新主线里不会继续扩散。

### 3. DPO 不直接拿连续收益做文本监督

这里没有试图把连续 return 硬回归进 token 目标，而是先桶化成方向标签，再构造 preference pair。  
这更稳，也更像 DPO 真正需要的监督形式。

### 4. DPO pair 来源被显式记录

保留了 `preference_source`，当前会标记类似：

- `teacher_concise_over_conservative`
- `teacher_full_over_conservative`
- `return_aligned_over_conservative`
- `return_aligned_over_teacher_full`

这样后面如果发现某类 pair 质量差，可以反查到底是哪种构造逻辑导致的。

### 5. SFT 和 DPO 都保留审计元数据

不是只留文本列，而是额外留：

- `factor`
- `ticker`
- `report_date`
- `evidence_count`
- `historical_count`

这样后面分析训练失败样本、某个 factor 表现差、或者某段年份效果崩掉时，不需要重新回原始 JSON 再做人肉定位。

## 当前已知注意事项

### 1. DPO 目前是 return-guided offline preference，不是 online RL

这条主线更准确的名字应该是：

- `CoT SFT`
- `return-guided DPO`

不是 PPO，也不是 GRPO。

### 2. 现在 DPO 不是单一 template reject，而是多候选打分版

当前 `build_dpo_dataset.py` 会先为同一个 prompt 生成多份规则候选，再做离线排序。  
这比最开始那版“teacher vs 一个模板句子”更像真实的 preference construction，但仍然属于轻量级离线版本。

当前总分口径更接近：

- `teacher_quality` 和 `grounding` 权重大
- `return_alignment` 权重次之

当前 reward model 的离线 sanity check 还说明了一件重要的事：

- `*_over_conservative` 这类 pair 很容易
- 真正难的是 `return_aligned_over_teacher_full`

这意味着当前 DPO 数据里：

- easy pair 足够多，适合先把基本 preference 学起来
- hard pair 主要集中在“teacher 风格”和“return 导向”发生冲突的时候
- 后续如果要继续改 pair construction，优先针对这类 hard pair，而不是继续增加更多 `conservative` 对照

这和“PRM 权重大、RM 次之”的直觉是一致的。

最近还观察到一个实际后果：

- 用这版 DPO 数据训出来的模型，在某些“有缓释措施的风险因子”上，
  会比 teacher/GT 更偏乐观
- 典型例子是：
  - `AAPL / structural_risks_tech.supply_chain_concentration`

这说明当前 pair construction 的一个潜在副作用是：

- 容易奖励“存在 mitigation narrative”的答案
- 即使 GT 更偏 `negative / neutral`

因此如果后续继续强化 return-guided preference，
要避免把这类“过度乐观”的偏移继续放大。

如果后面要再升级，优先方向会是：

- 引入更多候选回答来源
- 用更细的 scoring 逻辑选 `chosen/rejected`
- 把 grounding / teacher quality / return alignment 分开算分

### 3. `teacher_over_synthetic` 不等于“teacher 一定对”

这个标记只表示在当前规则下，teacher 更接近允许保留的偏好答案，不代表 teacher 在投资意义上永远正确。

### 4. 价格文件名有环境差异

`forward_returns_1m.py` 已兼容：

- `daily_prices_2010_2014.csv`
- `daily_prices_2010_2014 (1).csv`

这是因为你磁盘上的真实文件名带了 `(1)`。如果后面别人换环境，这里仍然是一个需要先检查的点。

## 当前实际样本数

当前已经落盘的默认数据集规模是：

- `hf_cot_sft`
  - train: `8451`
  - validation: `1034`
  - test: `1458`
- `hf_cot_dpo`
  - train: `6186`
  - validation: `767`
  - test: `1145`

这个 DPO 规模已经是去掉 neutral-return 后的版本。
