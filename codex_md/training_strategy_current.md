# Current Training Strategy

这份文档记录当前项目默认采用的训练策略，不是长期愿景，也不是 worklog。

它的用途是让新的 session 进入后，立刻知道：

- 当前训练到底分几步
- 每一步在学什么
- 当前已经做到哪一步
- 下一步该怎么升级

## 当前推荐的三步策略

当前最推荐的训练顺序是：

1. `SFT`
2. 当前规则版的“笨 DPO”
3. 更强的 `PRM + RM + confidence` 驱动的“聪明 DPO”

## Step 1: SFT

目标：

- 学会稳定输出格式
- 学会 `task13` 的基本推理形态
- 尽量贴近 GPT teacher 的风格

当前入口：

- dataset: `/scratch/xla2767/hold2/data/nlp/hf_cot_sft`
- script: `/gpfs/projects/p32908/nlp_code/train_sft.py`

当前理解：

- 这一步更像“学会怎么说”
- 不直接优化 stock return
- 主要吸收 teacher 的结构、CoT、格式和任务定义

当前真实输出契约已经更新成：

- `<think> reasoning </think>`
- `{"sentiment_label": "...", "confidence_score": ...}`

也就是说，当前 SFT 的重点不只是“输出 JSON”，而是：

- 让模型先在 thinking block 里展开推理
- 再在最终 JSON 里收口到下游真正需要的结构化字段

## Step 2: 当前规则版“笨 DPO”

目标：

- 在不引入昂贵 online 采样的前提下，先让模型开始接触“投资目标”
- 用离线 preference 数据做一个稳定的过渡阶段

当前入口：

- dataset: `/scratch/xla2767/hold2/data/nlp/hf_cot_dpo`
- script: `/gpfs/projects/p32908/nlp_code/train_dpo.py`

当前 DPO 数据不是单模板 reject，而是多候选打分版：

- `teacher_full`
- `teacher_concise`
- `return_aligned`
- `conservative`

然后对每个候选分别打：

- `teacher_quality`
- `grounding`
- `return_alignment`

再选最高分做 `chosen`，最低分做 `rejected`。

当前这些 `chosen / rejected` 也已经对齐到和 SFT 一样的 thinking-block 格式，  
不再继续使用旧的纯 JSON completion。

当前这一步的真实含义是：

- 还不是“真正聪明的 PRM/RM 驱动偏好学习”
- 但已经不是最早那种“teacher 对上一个随手模板”的弱版本
- 它已经能表达“teacher / grounding 更重要，return 次之”的偏好

当前注意点：

- 这一步不应训太久
- 它的价值主要是过渡和 warm start
- 如果训得太久，模型可能会过拟合当前规则候选的表达方式
- 当前 reward model 的离线表现也说明：
  - easy pair 基本都来自 `*_over_conservative`
  - 难点主要集中在 `return_aligned_over_teacher_full`
  - 所以后续是否继续做 DPO，不应只看整体 pairwise accuracy，还应优先看这类 hard pair
- 独立抽样检查还说明：
  - 当前 DPO 本体没有明显格式性崩坏
  - 但在某些带 mitigation narrative 的风险因子上会比 GT 更偏乐观
  - 因此 DPO 当前更像“可用但带方向性偏移的过渡模型”

## Step 3: 更强的“聪明 DPO”

目标：

- 让偏好学习不再依赖手工规则分数
- 用更真实的质量判断替代当前离线启发式

当前推荐理解：

- `PRM` 负责：
  - teacher-quality
  - rationale quality
  - grounding
  - 格式和证据约束
- `RM` 负责：
  - stock return
  - excess return
  - 和投资目标直接相关的标签一致性
- `confidence` 作为辅助稳定项

如果要把这一步真正做出来，最自然的升级方式是：

1. 同一个 prompt 在线采样多个候选
2. 用 `PRM + RM + confidence` 分别打分
3. 组合成总分
4. 构造新的在线 DPO pair
5. 再继续做更强一轮 DPO

## 当前已经有的模块

- `SFT dataset`
- `DPO dataset`
- `SFT trainer`
- `DPO trainer`
- `21d return / excess return` 贴标逻辑
- 多候选离线打分版 DPO builder
- `SFT full eval`
- `DPO full eval`
- `SFT merged inference`
- `DPO merged inference`

## 当前还没有完全独立出来的模块

- 单独的 `PRM trainer`
- 单独的 `PRM dataset`
- 真正在线采样的 candidate generator
- 用真实 `PRM / RM` 分数替代当前启发式分数的 online DPO pipeline

## 模型角色的当前推荐理解

如果当前选择：

- base: `Qwen/Qwen3-8B`
- reward model: `Skywork/Skywork-Reward-V2-Qwen3-4B`

那么更合理的理解是：

- `Qwen3-8B` 做当前主模型
- `Skywork Reward` 更接近 `RM / judge`
- 如果想让“PRM 权重大”这件事变成真实工程结构，最好再补一个偏 rationale / grounding 的独立 PRM

也就是说：

- `base + RM` 现在已经基本齐
- `PRM` 如果只想先用启发式代理，当前版本已经能近似表达
- `PRM` 如果想变成独立模块，后面还需要单独训练或引入

## Step 4: 在线 DPO scaffold

当前仓库里已经补了一个最小可跑的在线 DPO 脚手架：

- `train_online_dpo.py`

它做的事情是：

1. 从本地 prompt 池里抽样
2. 用当前 policy 生成多个候选
3. 用 `RM / PRM / confidence` 打分
4. 选出 `chosen / rejected`
5. 用这批 on-policy pairs 触发一小段 DPO 更新

这一步的定位是 scaffold，不是完整在线 RL：

- 还没有做复杂的 actor-critic
- 还没有做真正的 preference replay 管线
- 还没有把 PRM/RM 训练本身纳入闭环

但是它已经把最关键的结构搭起来了：

- on-policy candidate generation
- reward-style scoring
- DPO-style update loop

如果后面要升级成真正的 online preference learning，这个入口可以直接往上接。

## 当前对 GRPO 的务实判断

当前 GRPO 路线已经有最小脚手架，但要明确：

- 它的 train/eval 成本都显著高于 DPO
- 主要瓶颈通常不是显存，而是 rollout 时间
- 对当前这套任务，GRPO 更适合作为高成本增量实验，而不是默认主线

因此当前更务实的策略是：

1. 先把 SFT / DPO 结果看透。
2. 如果仍要继续试 GRPO，优先用小 eval、小 rollout、短跑 checkpoint 做 sanity check。
3. 不要把训练内 sample callback 的脏输出直接当成模型主行为结论。

## 当前对 merged + serving 的务实判断

最近已经验证：

- merged 模型不是天然就可靠，merge 路线必须和当前训练契约一致
- 修过的 `qlora_merge.py` 产物（例如 `sft_merged_v2`）已经能正常使用
- 对批量 test set inference，`sglang` 的吞吐明显优于本地单条 HF/Unsloth 推理

因此当前更合理的分工是：

- 训练与小样本 sanity check：继续用 Unsloth / 本地脚本
- 大规模 full test inference：优先 merge 后走 `sglang`
