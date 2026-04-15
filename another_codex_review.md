**问题**
1. `[P1]` Auditor 的带 COT 样本违反了它自己声明的输出格式  
[`build_dataset.py:31`](E:/nlpas/stock/build_dataset.py#L31), [`build_dataset.py:180`](E:/nlpas/stock/build_dataset.py#L180)  
`SYSTEM_AUDITOR` 明确写的是 “Output only valid JSON.”，但 `sanitize_auditor_output_single(..., True)` 实际输出的是 `<think>...</think>` 再加 JSON。也就是说，如果你用 `cot` 样本训练，模型会被明确监督去违背它自己的 schema。这会伤害格式稳定性，也会让下游解析变得模糊。

2. `[P1]` Auditor 的 target key 和 prompt/schema 的描述不一致  
[`build_dataset.py:35`](E:/nlpas/stock/build_dataset.py#L35), [`build_dataset.py:176`](E:/nlpas/stock/build_dataset.py#L176), [`build_dataset.py:185`](E:/nlpas/stock/build_dataset.py#L185)  
Prompt 要求的是 `reasoning_chain`、`sentiment_label`、`confidence_score`，但实际 teacher JSON 用的是 `factor`、`sentiment`、`confidence`。这种不一致会让模型学到一个和 prompt 要求不同的 API。如果你的 trainer/evaluator 期待的是 `sentiment_label` / `confidence_score`，这就不是风格问题，而是真 bug。

3. `[P1]` Auditor 的标签词表和真实标签空间可能不一致  
[`build_dataset.py:31`](E:/nlpas/stock/build_dataset.py#L31), [`build_dataset.py:187`](E:/nlpas/stock/build_dataset.py#L187)  
System prompt 让模型输出 `Very Bad, Bad, Neutral, Good, Very Good`，但 teacher label 是直接从 `audit_result.sentiment_label` 拷贝的，没有做 remap。如果源数据实际用的是 `very_negative/negative/neutral/positive/very_positive`，那 prompt 和 target 的标签空间就是不一致的。这完全可能让模型表现得比实际更差。

4. `[P2]` Miner 的长度阈值远高于任何现实训练上下文  
[`build_dataset.py:251`](E:/nlpas/stock/build_dataset.py#L251)  
Miner 只跳过估算大于 `~259935` tokens 的文件。对于任何正常的 8k/16k/32k SFT，这意味着很多超长样本仍然会进入数据集，然后在训练时被硬截断。最轻也是浪费数据，严重的话会给训练带来噪音。如果你的工作区间是长上下文但仍有上限，这个阈值应该和真实训练 context 对齐。

5. `[P2]` `auditor_mode=both` 会为同一个 prompt 生成两个风格冲突的 target  
[`build_dataset.py:299`](E:/nlpas/stock/build_dataset.py#L299), [`build_dataset.py:310`](E:/nlpas/stock/build_dataset.py#L310)  
同一个 prompt、同一个样本，却生成两个不同 assistant target：一个带 `<think>`，一个不带。如果你把它们混在一起训练，相当于在教模型对相同输入输出两套不兼容的响应契约。只有在你后续始终只过滤保留其中一种视图时，这才勉强说得过去。

**开放问题 / 假设**
- 我这里默认 `audit_result.sentiment_label` 是源 JSON 里的标准标签，而不是已经被转成 `Very Bad/Bad/...`。
- 我默认你的下游 Task 1.3 evaluator 想要的是接近 `reasoning_chain`、`sentiment_label`、`confidence_score` 这种结构化 JSON key。
- 我没有直接检查源 JSON 内容本身，只 review 了这个 dataset builder。

**整体评价**
这个 builder 可以作为起点，但 auditor 这部分有明显的“契约问题”：prompt、标签空间、target 格式没有对齐。  
如果你准备重写自己的 pipeline，我会保留它的 split 思路和 HF `messages` 数据结构，但 auditor 的 schema 我一定会先修正，再去相信训练结果。


- 你现在的 `HF builder`
- 你现在的 `train code`

我会按“现在有什么问题 -> 必须做什么 -> 怎么搭台子 -> 上服务器先干什么”来讲。

**1. 你现在这个 HF builder 在干什么**
你现在的 builder 本质上在造两类数据：

- `miner` 数据
  - 输入：原始 MD&A 文本 + question schema
  - 输出：Task 1.2 的 extraction JSON
- `auditor` 数据
  - 输入：某个 factor 的 current evidence + historical context
  - 输出：Task 1.3 的 sentiment judgement

你现在 auditor 数据还支持两种视图：

- `with_cot`
- `no_cot`

然后你训练代码里只拿了：

- `auditor_ds["train"].filter(lambda x: x.get("cot_visibility") == "no_cot")`

这说明你训练时实际上**已经在规避一部分 builder 的坑**。

---

**2. 我现在明确发现的 builder 问题**

**A. Auditor 的 prompt 和 target 契约不一致**
这是最大的问题。

你的 system prompt 写的是类似：

- 输出 `reasoning_chain`
- 输出 `sentiment_label`
- 输出 `confidence_score`
- 输出 only valid JSON

但你 teacher target 实际写出来的是：

- `factor`
- `sentiment`
- `confidence`

如果你后面模型按 prompt 学，或者 evaluator 按另一个 schema 读，这里一定会有扭曲。

这不是风格问题，是**数据接口定义没统一**。

---

**B. `with_cot` 版本不是纯 JSON**
你的 `with_cot` target 会输出：

- `<think> ... </think>`
- 再拼一个 JSON

但 prompt 说的是 “only valid JSON”。

这意味着：

- `with_cot` 样本和 prompt 自己打架
- 如果你以后训 thinking 版本，模型会学到一套“口头说 JSON，实际上不是 JSON”的坏习惯

你现在训练用 `no_cot`，所以这个问题暂时绕过去了。  
但 builder 本身还是脏的。

---

**C. 标签词表可能不一致**
你的 auditor prompt 里写的是：

- `Very Bad / Bad / Neutral / Good / Very Good`

但 source data 里的 `audit_result.sentiment_label` 很可能是：

- `very_negative / negative / neutral / positive / very_positive`

如果真是这样，那就是：

- prompt 的标签空间是一套
- target 的标签空间是另一套

这会直接伤训练。

---

**D. `auditor_mode=both` 会制造冲突样本**
同一个 prompt、同一个样本，你会导出两个 target：

- 一个 `with_cot`
- 一个 `no_cot`

如果后面有人不小心混着训，就相当于：

- 同一个输入
- 两个不同输出协议

这会把模型训乱。

你现在自己训练时筛了 `no_cot`，所以没完全中这个雷。  
但数据源本身还是不干净。

---

**E. 你现在没有做 schema-level validation**
builder 现在基本是“读 JSON -> 拼消息 -> 保存”。

但没有系统检查：

- assistant content 能不能 parse
- key 是否统一
- label 是否在允许集合里
- confidence 是否是合法数值
- 是否有空 rationale / 空 evidence
- 是否有超长样本
- 是否有重复样本

这意味着：
**builder 能跑完，不代表 dataset 是干净的。**

---

**F. Miner 的长度门槛没和真实训练上下文对齐**
你 builder 里 miner 只跳过极端长文件，但这个阈值和你真实训练的 `MAX_SEQ_LEN` 没绑定。

你训练脚本后来用 tokenizer 真正算长度并过滤了，这一步是对的。  
但 builder 本身还是不够严格。

---

**G. 你现在把 miner 和 auditor 混训了**
你训练里是：

- `train_ds = concatenate_datasets([miner_train, auditor_train]).shuffle(seed=42)`

这不是一定错。  
但如果你当前目标是把 Task 1.3 弄稳，那这会增加变量。

建议是：

- 先单独把 auditor 训稳
- 再决定要不要 miner + auditor multitask

---

**3. 你现在的训练代码已经处理了什么**

你训练代码已经做对了两件重要的事：

**A. 只取 `no_cot`**
这避免了：

- `<think>` 和 JSON 混在一起
- 同一个 prompt 有两个冲突 target

**B. 真正按 token 长度过滤**
你不是按字符估长度，而是：

- chat template 格式化后
- tokenizer.encode
- 超过 `MAX_SEQ_LEN` 就过滤掉

这个是对的，而且比 builder 那边靠谱很多。

---

**4. 但你的训练代码没有处理的问题**

虽然你训练代码比 builder 干净，但它没有修 builder 的根本问题：

- 它不会自动把 `sentiment` 改成 `sentiment_label`
- 它不会自动把 `confidence` 改成 `confidence_score`
- 它不会自动统一 label space
- 它不会自动检查 JSON schema
- 它不会自动保证 auditor 样本是高质量的

它只是：

- 读现成 `messages`
- 套 chat template
- 开训

所以如果底层 HF dataset 已经脏了，训练脚本不会救你。

---

**5. 我建议你现在怎么重搭台子**

我建议你把整个系统拆成**四层**。

### 第一层：Canonical data contract
先把 auditor 的唯一标准格式定死。

我建议你先只做一个版本：

**Auditor no-COT canonical target**
```json
{
  "sentiment_label": "positive",
  "confidence_score": 0.78
}
```

如果你要做 reasoning 版，再单独做：

**Auditor rationale version**
```json
{
  "reasoning_chain": "简短、grounded、不要废话",
  "sentiment_label": "positive",
  "confidence_score": 0.78
}
```

重点是：

- 不要一会儿 `sentiment` 一会儿 `sentiment_label`
- 不要一会儿 `confidence` 一会儿 `confidence_score`
- 不要一会儿 `Very Bad` 一会儿 `negative`
- 不要 `<think>` 混在主输出里

**先统一一种协议。**

---

### 第二层：Dataset builder
你应该直接做一个新的 clean builder，建议产出：

- `hf_auditor_v4_no_cot`
- `hf_auditor_v4_cot`
- `hf_miner_v4`

其中 auditor 的 clean builder 必须做：

- label vocabulary remap
- key name remap
- length stats
- schema validation
- duplicate detection
- split assignment
- metadata 保留

我建议 auditor 每条样本至少保留这些字段：

- `messages`
- `ticker`
- `form`
- `report_date`
- `factor`
- `split`
- `cot_visibility`
- `source_file`
- `label_raw`
- `label_normalized`

这样后面 debug 很方便。

---

### 第三层：Sanity check
在训练前，一定单独跑一个 dataset checker。

必须检查：

- `messages` 是否都是三轮：system / user / assistant
- assistant 内容能否 parse
- JSON key 是否统一
- label 是否只来自允许集合
- confidence 是否在 `[0,1]`
- token 长度分布：p50 / p90 / p99 / max
- 各 split 样本数
- 各 label 分布
- 是否有重复 prompt
- 是否有空 evidence / 空 target

这一步非常关键。  
你现在最缺的不是 trainer，而是 **dataset QA**。

---

### 第四层：Training stack
训练建议分开跑：

**Stage 1: Auditor-only no-COT SFT**
- 先只训 auditor
- 不混 miner
- 先把 task 1.3 label/format 学稳

**Stage 2: Auditor rationale SFT**
- 如果你真要 teacher COT，再单独训 rationale 版本
- 不要混着来

**Stage 3: Multitask**
- 只有前两步都稳定，再考虑 miner + auditor 混训

---

**6. 我对你当前任务的建议路线**

你现在别上来就搞 RL。  
先把地基打稳。

### Phase A
先重建干净的 `hf_auditor_v4_no_cot`

目标：
- Task 1.3 输出格式稳定
- label 空间统一
- evaluator 好写

### Phase B
用你熟悉的脚本训 `auditor-only SFT`

目标：
- loss 稳
- val 指标正常
- 输出 schema 稳定

### Phase C
如果你确实要 COT，再做 `hf_auditor_v4_cot`

但注意：

- COT 不要是自由长作文
- 尽量短、grounded、结构化
- 最好限制长度

### Phase D
等 SFT 稳了，再考虑 return-guided

---

**7. 你上服务器后应该先做什么**

我建议执行顺序是：

1. **不要先开训**
2. 先重建或检查当前 HF dataset
3. 单独做 auditor dataset QA
4. 先训 `auditor-only no_cot`
5. 看 loss 和 val 输出是否正常
6. 确认输出 schema 稳定后，再决定要不要上 COT
7. 最后再想 DPO / return-guided

---

**8. 我对你现在局面的总判断**

你现在最核心的问题不是 trainer。  
你 trainer 其实已经够用了。

你真正要解决的是：

- **HF builder 没把 auditor 的数据契约统一好**
- **你现在的数据源里混着多种输出协议**
- **你还没对 dataset 做严格 QA**

所以最优策略不是继续修训练器，而是：

**先重做干净 builder + sanity checker，再训你自己的 auditor SFT。**

---

**一句话总结**
你现在的 builder 主要问题是：

- prompt / target schema 不一致
- `with_cot` 不是合法 JSON 协议
- label 空间可能不一致
- 缺少严格的 dataset QA
- miner / auditor 混训会让调试更难

你现在该做的是：

- 先定义唯一标准 auditor schema
- 重建干净 HF dataset
- 先跑 auditor-only no-COT SFT
- 再考虑 COT 和 return-guided

如果你要，我下一条可以直接给你一个**服务器执行 checklist**，按命令和阶段写成可操作版本。


可以。你现在这条线我建议定成一个非常干净的两阶段：

**总路线**
1. `SFT`：模仿 GPT 的 `Task 1.3` COT 和输出格式。
2. `DPO`：不用 GRPO，不做 online RL。直接做 **return-guided DPO**，把“更符合未来 return 的答案”设成 `chosen`。

这两步已经够形成一个完整闭环了。

**核心原则**
- 不碰你朋友那套旧代码。
- 不混 `miner` 和 `auditor`。
- 先只做 `auditor / task 1.3`。
- 不让 Codex 跑任何训练，只让它把代码脚手架写好。
- 第二阶段虽然你口头叫 RL，但实现上建议就是 **offline DPO based on return**。

**你最终要的工程形态**
1. `build_sft_dataset.py`
   - 从你的 HF / 本地数据构造 `task1.3` 的 SFT 数据。
   - 只产出统一 schema。
2. `train_sft.py`
   - 纯 SFT。
   - 输入是 clean dataset。
3. `eval_sft.py`
   - 导出 prediction。
   - 保证格式统一。
4. `build_dpo_dataset.py`
   - 基于 SFT 输出、teacher、return 构造 `prompt/chosen/rejected`。
5. `train_dpo.py`
   - 从 SFT checkpoint 继续训 DPO。
6. `eval_dpo.py`
   - 导出 DPO prediction。
7. `compare_returns.py`
   - 比较 baseline / SFT / DPO 的 1mo excess return。

**第一阶段：COT mimic GPT**
目标：
- 让模型先稳定学会 `Task 1.3` 的回答方式。
- 重点不是收益，而是：
  - 输出格式稳定
  - label 稳定
  - rationale 像 GPT
  - 不胡说

我建议 SFT target 统一成这一种，不要混：
```json
{
  "reasoning_chain": "...",
  "sentiment_label": "positive",
  "confidence_score": 0.78
}
```

如果你担心 COT 太长太水，就让 Codex 额外支持一个开关：
- `--rationale-style full`
- `--rationale-style concise`

这样以后你可以先训 `concise rationale` 版本。

**第二阶段：return-guided DPO**
目标：
- 不是直接模仿 GPT
- 而是让模型偏好“既合理又更符合未来 return”的输出

我建议 `chosen/rejected` 的构造逻辑是：

`chosen score = teacher_quality + grounding + return_alignment`

具体来说：
- `teacher_quality`
  - 是否接近 GPT teacher 的 label / rationale
- `grounding`
  - 是否只使用当前 evidence 和 historical context
- `return_alignment`
  - 预测 label 是否和未来 1mo excess return 的方向 / 强弱一致

然后：
- 高分答案作为 `chosen`
- 低分答案作为 `rejected`

这一步不要做 online sampling policy optimization。  
直接离线构造 pair，然后 DPO 就行。

**return reward 怎么落地**
不要直接拿连续收益回归到文本。  
先离散化成标签偏好最稳。

例如：
- 很高正收益 -> `very_positive`
- 中度正收益 -> `positive`
- 接近 0 -> `neutral`
- 中度负收益 -> `negative`
- 很高负收益 -> `very_negative`

然后做一个 `label_alignment_score`。

这样第二阶段本质是：
- GPT 教你“怎么说”
- return 教你“哪种判断更有投资意义”

**我建议 Codex 写出来的目录**
```text
task13_pipeline/
  configs/
    sft.yaml
    dpo.yaml
  src/
    schemas.py
    prompts.py
    label_utils.py
    returns.py
    dataset_sft.py
    dataset_dpo.py
    teacher_utils.py
    train_sft.py
    eval_sft.py
    train_dpo.py
    eval_dpo.py
    compare_returns.py
    io_utils.py
  scripts/
    build_sft_dataset.py
    build_dpo_dataset.py
  README.md
```

**你要 Codex 先实现的最小功能**
- `schemas.py`
  - 定义统一 output schema
- `label_utils.py`
  - 统一 label vocabulary
  - 提供 return bucket -> label 的映射
- `dataset_sft.py`
  - 把 teacher COT 数据转成 SFT 样本
- `train_sft.py`
  - Unsloth + TRL 的干净 SFT trainer
- `dataset_dpo.py`
  - 把 teacher / candidate / return 组装成 DPO pair
- `train_dpo.py`
  - 从 SFT adapter 开始做 DPO
- `compare_returns.py`
  - 吃统一 prediction CSV 做 1mo excess return 对比

**Codex 现在不需要做的**
- 不需要跑任何命令
- 不需要接旧 repo
- 不需要接你朋友的 artifact
- 不需要做 GRPO
- 不需要做 PRM
- 不需要做 miner multitask

**你给 Codex 的任务描述建议**
你可以直接这样说：

“不要运行任何代码。请从零搭一个干净的 `task1.3` 两阶段训练脚手架。  
阶段一是 `GPT COT mimic SFT`，阶段二是 `return-guided offline DPO`。  
不要使用 GRPO/PRM，不要复用旧 repo。  
请实现统一 schema、dataset builder、train/eval 脚本、DPO pair builder、以及 return compare 脚本。  
默认只做 auditor/task1.3，不混 miner。  
所有输入输出接口要清晰，代码尽量模块化，便于我后续自己接服务器数据和训练。”

**最后的判断**
你现在最对的做法就是：
- 先让 Codex把**工程骨架**写出来
- 你自己再往里接数据
- 然后你自己上服务器跑

如果你要，我下一条可以直接帮你把这段变成一份可以原样发给 Codex 的中文 prompt。