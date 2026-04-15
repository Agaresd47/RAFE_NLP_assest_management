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

- 这个脚本目前是强依赖手写路径的实验脚本。
- merge 输入目录、merge 输出目录、验证时读取的目录还没完全一致。

## Step 4: 启动推理服务

当前 `qbatch.py` 依赖外部 `sglang` server。

当前示例启动方式在 `qbatch.py` 顶部注释里已经给出。

核心含义是：

- 用 merged 模型启动本地服务
- 让 `qbatch.py` 通过 OpenAI 风格接口批量请求该服务

当前注意点：

- server 模型路径必须和 merge 产物路径一致。
- 如果 merge 输出路径改了，server 启动命令和 `qbatch.py` 里的 `MODEL_PATH` 也要一起改。

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

- 当前版本是“通路打通版”，不是最终评测框架。
- 目前只跑部分测试数据，属于 smoke test 风格。
- 这说明 eval 主流程已经存在，但还没有完全收口。

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
