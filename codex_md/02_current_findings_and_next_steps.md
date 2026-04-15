# Current Findings And Next Steps

## 当前框架判断

当前结论：

- 仓库已经具备基础的 SFT 训练和 eval 框架。
- 这不是只有零散脚本，而是已经有一条清晰主链路。
- 但它目前更像“能跑通的 v0 实验框架”，还不是完全收口的工程化 pipeline。

当前主链路是：

- `build_dataset.py`: 构建 Miner / Auditor 数据集。
- `train.py`: 用 Unsloth + QLoRA 做 SFT 训练。
- `qlora_merge.py`: 将 adapter merge 回基座模型。
- `qbatch.py`: 用 merged 模型做 batch inference / eval，并导出结果。

## 已发现的关键问题

### 1. 数据入口没有统一

`build_dataset.py` 会把数据集 `save_to_disk` 到本地目录。

但 `train.py` 实际加载时，用的是固定的 Hugging Face dataset name，而不是命令行参数里的本地路径。

这说明逻辑链路已经存在，但接口还没真正接上。

### 2. 路径约定没有统一

不同脚本里混用了不同根路径和输出目录，比如：

- `/projects/...`
- `/gpfs/projects/...`
- `/scratch/...`
- 历史 Windows 路径 `E:\...`

这说明代码来源和运行环境已经发生过迁移，但没有完成统一收口。

### 3. merge 环节还不是稳定模块

`qlora_merge.py` 目前是一个强依赖手写路径的脚本。

它包含：

- 写死的 base model 路径
- 写死的 adapter checkpoint 路径
- 写死的 merge 输出路径
- 一个附带的临时验证片段

因此它代表“有 merge 这一步”，但还不是可复用模块。

### 4. eval 链路是通路打通版，不是完整评测框架

`qbatch.py` 目前已经可以完成 Miner -> Auditor -> CSV 输出的主流程。

但它仍然存在一些明显的实验态特征：

- 依赖外部启动的 `sglang` server
- 模型路径写死
- 只跑部分测试数据
- 注释与实际执行上限不完全一致

所以它证明了 eval 主流程已经存在，但还没有完全收口。

### 5. 远端计算资源是 session 级上下文，不是固定配置

当前项目的重任务执行可能依赖用户临时开出的远端计算节点。

这里真正稳定的信息不是某一个节点名，而是下面这些约束：

- 节点名和目标主机可能每次 session 都会变化。
- 新 session 不应把上一次的主机名当成项目固定配置。
- 新 session 进入后，应先确认用户是否已经开好了本次要用的远端 session / 节点。
- 用户手动交互式 SSH 登录可能正常，但 Codex 当前的非交互远程执行不一定等价，需要单独验证。
- 当前已验证的更可靠方式是：先在本地起一个 `tmux` session，再从该交互式终端里 `ssh` 到远端节点。
- 因此后续如果需要在远端节点上持续执行命令，不要默认依赖一次性的非交互 `ssh host cmd`。

## 当前优先级判断

当前最重要的，不是新增任务，而是把已有链路做一致性收口。

优先级应是：

1. 统一数据入口。
2. 统一训练、merge、eval 的模型与目录约定。
3. 去掉明显一次性实验脚本风格的硬编码。
4. 把端到端运行方式沉淀成清晰接口。

## 下一步改动应优先做什么

建议后续优先把下面几件事做好：

### A. 收口 `build_dataset.py` 和 `train.py`

目标是明确：

- 数据到底从哪里来。
- `train.py` 是否应默认加载本地磁盘数据集。
- 命令行参数是否真正生效。

### B. 收口 `train.py` 和 `qlora_merge.py`

目标是明确：

- 训练输出目录结构是什么。
- merge 脚本应该读取哪个 checkpoint。
- merge 后模型目录如何命名。

### C. 收口 `qlora_merge.py` 和 `qbatch.py`

目标是明确：

- eval 默认读取哪个 merged 模型。
- server 启动命令与 eval 脚本的模型路径是否一致。
- 是否保留 smoke test 模式和 full eval 模式两个入口。

## 这份文档的用途

这份文档不是 worklog。

它的用途是让新的 Codex session 读完后，立刻知道：

- 这个项目已经做到哪一步了。
- 当前判断是什么。
- 下一步应该优先改哪里。
