# A2-P：Profiling 与性能分析

> 状态：发布候选稿，请勿提交。题面版本 `26.1.4-rc.3`。
>
> `A2-P` 是 Stanford A2 的 Profiling 子作业，只覆盖 Profiling 相关内容，不包含
> activation checkpointing、Triton kernel、DDP、optimizer state sharding、FSDP
> 或并行策略分析。上游 PDF 中超出本页范围的题目不计入 `A2-P`。
>
> 上游来源为
> [stanford-cs336/assignment2-systems 固定快照](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736)，
> [原版 PDF](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)
> 固定到 `26.1.4` 对应的
> [starter commit `ca8bc81a59b70516f7ebb2da4808daade877c736`](https://github.com/stanford-cs336/assignment2-systems/commit/ca8bc81a59b70516f7ebb2da4808daade877c736)。实验室版提交目录、Markdown 报告、
> 文件大小、公开性和 PR 规则以本页为准；与上游要求冲突时，以本页为准。
>
> 组织内操作指南：
> [CS336 Assignment 2 Systems：Profiling（性能分析）实验指南](https://acnc6zeentra.feishu.cn/docx/D3omdgl6NocdKNxNvc5cW7KJnHd)。
> 链接是公开元数据，正文仍须保持组织内可见，不得开启互联网公开访问。

本作业要求建立完整的“测量—定位—解释”链路：先正确测量一个训练 step，再把时间归因到
CUDA kernel 和模型阶段，最后解释混合精度与显存峰值。重点不是只给出一个更快的数字，
而是让报告中的每个结论都能回到命令、配置和轻量原始数据。

评分标准与核验方式见 [`EVALUATION.md`](EVALUATION.md)。开始前必须阅读
[公开性与提交规则](../../docs/submission-rules.md)。

`A2-P` 纳入上游的五个小题：

| 上游 problem | 本页位置 |
| --- | --- |
| `benchmarking_script` | 任务一：End-to-End Benchmark |
| `nsys_profile` | 任务二：Compute Profiling |
| `mixed_precision_accumulation` | 任务三：四种累加实验 |
| `benchmarking_mixed_precision` | 任务三：FP32 与 BF16 autocast |
| `memory_profiling` | 任务四：Memory Profiling |

题面不复制作业答案、预填数字或上游给出的待运行代码。涉及四段指定代码、ToyModel 和模型规模
定义时，以固定 commit 的原版 PDF 与 starter 为准；提交与评分边界仍以本页为准。

## 1. 学习目标

完成后，你应当能够：

1. 正确处理 CUDA 异步执行、warm-up 和测量边界；
2. 使用 Nsight Systems 或 `torch.profiler` 采集训练轨迹，并区分系统级 kernel 证据与
   框架级 op；
3. 比较 FP32 与 BF16 autocast 的 dtype、时间、数值和显存差异；
4. 使用 PyTorch memory history 或 `torch.profiler` 的 memory profiling 解释峰值显存；
5. 用公开、脱敏、体积受控的 Markdown 报告复现实验结论。

## 2. 固定工作目录与版本

`SummerQuest-2026` 与上游工作仓库必须保持同级：

```text
<父目录>/
├── SummerQuest-2026/
└── assignment2-systems/
```

在 SummerQuest 仓库根目录执行：

```bash
git clone https://github.com/stanford-cs336/assignment2-systems.git ../assignment2-systems
git -C ../assignment2-systems checkout ca8bc81a59b70516f7ebb2da4808daade877c736
git -C ../assignment2-systems switch -c a2-p/<你的-GitHub-ID>
git -C ../assignment2-systems rev-parse HEAD
```

最后一条命令必须输出上述固定 commit。实现、依赖、虚拟环境和大型 profiler 原始文件都留在
`../assignment2-systems`；不要把上游仓库、公共 tests、依赖锁、数据、trace 或 snapshot
整体复制进 SummerQuest。

`../assignment2-systems` 只是本地执行目录，不是可提交的文档链接。报告如需引用上游源码、
PDF 或其他仓库外资料，必须使用可公开访问的 GitHub HTTPS 绝对 URL，并优先固定到上述
commit；不得写入本机绝对路径、`file://` 链接或工作区外的相对链接。

在上游工作仓库根目录创建：

```text
assignment2-systems/
├── profiling/
│   ├── benchmark.py
│   ├── nvtx_ranges.py
│   ├── memory_snapshot.py
│   └── summarize.py
├── results/                    # 本地原始结果，不直接整体提交
└── writeup.md                  # 可选工作稿；最终报告写入 SummerQuest 的 README.md
```

参数名称可以不同，但 `benchmark.py` 至少要表达：

- `model-size`、`batch-size`、`context-length`；
- `forward`、`forward_backward`、`train_step` 三种 mode；
- `warmup`、`steps`、`dtype`、`seed` 和 `output`。

每次运行都要把完整命令、实际配置、硬件/软件版本和结果路径写入 metadata。硬件信息只保留
公开且必要的型号与版本，不记录主机名、IP、用户名或内部目录。

## 3. 创建提交目录

已有个人目录的同学，在 SummerQuest 根目录运行：

```bash
python3 scripts/create_assignment.py --name '<同学真名>' --assignment A2-P
```

脚手架会校验固定兄弟仓库，并创建：

```text
students/<同学真名>/assignments/A2-P/
├── README.md                         # 必交：公开 Markdown 主报告
├── submission/
│   └── profiling/
│       └── **/*.py                   # 必交：自己编写的测量与汇总代码
├── results/                          # 必交：轻量、脱敏、机器可读的汇总
│   ├── benchmark.csv
│   ├── profile/
│   │   ├── trace_summary.csv
│   │   └── run_metadata.json
│   ├── mixed_precision.json
│   └── memory/
│       ├── peaks.csv
│       └── run_metadata.json
└── assets/
    └── *.{png,jpg,jpeg,webp,svg}     # 必交至少 3 张关键截图，必须被 README 引用
```

完成或更新 `../assignment2-systems/profiling/` 后运行：

```bash
python3 scripts/sync_a2p_submission.py --name '<同学真名>'
```

同步脚本只复制 `profiling/**/*.py`，不会复制上游代码、公共测试、`results/`、trace、
snapshot 或依赖文件。`results/` 中的轻量汇总和 `assets/` 中的压缩图片由你确认脱敏后
放入个人 `A2-P` 目录。

## 4. 任务一：End-to-End Benchmark

实现统一 benchmark 入口，并至少支持：

| mode | 必须包含 | 计时边界要求 |
| --- | --- | --- |
| `forward` | forward；通常使用 `no_grad` | 不含 loss、backward 和 optimizer |
| `forward_backward` | forward、loss、`backward()` | 每步清理梯度，不能跨 step 累积 |
| `train_step` | zero grad、forward、loss、backward、optimizer step | 覆盖完整训练 step |

最低实验要求：

1. 以 small 模型、batch size 4、context length 512、FP32 为统一基线；
2. 三种 mode 各执行至少 5 个 warm-up step 和 10 个 measurement step；
3. 对 `train_step` 额外比较 warm-up 0 与 warm-up 5，其余配置和计时边界保持不变；
4. 每个被测 CUDA step 后调用 `torch.cuda.synchronize()`，数据生成和初始化不得计入；
5. 保存每次 raw timing、均值、样本标准差和变异系数（CV）。

报告必须说明计时器、同步位置、warm-up 边界，以及 warm-up 前后差异的原因。只有均值而没有
raw timing 和标准差，不满足要求。

## 5. 任务二：Compute Profiling

### 5.1 六个 `train_step` trace

选择两个模型规模和三个大于 128 的二次幂 context length，形成 `2 × 3 = 6`
个配置。六个 trace 全部使用完整 `train_step`；本小题不要求额外采集
forward-only 或 forward+backward trace。每个配置只捕获一个预热后的稳定
measurement step。

可以使用 **Nsight Systems** 或 **`torch.profiler`** 完成这六个 trace。两种方案
同等接受，但六个配置必须使用同一主工具和同一测量口径。标记方式可以是
NVTX 或 `torch.profiler.record_function`，至少包含：

- `profile/warmup`、`profile/measure`；
- `forward`、`backward`、`optimizer`；
- `attention/scores`、`attention/softmax`、`attention/value`。

对六个配置中的每一个，至少保存模型、context、`train_step`、dtype、工具、
命令和本地 trace 文件名。提交的 `results/profile/trace_summary.csv` 必须包含主要
op/kernel、Calls、累计 CPU/CUDA 时间和阶段范围。报告还要选一个代表性配置，
比较 forward、backward、optimizer 与 attention 子阶段。

### 5.2 方案 A：Nsight Systems

Nsight Systems 可提供 CUDA API 到 GPU kernel 的系统级关联、kernel Calls 和 NVTX 区间。
示例命令只说明工具边界：

```bash
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=functions-trace,autogradshapes-nvtx \
  --output=results/profile/<run_name> \
  -- python profiling/benchmark.py <你的参数>

nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  --format csv \
  results/profile/<run_name>.nsys-rep
```

### 5.3 方案 B：`torch.profiler` 与 Perfetto

`torch.profiler` 是可接受的 trace 替代方案。启用 CPU/CUDA activities，用短 schedule 只捕获
一个稳定 step，并导出 Chrome trace 到本地。可以在 Perfetto 中查看 op、CUDA kernel、
线程、stream 和阶段区间。报告必须说明 `torch.profiler` 不提供与 nsys 完全相同的
系统级 CUDA API 关联；选择该方案时，不要伪造 nsys 专属字段。

无论选哪种工具，`.nsys-rep`、SQLite、完整 Chrome trace 和完整 timeline 都不进入
GitHub。只提交六个 run 的轻量汇总、metadata，以及一张能支撑关键分析的裁剪、
脱敏、压缩截图。不要提交整张桌面、完整终端或全量 Profile 视图。

## 6. 任务三：Mixed Precision

完成两组实验：

1. **累加误差**：原样运行固定版本 PDF 的 `mixed_precision_accumulation` 小题给出的
   四段写法，分别讨论低精度累加器、FP16 输入量化和 FP32 累加器的影响；报告实际输出和
   2–3 句误差解释。
2. **ToyModel 与 benchmark**：ToyModel 必须使用 CUDA BF16 autocast，记录参数、第一层
   输出、LayerNorm 输出、logits、loss 和 gradient dtype；在相同 batch、context、
   warm-up 与 steps 下比较 FP32 和 BF16 autocast 的时间、峰值显存和数值趋势。

累加误差小题仍保留上游固定的 FP16 输入/累加器对照；这与 ToyModel 和语言模型使用
BF16 autocast 是两组不同的实验，报告中不得混淆。

报告必须区分“输入先被量化造成的误差”和“累加器精度造成的误差”，并说明 reduction、
LayerNorm、Tensor Core 与动态范围对结果的影响。不要预填或照抄固定测量数字。

## 7. 任务四：Memory Profiling

最低实验矩阵为 XL 模型、context 128 与 2048，分别采集 forward-only 和完整
`train_step`：

1. warm-up 完成后再开启 PyTorch memory history；
2. 每个配置保存独立 snapshot，并用 memory visualizer 读取；
3. 报告 active、allocated、reserved 和峰值，不混用统计口径；
4. 推导 residual stream tensor 的理论大小，并与最大 allocation、stack trace 和阶段峰值对照；
5. 至少提交两张脱敏后的 Active Memory Timeline 截图；
6. 使用 PyTorch memory history，或开启 `profile_memory=True` 的 `torch.profiler`，
   按 TransformerBlock 解释 saved residual 的释放与 gradient 的产生。

如果 XL/context 2048 在 batch size 1 仍 OOM，保留失败配置、阶段、异常类型和峰值摘要，
再按 XL/context 1024、Large/context 2048 的顺序尝试。不得静默缩小配置后仍把结果标成
XL/context 2048。由平台、CUPTI 或管理员导致的阻塞应在飞书补充文档中记录并联系助教。

## 8. Markdown 报告要求

最终主报告固定为个人 `A2-P` 目录下的 `README.md`。书面分析、命令、表格和图表统一使用
Markdown；不提交 PDF、Office 文档、notebook 或 notebook 导出文件。本作业保留上游原始
分值，总分 **16 分**，不归一化为 100 分。

报告必须包含：

1. 完成范围、未完成项、题面版本和固定 starter commit；
2. 公开、脱敏的 GPU、驱动、CUDA、PyTorch 与工具版本；
3. benchmark 命令、配置、raw timings、均值、标准差、CV 和 warm-up 对照；
4. 六个 `train_step` Profile 配置、所选工具、阶段标记、op/kernel Calls、CPU/CUDA
   时间汇总、代表性 timeline 与归因解释；
5. mixed precision 的实际输出、dtype 表、时间/显存对照和误差分析；
6. memory 峰值表、至少两张时间线、最大 allocation 和 residual/gradient 分析；
7. OOM、CUPTI、工具或资源限制，以及仍可复现的最小命令；
8. 组织内公开的飞书补充文档链接。

报告中的每个数字都必须能回到 `results/` 的一行数据、一个 metadata 文件或一条明确命令。
图片必须使用相对路径并包含有意义的 alt text，不能只粘贴无法搜索或无法解释的截图。

## 9. 文件与附件限制

为保证公开仓库可审查、可 clone，`A2-P` 使用比 GitHub 平台上限更严格的规则：

| 范围 | 限制 |
| --- | ---: |
| 学生目录内任意单文件 | 不超过 5 MiB（仓库统一硬限制） |
| `A2-P` 的 `README.md` | 不超过 1 MiB |
| `results/` 与 `assets/` 公开附件合计 | 不超过 2 MiB |

只允许：

- `submission/profiling/**/*.py`；
- `results/**/*.{csv,json,jsonl,md,txt}`；
- `assets/**/*.{png,jpg,jpeg,webp,svg}`。

明确禁止提交：

- `.nsys-rep`、SQLite、memory snapshot、pickle、完整 Chrome trace；
- 压缩包、数据集、模型权重、checkpoint、虚拟环境、缓存和依赖锁；
- PDF、Office 文档、notebook 与 notebook 导出；
- 未裁剪的终端截图、主机名、IP、用户名、内部路径、UUID、进程列表和任何凭据。

附件指 `results/` 与 `assets/` 中的轻量汇总、metadata 和图片；`README.md` 与
`submission/` 代码不计入 2 MiB 附件限额。截图应先裁剪到关键时间段或表格，再用
PNG 压缩、WebP 或其他无损/高质量方式缩小。Profile 只提交支撑结论的关键部分。

大型 profiler 原始文件默认留在个人工作目录，不作为提交物。助教抽查时再按指定的组内
受控方式提供；不要为了“证明做过”把大型原始文件上传到公开 GitHub 或随意附在飞书正文。

## 10. 提交前自检与 PR

```bash
python3 scripts/sync_a2p_submission.py --name '<同学真名>'
python3 scripts/validate_repo.py
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

一个 PR 只能修改一名同学的
`students/<同学真名>/assignments/A2-P/`。分支使用 `a2-p/<GitHub-ID>`，PR 标题使用
`[A2-P] 姓名 - 简短说明`，commit 使用 Conventional Commits，例如：

```text
feat(a2-p): submit 张三 profiling report
```

## 11. 最终验收清单

- [ ] 固定 starter commit 正确，工作仓库位于 `../assignment2-systems`。
- [ ] 三种 benchmark mode、同步、warm-up 和统计口径完整。
- [ ] 已用 nsys 或 `torch.profiler` 完成两个模型规模、三个 context 的六个
      `train_step` trace。
- [ ] Profile 只提交轻量汇总和关键截图，未上传完整 trace。
- [ ] mixed precision 同时覆盖累加误差、dtype、时间和显存。
- [ ] memory profiling 覆盖规定矩阵或如实记录 fallback。
- [ ] `README.md` 是完整 Markdown 主报告，所有数字都可追溯。
- [ ] 代码、汇总和图片位于固定目录，文件类型与大小通过校验。
- [ ] 未提交 trace、snapshot、权重、数据、压缩包、内部信息或凭据。
- [ ] 飞书补充文档为组织内公开，未开启互联网公开访问。

常用资料：
[Nsight Systems](https://docs.nvidia.com/nsight-systems/)、
[PyTorch Profiler](https://pytorch.org/docs/stable/profiler.html)、
[PyTorch Memory Visualizer](https://pytorch.org/memory_viz)、
[Perfetto UI](https://ui.perfetto.dev/)。
