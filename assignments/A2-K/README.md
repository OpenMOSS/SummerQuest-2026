# A2-K：单卡显存优化与 GPU Kernels

> 状态：发布候选稿，请勿提交。题面版本 `26.1.4-k-rc.3`。
>
> `A2-K` 是 Stanford A2 的第二个子作业，覆盖 Single-GPU Memory 与 GPU Kernels。
> 它不重复 `A2-P` 的 Profiling 任务，也不包含 DDP、optimizer state sharding、FSDP、
> tensor parallel 或多机训练；这些并行训练内容属于后续 `A2-D`。
>
> 上游来源为
> [stanford-cs336/assignment2-systems 固定快照](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736)，
> [原版 PDF](https://github.com/stanford-cs336/assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/cs336_assignment2_systems.pdf)
> 固定到 `26.1.4` 对应的
> [starter commit `ca8bc81a59b70516f7ebb2da4808daade877c736`](https://github.com/stanford-cs336/assignment2-systems/commit/ca8bc81a59b70516f7ebb2da4808daade877c736)。原版题面 PDF 的版本号为 `26.1.3`；
> `26.1.4` 只调整了代码测试。本页缩减原版的硬件规模，并把提交改为公开 Markdown
> 报告、轻量结果文件和受控附件；冲突时以本页为准。

本作业要求建立“显存权衡—正确性—性能”链路：先量化 activation checkpointing 的
显存/计算交换，再实现明确的 PyTorch attention 基线，最后完成学生自己编写的
FlashAttention-2 Triton 前向 kernel，并用可复现的 GPU 测量说明它何时更快、为什么更省
显存。只跑通 CPU 模拟、只给一张性能图或直接调用已有 fused attention，都不等于完成。

评分标准与核验方式见 [`EVALUATION.md`](EVALUATION.md)。开始前必须阅读
[公开性与提交规则](../../docs/submission-rules.md)。

## 1. 与原版 A2 的关系

`A2-K` 纳入原版的 6 道题，保留原始分值，总分 **33 分**，不归一化为 100 分：

| 上游 problem | 原始分值 | A2-K 任务 |
| --- | ---: | --- |
| `gradient_checkpointing` | 4 | 任务一：Activation Checkpointing |
| `pytorch_attention` | 2 | 任务二：显式 PyTorch Attention |
| `torch_compile` | 2 | 任务二：`torch.compile` 对照 |
| `flash_forward` | 15 | 任务三：FlashAttention-2 前向 |
| `flash_backward` | 5 | 任务四：重计算式反向 |
| `flash_benchmarking` | 5 | 任务五：正确性与性能矩阵 |

原 PDF 是算法、公式与接口定义的主要参考；本页不复制作业答案、kernel 模板的完整实现或
预填测量数字。原版的 optional Triton backward 和 leaderboard 不属于必做内容。

## 2. 学习目标

完成后，你应当能够：

1. 解释 activation checkpointing 如何用重计算换取峰值显存，并正确测量代价；
2. 区分显式 PyTorch attention、`torch.compile` 生成的 kernel 与自己编写的 Triton kernel；
3. 实现数值稳定的 online softmax 与 FlashAttention-2 tiled forward；
4. 正确保存 log-sum-exp，并用重计算完成 `dQ`、`dK`、`dV`；
5. 用多组 shape、causal/non-causal、输出与梯度误差验证正确性；
6. 在同一硬件、输入、dtype 和测量边界下比较延迟、显存与 speedup；
7. 只提交公开、脱敏、体积受控并可追溯的代码、数据和图表。

## 3. 固定环境、工作目录与版本

### 3.1 RTX 4090 24GB 标准环境

最终性能结果必须在**单张 NVIDIA GeForce RTX 4090 24GB** 上得到。其他 GPU 可以用于开发，
但不能替代本作业的正式矩阵，也不能与 4090 的数字混合计算 speedup。本页所有必做 shape
均已由教师实现按下述 23 GiB allocator 预算完整执行；该预算是资源上限，不是可改小 shape 的
理由。正式运行还必须满足：

- 只让一个进程使用一张物理 GPU；不得使用多卡、CPU/NVMe offload 或远程推理服务；
- checkpoint、compile、correctness 和 attention benchmark 各自使用新的 Python 进程串行
  执行，不得把多组模型同时留在显存中，也不得并发运行正式矩阵；
- GPU 使用默认频率和 power limit，不手动超频或降功耗；运行期间没有其他计算任务；
- 开始正式矩阵前可用显存不少于 `22 GiB`；不足时等待资源释放，不得缩小 shape；
- 每个正式实验进程必须在第一次 CUDA allocation 前把 PyTorch allocator 上限设为
  `23552 MiB`（23 GiB），并把实际 fraction 与上限写入 metadata；
- performance 统一使用 BF16；扩展正确性至少包含一个关闭 TF32 的 FP32 配置；
- 输入、模型、optimizer 和随机数据在计时区间外创建；每个被测 CUDA 区间前后正确同步；
- 显存测量前调用 `torch.cuda.reset_peak_memory_stats()`，同时报告
  `max_memory_allocated()` 与 `max_memory_reserved()`；
- attention microbenchmark 使用
  `triton.testing.do_bench(warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])`
  或严格等价的 CUDA event 流程；这里 `warmup` 和 `rep` 的单位是毫秒。

统一使用以下等价逻辑设置 allocator 上限；必须在创建 CUDA tensor、模型或 optimizer 之前
调用。真实 24GB 卡与更大显存的开发卡都使用同一个 23 GiB 预算，从而避免在大卡上无意写出
24GB 卡无法复现的实现：

```python
import torch

total_bytes = torch.cuda.get_device_properties(0).total_memory
allocator_limit_bytes = 23 * 1024**3
allocator_fraction = min(1.0, allocator_limit_bytes / total_bytes)
torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)
```

`set_per_process_memory_fraction` 只约束 PyTorch allocator，不包含 CUDA context 和驱动开销；
因此 `peak_reserved <= 23552 MiB` 是必需条件，但不能代替整卡无其他任务和 24 GiB 总上限。
如固定教师实现可以完成而你的实现触发 allocator OOM，应先排查张量生命周期、显式二次方
中间量和跨配置残留，不能静默降配。

开始正式运行前保存以下脱敏信息：

```bash
nvidia-smi \
  --query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate \
  --format=csv,noheader
```

`results/run_metadata.json` 必须记录 GPU 型号、总显存、开始时可用显存、Driver、CUDA、
PyTorch、Triton、power limit、P-state、TF32 设置、计时器、warm-up 和 measurement
设置；不得记录 UUID、主机名、用户名、内部资源编号或路径。助教复跑时以同规格 4090 为准。

### 3.2 固定工作目录与版本

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
git -C ../assignment2-systems switch -c a2-k/<你的-GitHub-ID>
git -C ../assignment2-systems rev-parse HEAD
```

最后一条命令必须输出上述固定 commit。实现、官方 tests、虚拟环境、编译缓存、完整 trace
和本地原始结果都留在 `../assignment2-systems`，不要把上游仓库整体复制进 SummerQuest。

在上游仓库中使用以下学生代码边界：

```text
assignment2-systems/
├── cs336_systems/
│   └── a2k/
│       └── **/*.py                 # A2-K 实现
├── tests/
│   └── adapters.py                 # 连接官方 tests
├── student_scripts/
│   └── a2k/
│       └── **/*.py                 # benchmark、正确性与汇总脚本
└── local_results/                  # 本地原始结果，不整体提交
```

不得把实现塞入 `tests/test_attention.py`，也不得修改公共测试来绕过 adapter。A2-K 代码必须
能通过 `tests/adapters.py` 调用。

## 4. 创建提交目录

已有个人目录的同学，在 SummerQuest 根目录运行：

```bash
python3 scripts/create_assignment.py --name '<同学真名>' --assignment A2-K
```

脚手架会校验固定兄弟仓库，并创建：

```text
students/<同学真名>/assignments/A2-K/
├── README.md
├── submission/
│   ├── cs336_systems/
│   │   └── a2k/
│   │       └── **/*.py
│   ├── tests/
│   │   └── adapters.py
│   └── student_scripts/
│       └── a2k/
│           └── **/*.py
├── results/
│   ├── correctness.json
│   ├── unit_tests.txt
│   ├── checkpointing.csv
│   ├── attention_baseline.csv
│   ├── compile_comparison.csv
│   ├── flash_benchmark.csv
│   ├── memory_evidence.json
│   └── run_metadata.json
└── assets/
    └── *.{png,jpg,jpeg,webp,svg}   # 至少 2 张，必须被 README 引用
```

完成或更新上游工作区的 A2-K 代码后运行：

```bash
python3 scripts/sync_a2k_submission.py --name '<同学真名>'
```

同步脚本只复制 `cs336_systems/a2k/**/*.py`、`tests/adapters.py` 和
`student_scripts/a2k/**/*.py`。它不会复制公共 tests、其他 A2 子作业代码、结果、编译缓存、
trace、依赖或上游仓库元数据。轻量结果和压缩图片由本人确认脱敏后放入个人 A2-K 目录。

## 5. 任务一：Activation Checkpointing

### 5.1 理论分析

回答原版 `gradient_checkpointing` 的理论部分：

1. 对由 `N` 个相同 Transformer block 组成的序列，说明在忽略计算代价时如何安排
   checkpoint，包括是否嵌套；
2. 给出峰值 activation memory 与总计算量相对 `N` 的渐近表达；
3. 提供不超过 20 行的伪代码或代码骨架，清楚标出 checkpoint 边界。

不能只写“每层 checkpoint”；必须解释保存的边界 activation、重计算区间和峰值出现位置。

### 5.2 固定实验

标准矩阵使用 Stanford medium 配置、24 层、batch size 1、context length 1024、BF16
autocast、FP32 参数和 AdamW，测量一个完整 training step。固定比较：

- 不使用 checkpoint；
- 非嵌套 checkpoint block size 为 `1`、`2`、`4`、`8` 层。

每个配置至少 3 个 warm-up step 和 5 个 measurement step。每轮测量前重置 peak memory
统计，记录 5 个 step latency 原始值、p50、peak allocated 和 peak reserved。模型、输入、
loss、optimizer、seed 与测量边界保持一致；不得把模型构造、首次编译或数据生成计入正式
step。

完成标准矩阵后，再在 context length 2048 上运行“不使用 checkpoint”和标准矩阵中
peak allocated 最低的 checkpoint 配置。baseline OOM 可以作为有效边界记录，但至少一个
checkpoint 配置必须在 23 GiB allocator 预算内成功。若该配置仍 OOM，保留失败记录并联系
助教排查；context length 1536 只能作为诊断，不得替代必做的 2048 配置。不得静默改变配置
标签或只保留成功行。

把结果写入 `results/checkpointing.csv`，至少包含：

```text
config_id,model_size,num_layers,context_length,batch_size,dtype,
checkpoint_block_size,nested,warmup_steps,measurement_steps,
step_time_ms_samples,step_time_ms_p50,peak_allocated_mib,peak_reserved_mib,status
```

报告必须解释最佳 block size 为什么不是只由 checkpoint 数量决定，并同时讨论显存收益和
重计算代价。

## 6. 任务二：PyTorch Attention 与 `torch.compile`

### 6.1 显式 PyTorch 基线

实现显式 attention 基线：`QK^T`、scale、causal mask、softmax、`PV`。基线不得调用
`torch.nn.functional.scaled_dot_product_attention`、第三方 FlashAttention 或其他会自动
派发 fused attention 的接口。

固定使用 batch size 1、BF16、causal attention，测试以下核心笛卡尔积：

- sequence length：`512`、`2048`、`8192`；
- head dimension：`64`、`128`。

每个配置记录 forward、backward 和 forward-backward 的 p20/p50/p80 latency、正式测量
设置、peak allocated、peak reserved 和状态；OOM 作为结果行保留。输入分配和随机生成
不计时。结果写入
`results/attention_baseline.csv`。

### 6.2 `torch.compile` 对照

对以下三个代表配置比较 eager 与 compiled attention：

- `(sequence=512, head_dim=64)`；
- `(sequence=2048, head_dim=128)`；
- `(sequence=8192, head_dim=128)`。

必须把首次 compile/cold-start 时间与 steady-state latency 分开。另在 Stanford small 模型、
batch size 1、context length 512、BF16 上比较 eager/compiled 的 forward、forward-backward
与完整 training step。结果写入 `results/compile_comparison.csv`。

报告不能仅以“compiled 更快”作结；需要说明 graph break、shape specialization、编译缓存和
测量稳定性。

## 7. 任务三：FlashAttention-2 前向

### 7.1 纯 PyTorch tiled 参考

实现原版要求的纯 PyTorch `torch.autograd.Function`：

- 以 tile 方式计算 attention，不调用 Triton；
- 输出 `O`；
- 保存 `Q`、`K`、`V`、`O` 与唯一一个 shape 为 `[batch, n_queries]` 的
  log-sum-exp `L`；
- 接口包含默认值为 `False` 的 `is_causal`；
- 通过 `tests.adapters.get_flashattention_autograd_function_pytorch` 暴露类对象。

该实现用于逐 tile 调试和反向参考，不以性能为目标。

### 7.2 学生编写的 Triton 前向

使用自己编写的 `@triton.jit` kernel 实现 FlashAttention-2 tiled forward：

1. query tile 由一个 program instance 负责；
2. key/value tile 在 kernel 内循环；
3. 使用数值稳定的 online softmax；
4. accumulator 与 online softmax 状态使用 FP32；
5. 输出 `O` 并保存 `L`；
6. 同时支持 causal 与 non-causal；
7. 通过 `tests.adapters.get_flashattention_autograd_function_triton` 暴露类对象。

不得把 PyTorch baseline、`scaled_dot_product_attention`、第三方 flash-attn、xFormers、
课程外已有 kernel 或远程服务包装成 Triton 实现。阅读论文和文档可以，但提交代码必须由
本人实现并能解释 tile、pointer、mask 与数值稳定性。

`TRITON_INTERPRET=1` 可以用于 CPU 调试，但只证明 interpreter 路径；它不能替代真实 GPU
kernel、CUDA 官方测试或性能测量。

## 8. 任务四：FlashAttention-2 反向

按照原版公式使用重计算得到 `dQ`、`dK`、`dV`。必做版本允许使用普通 PyTorch 函数与
`torch.compile`，但必须接入 PyTorch 和 Triton 两个 `autograd.Function`，支持 causal 与
non-causal，并返回与输入顺序一致的梯度。

在真实 CUDA GPU 上运行：

```bash
uv run pytest tests/test_attention.py -v
```

报告测试数量、通过/失败/跳过数量、GPU 型号、命令和 commit。没有 CUDA 时被跳过的 Triton
测试不能写成“通过”。将脱敏后的测试输出保存为 `results/unit_tests.txt`。

自定义 Triton backward 是可选扩展，不计入必做分数，也不能弥补前向正确性缺失。

## 9. 任务五：正确性与性能矩阵

### 9.1 扩展正确性

除官方 tests 外，至少覆盖：

- 3 个随机 seed；
- head dimension `32`、`64`、`128`；
- causal 与 non-causal；
- forward output、log-sum-exp、`dQ`、`dK`、`dV`。

记录 shape、dtype、最大绝对误差、最大相对误差、容差和 pass/fail。结果写入
`results/correctness.json`。至少一个正确性配置使用 FP32；性能矩阵统一使用 BF16。

### 9.2 固定性能矩阵

在同一张 GPU 上比较：

1. 显式 eager PyTorch attention；
2. compiled PyTorch attention；
3. 学生 Triton FlashAttention-2。

固定 batch size 1、BF16、causal。核心矩阵要求三种实现全部参加，使用：

- sequence length：`512`、`2048`、`8192`；
- head dimension：`64`、`128`；
- phase：forward、backward、forward-backward。

另做长序列边界矩阵：sequence length `16384`、head dimension `64` 和 `128`、三种 phase，
至少比较 eager PyTorch 与学生 Triton；compiled PyTorch 在该边界为可选。即使 eager OOM，
也必须保留失败行并继续尝试学生 Triton，不得缩小 shape。

按 3.1 的固定 `do_bench` 协议测量。每行记录 implementation、shape、dtype、phase、
p20/p50/p80 latency、peak allocated、peak reserved、相对同 shape eager 的 speedup、
status，以及 Triton 的 query/key tile、num warps 和 num stages。结果写入
`results/flash_benchmark.csv`。只有 implementation 之外的所有条件相同且两行都成功时
才能计算 speedup；不得跨 GPU、跨 shape、跨 dtype、跨 causal 设置或使用 OOM 行计算。

## 10. Markdown 报告与必交结果

最终主报告固定为个人 A2-K 目录下的 `README.md`，不提交 PDF、Office、notebook 或
notebook 导出。报告必须包含：

1. 完成范围、未完成项、题面版本和固定 starter commit；
2. 公开、脱敏的 RTX 4090 24GB、空闲显存、Driver、CUDA、PyTorch、Triton、power limit、
   P-state、TF32 与编译配置；
3. checkpoint 理论、代码骨架、固定矩阵和显存/时间权衡；
4. 显式 PyTorch attention 与 compiled attention 的边界和结果；
5. Triton forward 的 tile、online softmax、mask、精度和保存张量设计；
6. 官方 GPU tests 与扩展正确性结果，明确区分 pass、fail、skip；
7. 核心性能矩阵、16384 长序列边界、OOM/编译失败和至少两张图；
8. 每个关键数字对应的轻量结果文件与最小复现命令；
9. 组织内公开的飞书补充文档链接。

`results/run_metadata.json` 至少记录 commit、seed、命令，以及 3.1 规定的硬件和测量字段。
不得记录主机名、用户名、IP、内部路径、GPU UUID、进程列表或凭据。

`results/memory_evidence.json` 必须汇总所有正式进程的最高 `peak_allocated_mib`、
`peak_reserved_mib`、allocator limit/fraction、24 GiB 硬上限与 `within_24gib`。如果有条件
额外采集 `nvidia-smi` 进程峰值，可以作为补充字段，但不要提交包含进程列表或内部标识的
原始采样日志。

最小结构如下；数值必须来自本人的正式矩阵，不能照抄示例占位符：

```json
{
  "allocator": {
    "allocator_fraction": 0.0,
    "allocator_limit_mib": 23552
  },
  "hard_limit_mib": 24576,
  "pytorch_peak_allocated_mib": 0.0,
  "pytorch_peak_reserved_mib": 0.0,
  "within_24gib": true
}
```

## 11. 文件与附件限制

| 范围 | 限制 |
| --- | ---: |
| 学生目录内任意单文件 | 不超过 5 MiB |
| A2-K `README.md` | 不超过 1 MiB |
| `results/` 与 `assets/` 公开附件合计 | 不超过 2 MiB |

只允许：

- `submission/cs336_systems/a2k/**/*.py`；
- `submission/tests/adapters.py`；
- `submission/student_scripts/a2k/**/*.py`；
- `results/**/*.{csv,json,jsonl,md,txt}`；
- `assets/**/*.{png,jpg,jpeg,webp,svg}`。

明确禁止提交：

- `.nsys-rep`、Chrome trace、memory snapshot、pickle、SQLite；
- Triton/PyTorch compile cache、PTX、CUBIN、共享库、wheel；
- 数据、模型权重、checkpoint、虚拟环境、依赖锁和上游 `.git`；
- 压缩包、PDF、Office、notebook 与 notebook 导出；
- 未裁剪终端截图、内部主机名、IP、用户名、路径、UUID、进程信息和任何凭据。

附件指 `results/` 与 `assets/` 中的轻量汇总、metadata 和图片；`README.md` 与
`submission/` 代码不计入 2 MiB 附件限额。图片应裁剪到关键曲线或表格并压缩，完整 benchmark
日志和逐秒显存采样保留在个人工作目录，助教抽查时再按指定的组内受控方式提供。

## 12. 提交前自检与 PR

```bash
python3 scripts/sync_a2k_submission.py --name '<同学真名>'
python3 scripts/validate_repo.py
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

一个 PR 只能修改一名同学的 `students/<同学真名>/assignments/A2-K/`。分支使用
`a2-k/<GitHub-ID>`，PR 标题使用 `[A2-K] 姓名 - 简短说明`，commit 示例：

```text
feat(a2-k): submit 张三 memory and kernels report
```

## 13. 最终验收清单

- [ ] 固定 starter commit 正确，工作仓库位于 `../assignment2-systems`。
- [ ] 所有正式结果来自单张 RTX 4090 24GB，开跑前可用显存不少于 22 GiB。
- [ ] 各正式脚本串行、独立进程执行，首次 CUDA allocation 前设置了 23 GiB allocator 上限。
- [ ] checkpoint 的 1024 标准矩阵与 2048 边界实验完整，OOM/fallback 如实记录。
- [ ] PyTorch 基线是显式 attention，没有调用已有 fused attention。
- [ ] pure PyTorch tiled 与学生 Triton forward 均通过对应正确性检查。
- [ ] Triton forward 包含真实 `@triton.jit` kernel、online softmax 和 causal mask。
- [ ] PyTorch/Triton 两个 autograd path 都能返回正确的 `dQ`、`dK`、`dV`。
- [ ] 官方 GPU tests 没有把 skip 写成 pass。
- [ ] 核心矩阵与 16384 边界矩阵使用同硬件、同输入、同 dtype、同 causal 和同测量边界。
- [ ] README 中每个关键数字都能回到 `results/` 或明确命令。
- [ ] `memory_evidence.json` 证明 peak reserved 不超过 23552 MiB，并如实记录 24 GiB 判定。
- [ ] 至少两张图片被 README 引用，文件类型和大小通过校验。
- [ ] 未提交缓存、binary、trace、权重、数据、压缩包、内部信息或凭据。

常用资料：
[Triton](https://triton-lang.org/)、
[PyTorch `torch.compile`](https://pytorch.org/docs/stable/torch.compiler.html)、
[PyTorch activation checkpointing](https://pytorch.org/docs/stable/checkpoint.html)、
[FlashAttention-2](https://arxiv.org/abs/2307.08691)。
