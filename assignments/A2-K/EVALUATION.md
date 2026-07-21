# A2-K 评估补充说明（批改助教）

> 本文件说明评估方式，不改变 [`README.md`](README.md) 中的任务和提交要求。A2-K 当前为
> 发布候选稿；正式发布前不接收学生提交。

## 评估原则

- 评估链路是“学生代码 → 官方/扩展正确性 → 轻量原始数据 → 表格/图 → 解释”。
- 正式 benchmark 的唯一标准环境是单张 NVIDIA GeForce RTX 4090 24GB，开跑前可用显存
  不少于 22 GiB；其他 GPU 的开发结果不能替代正式矩阵。
- FlashAttention-2 的核心是学生编写的真实 Triton kernel。包装已有 fused attention、
  只提交 PyTorch 实现或只运行 `TRITON_INTERPRET=1` 不能获得 Triton 实现分。
- CUDA 测试被 skip 不等于通过；必须区分 pass、fail、skip。
- 性能比较必须控制硬件、shape、dtype、causal、输入和测量边界。不能用不等价配置计算
  speedup，也不能因 OOM 静默删除结果行。
- 评分看真实完成度、正确性、复现性和分析质量，不奖励预填数字或无法追溯的截图。

## 评分权重

保留原版 33 个分值，不归一化：

| 部分 | 分值 | 核验重点 |
| --- | ---: | --- |
| Activation Checkpointing | 4 | 渐近分析、checkpoint 边界、固定矩阵、显存/时间权衡 |
| PyTorch Attention Benchmark | 2 | 显式基线、完整 shape 矩阵、同步、OOM 记录 |
| `torch.compile` | 2 | cold/steady-state 分离、attention 与完整模型对照 |
| FlashAttention-2 Forward | 15 | PyTorch tiled、真实 Triton kernel、LSE、online softmax、causal |
| FlashAttention-2 Backward | 5 | `dQ/dK/dV`、重计算、两个 autograd path、causal |
| Correctness and Benchmarking | 5 | 官方 GPU tests、扩展误差矩阵、等价性能对照、图表 |

报告、代码、结果和图片的公开性、可追溯性与文件限制在每部分内评分，不另设只靠排版获得的
分数。

## 硬验收边界

以下情况不能获得相应部分完整分数：

- 没有学生编写的 `@triton.jit` forward kernel：Flash forward 的 Triton 实现项记 0，
  Flash benchmark 的 Triton 对照项记 0；
- Triton CUDA tests 全部 skip：只能确认非 GPU 路径，不能确认 GPU 正确性与性能；
- 调用 `scaled_dot_product_attention`、第三方 flash-attn、xFormers 或复制已有 kernel
  冒充实现：相关实现与 benchmark 项记 0，并转交课程诚信流程；
- 缺少 LSE、causal 或梯度验证：对应正确性子项不得满分；
- 只有聚合均值，没有 measurement 次数、p50、配置或原始轻量数据：性能子项不得满分；
- 没有 4090 24GB 元数据、开跑前空闲显存不足 22 GiB、混用多张卡，或没有在首次 CUDA
  allocation 前设置 23552 MiB allocator 上限：正式性能矩阵无效；
- `peak_reserved` 超过 23552 MiB、并发运行正式矩阵或缺少 `memory_evidence.json`：24G
  可复现性未通过，相关性能与显存结果不得满分；
- 静默降配、删除 OOM 行或跨硬件计算 speedup：对应实验结果无效。

## 核验方式

1. 运行 `python3 scripts/validate_repo.py`，检查目录、必交结果、文件类型、大小和公开性。
2. 在固定 commit 的上游工作仓库运行 `uv run pytest tests/test_attention.py -v`。
3. 检查 `tests/adapters.py` 是否确实返回学生实现类，而非跳过、mock 或已有实现包装。
4. 检查 `cs336_systems/a2k/` 中是否存在真实 Triton kernel、online softmax 状态和 causal
   mask，并要求学生解释 tile 与精度选择。
5. 随机复跑至少一个 causal 和一个 non-causal 配置，核对 `O`、`L`、`dQ`、`dK`、`dV`。
6. 在助教 4090 上从 checkpoint、compile 和 Flash 表各抽一行，使用 metadata 中的命令
   复跑；核对固定 shape、p50、23 GiB allocator guard 和峰值显存口径。
7. 检查至少两张图能回到 CSV/JSON，而不是只有截图数字。

## 需要退回修正的情况

- 把 CPU interpreter、skip 或未执行测试写成 GPU 通过；
- 把 PyTorch baseline 或已有 fused attention 写成学生 Triton kernel；
- `tests/adapters.py` 没有连接提交代码，或修改公共 tests 降低要求；
- benchmark 把输入创建、首次 compile、不同 dtype 或不同硬件混入对照；
- 缺少 checkpoint 的 1024 标准矩阵、2048 边界、固定 block size 或 OOM/fallback 记录；
- 缺少 attention 核心矩阵或 16384 长序列 eager/Triton 边界对照；
- 缺少 23 GiB allocator guard、`memory_evidence.json`，或把更大显存卡的无约束结果当作
  24GB 可复现结果；
- 提交 compile cache、PTX/CUBIN、binary、完整 trace、上游仓库、依赖环境或超限附件；
- 报告包含内部地址、账号、路径、UUID、进程、未公开项目或任何凭据；
- 报告中的关键数字无法追溯到结果文件和命令。
