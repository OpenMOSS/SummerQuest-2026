# A2-P 评估补充说明（批改助教）

> 本文件说明评估方式，不改变 [`README.md`](README.md) 中的任务和提交要求。A2-P 当前为
> 发布候选稿；正式发布前不接收学生提交。

## 评估原则

- 评估链路是“命令与配置 → 轻量原始数据 → 表格/截图 → 解释”，缺少前面的证据时，
  只给出结论不能获得完整分数。
- Nsight Systems 和 `torch.profiler` 都可以作为六个 `train_step` trace 的主工具。
  选择 `torch.profiler` 时，不要求学生提供 nsys 专属的 CUDA API 关联字段；
  但必须有 CPU/CUDA activities、阶段标记、Calls、CPU/CUDA 时间和可阅读的时间线。
- 评分看真实完成度和分析质量，不奖励预填、猜测或无法追溯的测量数字。
- 平台、CUPTI、显存或管理员原因造成的阻塞不等于未完成，但必须如实记录原配置、失败阶段、
  异常类型、已尝试的 fallback 和联系助教的情况。

## 评分

保留上游五道 Profiling 小题的原始分值，总分 **16 分**，不归一化为 100 分：

| 部分 | 分值 | 核验重点 |
| --- | ---: | --- |
| End-to-End Benchmark | 4 | 三种 mode、同步、warm-up、raw timings、均值/标准差/CV |
| Compute Profiling | 5 | 六个 `train_step` trace、阶段标记、op/kernel Calls、CPU/CUDA 时间与归因 |
| Mixed Precision Accumulation | 1 | 四种累加结果与两类误差来源 |
| Mixed Precision Benchmark | 2 | ToyModel BF16 dtype、FP32/BF16 时间、显存与趋势解释 |
| Memory Profiling | 4 | 规定矩阵、timeline、峰值口径、allocation、residual/gradient 分析 |

报告可读性、公开性、复现性和证据完整性在每一部分内评分，不另设只靠排版获得的分数。

## 核验方式

1. 运行 `python3 scripts/validate_repo.py` 检查目录、文件类型、大小、占位符、飞书链接和
   明显凭据。
2. 阅读 `README.md`，随机抽取至少三个数字，确认可以定位到 `results/` 或明确命令。
3. 检查 `submission/profiling/` 的计时边界、CUDA 同步、warm-up、阶段标记和 metadata。
4. 对一个 benchmark 配置复跑短测；必要时要求学生现场打开本地 `.nsys-rep`、
   `torch.profiler` Chrome trace 或 memory snapshot。大型原始文件不应因抽查而进入 GitHub。
5. 对 OOM/CUPTI/资源阻塞按题面 fallback 核对，不把诚实记录的基础设施问题当作造假。

## 需要退回修正的情况

- 只提交 `torch.profiler` operator 表，没有六个 trace、CUDA activity、阶段标记或时间线；
- 把首次 CUDA 初始化或数据生成计入正式 measurement；
- 缺少同步、raw timing、标准差或实验配置；
- 静默改变模型/context/batch 后仍使用原配置标签；
- 提交 `.nsys-rep`、snapshot、完整 trace、权重、数据、压缩包，或 `results/` 与
  `assets/` 合计超过 2 MiB；
- 报告包含内部地址、账号、路径、进程、UUID 或任何凭据；
- 报告中的关键数字无法追溯，或与提交的轻量数据明显矛盾。
