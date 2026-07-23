# A2-P：Profiling 与性能分析

## 1. 实验概况

本报告覆盖 End-to-End Benchmark、Compute Profiling、Mixed Precision 和
Memory Profiling。题面版本为 `26.1.4-rc.3`，固定 starter commit 为
`ca8bc81a59b70516f7ebb2da4808daade877c736`。

实验环境：

- GPU：NVIDIA GeForce RTX 4090；
- PyTorch：2.6.0+cu124；
- Python：3.11.9；
- 工作目录：`../assignment2-systems/`；
- CUDA step 计时后调用 `torch.cuda.synchronize()`。

组织内补充文档链接：待补充。

## 2. End-to-End Benchmark

提交的机器可读汇总为 `results/benchmark.csv`。每一行对应一个原始 JSON
结果，保留了配置、均值、标准差、CV 和 raw timing 字段。

| source | mode | warmup | steps | mean ms | std ms | CV |
|---|---|---:|---:|---:|---:|---:|
| results/benchmark/benchmark_small_forward.json | forward | 5 | 10 | 23.940465785562992 |  | 0.004338412532068763 |
| results/benchmark/benchmark_small_forward_backward.json | forward_backward | 5 | 10 | 80.4965490475297 |  | 0.0012526447627510037 |
| results/benchmark/benchmark_small_train_step_w0.json | train_step | 0 | 10 | 128.32515221089125 |  | 0.9283876745433238 |
| results/benchmark/benchmark_small_train_step_w5.json | train_step | 5 | 10 | 90.56057427078485 |  | 0.0039344280376474835 |
| results/benchmark/smoke_forward.json | forward | 1 | 3 | 9.942912186185518 |  | 0.015269983581048212 |
| results/benchmark/smoke_forward_backward.json | forward_backward | 1 | 3 | 42.40483480195204 |  | 0.023403906491531797 |
| results/benchmark/smoke_train_step.json | train_step | 1 | 3 | 56.813385958472885 |  | 0.015773662189250862 |

`train_step` 额外比较了 warm-up=0 和 warm-up=5，用于观察 CUDA 初始化和
首次运行开销。

## 3. Compute Profiling

六个 train-step 配置为：

- `small_c256`；
- `small_c512`；
- `small_c1024`；
- `medium_c256`；
- `medium_c512`；
- `medium_c1024`。

阶段标记包括 `profile/warmup`、`profile/measure`、`forward`、`backward`、
`optimizer`、`attention/scores`、`attention/softmax` 和 `attention/value`。

轻量结果位于：

- `results/profile/trace_summary.csv`；
- `results/profile/stage_summary.csv`。

代表性 timeline：

![medium c512 timeline](assets/medium_c512_timeline.png)

custom range 可能互相重叠，因此阶段百分比不能直接相加。backward 的
自定义 CUDA 时间还可能受到 autograd 异步调度影响，解释时结合 operator/kernel
行进行归因。

## 4. Mixed Precision

累加误差实验比较 FP32/FP16 输入和累加器。FP16 输入加 FP16 累加器的误差
最大；FP32 累加器可以降低累加误差，但不能恢复输入量化已经丢失的精度。

ToyModel 实验使用 CUDA BF16 autocast，记录了参数、第一层输出、LayerNorm
输出、logits、loss 和 gradient dtype，并比较 FP32 与 BF16 的时间、显存和
loss 趋势。

结果位于 `results/mixed_precision.json`。

## 5. Memory Profiling

Memory 结果汇总在 `results/memory/peaks.csv` 和
`results/memory/run_metadata.json`。

| run | model | context | mode | status | max allocated MiB | max reserved MiB |
|---|---|---:|---|---|---:|---:|
| large_c2048_train_step | large | 2048 | train_step | success | 46494.697 | 47188.0 |
| xl_c1024_train_step | xl | 1024 | train_step | oom | 47473.793 | 48082.0 |
| xl_c128_forward | xl | 128 | forward | success | 13154.051 | 13168.0 |
| xl_c128_train_step | xl | 128 | train_step | oom | 47739.459 | 48110.0 |
| xl_c2048_forward | xl | 2048 | forward | success | 15305.682 | 15858.0 |
| xl_c2048_train_step | xl | 2048 | train_step | oom | 47369.502 | 47636.0 |

XL/context=2048 train_step 的 OOM 配置按题面要求保留；Large/context=2048
是 fallback 成功配置。`memory_snapshot.pickle` 只保留在远程工作目录，不提交。

![XL context 128 memory timeline](assets/memory_timeline_xl_c128.png)

![XL context 2048 memory timeline](assets/memory_timeline_xl_c2048.png)

## 6. 文件限制与脱敏

提交目录只包含 profiling 代码、CSV/JSON 轻量汇总、三张脱敏图片和本报告。
不提交完整 Chrome trace、`.nsys-rep`、SQLite、memory snapshot、模型权重、
数据集、虚拟环境、缓存、内部路径或凭据。

复核命令：

```bash
python3 scripts/validate_repo.py
git diff --check
```

报告中的数字应能回到 `results/` 的 CSV、JSON 或明确实验命令。
