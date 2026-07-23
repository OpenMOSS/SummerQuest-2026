# A2-P 公开提交：王扬


## 基本信息

- 作业题面版本：`26.1.4-rc.3`
- 完成范围：`benchmarking_script`、`nsys_profile`、`mixed_precision_accumulation`、`benchmarking_mixed_precision`
- 已完成项：`memory_profiling` 已覆盖 XL 在 `ctx=128` 与 `ctx=2048` 的 forward / train_step，均提供 fp32 与 bf16 的峰值结果
- 上游 starter commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`
- 本地工作仓库：`../assignment2-systems`

## 环境与工具

| 项目 | 公开、脱敏的信息 |
| --- | --- |
| GPU | `NVIDIA H200` |
| Driver / CUDA | `12.8` / `12.8` |
| PyTorch | `2.10.0+cu128` |
| Compute profiler | `Nsight Systems` + `torch.profiler` |
| 其他限制 | `results/nsys/` 只保留轻量导出，不提交 `.nsys-rep` 和完整 trace |

## 1. End-to-End Benchmark

### 复现命令与计时方法

`results/benchmark.csv` 汇总了所有公开的 benchmark 结果；逐 step raw timings、命令、环境与参数记录在 `results/benchmark/main/*.json`、`results/benchmark/warmup/*.json` 和 `results/benchmark/mixed_precision/*.json`。计时边界按 `forward`、`forward_backward`、`train_step` 区分，并在每个测量 step 后显式 `torch.cuda.synchronize()`。

### 结果

见 `results/benchmark.csv` 和 `assets/benchmark_fp32_mean_ms.png`。统一基线为 `small / batch 4 / ctx 512 / fp32` 时，`forward`、`forward_backward`、`train_step` 的均值分别为 `19.649 ms`、`58.726 ms`、`74.072 ms`。

`warmup=0/1/2/5` 的对照也已保留在 `results/benchmark/warmup/`，其中 `warmup=0` 的 `train_step` 均值明显偏高，且波动更大。

### 分析

结果符合预期的层级关系：`forward < forward_backward < train_step`。`warmup=0` 时，首次迭代把 CUDA context、allocator 和 kernel 选择等一次性开销混进测量；当 warm-up 增加到 `2` 以后，均值和 CV 基本收敛。

## 2. Compute Profiling

### 六个 `train_step` trace 与命令

`results/profile/trace_summary.csv` 记录了 6 个 `train_step` 配置的轻量摘要，覆盖 `large` 与 `xl` 两个模型、`ctx 256/512/1024` 三个 context length。阶段范围统一按 `measure@profile`。每个 run 的公开索引都对应一个本地 `source_stats_file`，用来回查原始 `nsys` 统计导出和对应的 trace 产物。

| model | ctx | tool | stage | top kernel calls | matmul time | softmax time | source stats |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| large | 256 | nsys | `measure@profile` | 217 | 148916217.0 | 58976.0 | `results/nsys/large_train_step_ctx256_nsys_stats.csv` |
| large | 512 | nsys | `measure@profile` | 108 | 284889673.0 | 120639.0 | `results/nsys/large_train_step_ctx512_nsys_stats.csv` |
| large | 1024 | nsys | `measure@profile` | 181 | 570494699.0 | 232829.0 | `results/nsys/large_train_step_ctx1024_nsys_stats.csv` |
| xl | 256 | nsys | `measure@profile` | 225 | 464910931.0 | 59104.0 | `results/nsys/xl_train_step_ctx256_nsys_stats.csv` |
| xl | 512 | nsys | `measure@profile` | 97 | 909253401.0 | 119966.0 | `results/nsys/xl_train_step_ctx512_nsys_stats.csv` |
| xl | 1024 | nsys | `measure@profile` | 224 | 1851446541.0 | 227038.0 | `results/nsys/xl_train_step_ctx1024_nsys_stats.csv` |

相关 `.nsys-rep`、SQLite 和完整 trace 留在本地工作仓库；公开目录只提交 `results/nsys_kernel_summary.csv`、`results/nsys_api_summary.csv`、`results/nsys_top5_kernels.csv`、`results/profile/trace_summary.csv` 和 `results/profile/run_metadata.json`。

### Kernel、Calls 与时间线

关键汇总见 `results/nsys_kernel_summary.csv`、`results/nsys_top5_kernels.csv` 和 `results/profile/stage_summary.csv`，配图见 `assets/nsys_top_kernel_time.png`、`assets/nsys_matmul_vs_softmax.png` 和 `assets/nsys_memory_xl_ctx128_train_step.png`。代表性配置里，主时间仍集中在 matmul kernel，softmax 的累计时间远低于 matmul，`softmax_to_matmul_ratio` 处于 `1e-4` 量级。`forward` 主要被大矩阵乘吞没，`backward` 的累计时间显著更高，`optimizer` 虽然较短但稳定可见；attention 子阶段里，`attention/scores` 对应的相关矩阵计算最重，`attention/softmax` 更像同步与归一化瓶颈，`attention/value` 则对应后段的聚合计算。

### 工具边界

这里以 `nsys` 作为主证据，公开提交保留 kernel/API 轻量汇总，不提交完整 trace；本地 `torch.profiler` trace 只作为补充时间线阅读材料，不替代 nsys 的 kernel / API 汇总字段。

## 3. Mixed Precision

### 四种累加实验

`results/mixed_precision/accumulation.json` 记录了四种写法的真实输出。结果显示，`fp16` 累加时误差最大，`fp16` 输入但 `fp32` 累加时误差明显收敛；把 `fp16` 先转成 `fp32` 再累加，结果与直接 `fp16` 输入 + `fp32` 累加一致，说明主误差来自输入量化，而不是累加器。

| case | input dtype | accumulator dtype | result | absolute error | interpretation |
| --- | --- | --- | ---: | ---: | --- |
| case_1 | fp32 | fp32 | 10.000133514404297 | 0.000133514404296875 | 作为全精度对照，误差来自有限步长与浮点舍入。 |
| case_2 | fp16 | fp16 | 9.953125 | 0.046875 | 输入量化和低精度累加器同时放大误差。 |
| case_3 | fp16 | fp32 | 10.00213623046875 | 0.00213623046875 | 保留 fp32 累加器后，误差主要来自输入量化。 |
| case_4 | fp16_then_cast_to_fp32 | fp32 | 10.00213623046875 | 0.00213623046875 | 与 case_3 一致，进一步说明主要损失发生在输入被量化时。 |

### FP32 与 BF16 autocast

`results/mixed_precision.json` 中保留了 `toy_fp16_inspect.json` 与 `toy_bf16_inspect.json` 的检查结果。两者都显示参数是 `float32`，`LayerNorm` 输出是 `float32`，梯度是 `float32`；在 autocast 下，线性层和 logits 采用对应低精度 dtype。`assets/mixed_precision_bf16_speedup.png` 展示了 BF16 相对 FP32 的时间趋势。

| model | mode | fp32 mean ms | bf16 mean ms | speedup | note |
| --- | --- | ---: | ---: | ---: | --- |
| small | forward | 19.402967486530542 | 12.85653905943036 | 1.51x | 低精度主要影响线性层和 logits。 |
| medium | forward | 53.27919693663716 | 26.995753776282072 | 1.97x | BF16 带来更明显的吞吐改善。 |
| large | forward | 119.69942320138216 | 39.532746002078056 | 3.03x | 更大的矩阵乘更容易吃到 BF16 Tensor Core 收益。 |
| xl | forward | 332.0808389224112 | 65.04515446722507 | 5.10x | 该配置下 BF16 速度优势最明显。 |

当前公开结果里，ToyModel 的 dtype、时间和数值趋势已经完整记录；同一机器上的显存峰值也已经在 `results/memory/peaks.csv` 中按 `xl / ctx=128,2048 / forward, train_step / fp32, bf16` 逐项列出。对照这组结果可以直接看到：`forward` 下 `bf16` 的峰值高于 `fp32`，而 `train_step` 下 `ctx=2048` 的 `bf16` 峰值低于 `fp32`，说明 BF16 的显存收益会随阶段和缓存行为变化，不应简单概括为“总是更省显存”。

## 4. Memory Profiling

### 配置、峰值与 fallback

`results/memory/peaks.csv` 记录了 8 个 memory snapshot。XL 在 `ctx=128` 与 `ctx=2048` 的 forward / train_step 均已成功采集 fp32 与 bf16 结果，且公开的 `results/memory/*.json` 保存了同一批 run 的 `active`、`allocated`、`reserved` 和 `requested` 峰值，能逐项回查。

| model | ctx | mode | dtype | active peak bytes | allocated peak bytes | reserved peak bytes | requested peak bytes |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| xl | 128 | forward | fp32 | 13818056704 | 13818056704 | 13841203200 | 13683118080 |
| xl | 128 | forward | bf16 | 20564021248 | 20564021248 | 20778582016 | 20428759040 |
| xl | 128 | train_step | fp32 | 55241533952 | 55241533952 | 56161730560 | 55105513476 |
| xl | 128 | train_step | bf16 | 55238613504 | 55238613504 | 57921241088 | 55102953476 |
| xl | 2048 | forward | fp32 | 16074205184 | 16074205184 | 16651386880 | 15939266560 |
| xl | 2048 | forward | bf16 | 22067865600 | 22067865600 | 22588424192 | 21932402688 |
| xl | 2048 | train_step | fp32 | 97935502336 | 97935502336 | 99723771904 | 97792075784 |
| xl | 2048 | train_step | bf16 | 88640547840 | 88640547840 | 90269810688 | 88472791048 |

### Timeline、allocation 与 residual/gradient

已保留的图包括 `assets/memory_xl_ctx128_forward_fp32_timeline.png`、`assets/memory_xl_ctx128_train_step_fp32_timeline.png` 和 `assets/memory_xl_ctx128_forward_fp32_detail10.png`。`results/memory/*.json` 中给出了 `active`、`allocated`、`reserved` 和峰值字节数，可用来对照 residual stream tensor 在 forward 与 backward 阶段的释放和梯度生成。

XL 的 residual stream 理论规模可按 `batch_size × context_length × d_model × bytes_per_elem` 估算；对本次配置，`d_model = 2560`，因此 `ctx=2048` 的 residual 量级是 `ctx=128` 的 16 倍。这个线性关系能解释 forward 里 `bf16` 的峰值高于 `fp32`：例如 `ctx=128` 下 `forward` 的 `bf16` active peak 为 `20,564,021,248` 字节，而 `fp32` 为 `13,818,056,704` 字节。`train_step` 则同时叠加了 backward 保存张量、梯度和 optimizer 相关缓冲，所以峰值显著抬升；在 `ctx=2048` 下，`bf16` active peak 为 `88,640,547,840` 字节，低于 `fp32` 的 `97,935,502,336` 字节，说明 BF16 对最终峰值的影响还取决于 allocator 缓存、阶段内临时张量和释放时点，而不只是单个张量的元素大小。`reserved` 普遍高于 `allocated`，反映的是缓存分配器保留而非正在活跃使用的显存。

## 5. 限制与复现

- 代码同步命令：`python3 scripts/sync_a2p_submission.py --name '王扬'`
- 轻量结果目录：`results/`
- 未提交的本地大型原始文件：`results/nsys/*.nsys-rep`、`results/nsys/*.sqlite`、完整 trace 和 memory snapshot 过程文件
- 已知限制：无新增 memory 采集限制；完整 `torch.profiler` trace 仅作为本地补充证据，不进入公开提交
- 最小复现步骤：先同步 `submission/profiling/`，再读取 `results/*.csv`、`results/*.json` 和 `assets/*.png`

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/WgCDdzATZoI1bqxhnpGcuLssn8f?from=from_copylink

## 自检

- [x] 本 PR 只包含我本人本次 A2-P 的文件。
- [x] `README.md` 是 Markdown 主报告，所有图片使用相对路径和有意义的文件名。
- [x] 每个关键数字都能回到命令、`results/` 或 metadata。
- [x] 引用仓库外源码或资料时使用固定 commit 的 GitHub HTTPS 绝对 URL，未写入本机路径或 `file://` 链接。
- [x] 已用 nsys 或 `torch.profiler` 完成六个 `train_step` trace，并提交轻量汇总。
- [x] 已提交 1 张 Compute Profile 关键图和至少 2 张 Memory Timeline，均已引用。
- [x] `results/` 与 `assets/` 公开附件合计不超过 2 MiB。
- [x] 未提交 `.nsys-rep`、snapshot、完整 trace、权重、数据、压缩包或依赖环境。
- [x] GitHub 内容不含内部主机名、IP、账号、路径、UUID、进程或未公开项目。
- [x] GitHub 和飞书正文都不含 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档为组织内公开，且未开启互联网公开访问。
