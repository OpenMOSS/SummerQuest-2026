# A2-P 公开提交：马万里

## 基本信息

- 作业题面版本：`26.1.4-rc.3`
- 完成范围：任务一（Benchmark）、任务二（Compute Profiling）、任务三（Mixed Precision）、任务四（Memory Profiling）均已执行，含资源受限的如实记录。
- 未完成项：`xl/ctx2048` 的完整 fp32 `train_step` 因显存不足（24GB 与 80GB(A800) 均 OOM，失败前 `max_memory_allocated` ≈ 77.5 GiB）而无法采集，已在 `results/memory/failures.jsonl` 如实记录（见第 4 节），本次未做更小模型回退；无其它未完成项。
- 上游 starter commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`
- 本地工作仓库：`../assignment2-systems`

## 环境与工具

| 项目 | 公开、脱敏的信息 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090（compute capability 8.9，可用显存约 23.5 GiB） |
| Driver / CUDA | NVIDIA driver 560.28.03（RTX 4090 节点）/ 580.159.03（A800 节点）；CUDA 12.6 |
| PyTorch | 2.11.0+cu126 |
| Compute profiler | NVIDIA Nsight Systems 2023.1.2.43 |
| 其他限制 | 无 GUI 的集群环境：无法使用 Nsight GUI / Perfetto 网页，改用 nsys 离线汇总 + matplotlib 离线绘制 timeline；无中文字体，图表文字用英文 |

## 1. End-to-End Benchmark

### 复现命令与计时方法

统一入口 `profiling/benchmark.py`，支持 `--model-size --batch-size --context-length --dtype --mode --warmup --steps --seed --output`。三种 mode 的计时边界：

| mode | 内容 | 说明 |
| --- | --- | --- |
| `forward` | `model(input)`，包在 `no_grad` | 不含 loss/backward/optimizer |
| `forward_backward` | forward → `cross_entropy` → `loss.backward()`；每步 `model.zero_grad` | 梯度不跨 step 累积 |
| `train_step` | zero_grad → forward → loss → backward → `optimizer.step()` | 完整训练 step |

计时方法：
- 用 `time.perf_counter()` 计时。
- **每个被测 CUDA step 后调用 `torch.cuda.synchronize()`**；测量循环开始前再显式 `torch.cuda.synchronize()` 作为同步边界，避免上一个 warm-up step 的尾部 kernel 混入。
- 数据生成与模型初始化**不计时**。
- 先跑 `warmup` 步，再对 `steps` 步计时；记录 raw timing + 均值 + 样本标准差 + 变异系数(CV)。

复现命令（small / batch=4 / context=512 / fp32 基线，**5 warm-up / 10 测量**，train_step 额外对比 warm-up=0）：
```bash
B=profiling/benchmark.py
python $B --model-size small --batch-size 4 --context-length 512 --dtype fp32 --mode forward          --warmup 5 --steps 10 --output results/benchmark.csv
python $B --model-size small --batch-size 4 --context-length 512 --dtype fp32 --mode forward_backward --warmup 5 --steps 10 --output results/benchmark.csv
python $B --model-size small --batch-size 4 --context-length 512 --dtype fp32 --mode train_step       --warmup 5 --steps 10 --output results/benchmark.csv
python $B --model-size small --batch-size 4 --context-length 512 --dtype fp32 --mode train_step       --warmup 0 --steps 10 --output results/benchmark.csv
```

### 结果

来自 `results/benchmark.csv`（small, batch=4, context=512, dtype=fp32, n=10）：

| mode | warmup | mean (s) | std (s) | CV (%) |
| --- | ---: | ---: | ---: | ---: |
| forward | 5 | 0.023460 | 0.000179 | 0.76 |
| forward_backward | 5 | 0.079043 | 0.000101 | 0.13 |
| train_step | 5 | 0.089029 | 0.000134 | 0.15 |
| train_step | 0 | 0.121089 | 0.101093 | 83.49 |

（raw timings 每行一条写在同一 CSV 的 `time_sec`，每条配置 10 条。）

### 分析

- **warm-up 前后差异**：warm-up=0 不预热时，第一个测量 step 包含了 CUDA 上下文初始化、cuDNN 自动调优与首次显存分配等**一次性开销**，首测约 0.409s，直接把均值抬到 0.121s、CV 拉到 83%；而 warm-up=5 时 CV 仅 0.15%，说明 5 步预热已让分配器与 kernel 选择进入稳定态。只预热 1–2 步时，首测仍可能带上 CUDA/cuDNN 初始化余量，所以结果仍偏大。
- **mode 排序合理**：forward(0.023) < forward_backward(0.079) < train_step(0.089)。backward 约为 forward 的 3.4x（梯度反传更重的 matmul + elementwise）；optimizer（0.079→0.089）额外约 0.010s，对应 AdamW 大量逐元素更新。
- 除 warm-up=0 外，std 都在 ±0.0001–0.0002s，**测量稳定**；正是“先 warm-up 再计时、每步同步”带来的低抖动。

## 2. Compute Profiling

### 六个 `train_step` trace 与命令

选 `small`、`medium` 两个规模 × {256,512,1024} 三个 context，共 6 个**完整 `train_step`** trace，统一用 **Nsight Systems**，统一 **batch=1** 以保证六条口径一致（`medium/context=1024` 在 batch=4 时 24GB 显存放不下，故六条整体降到 batch=1，见第 5 节）。

- 工具：`nsys profile`（nsys 2023.1.2.43），每条只捕获一个**预热后的稳定测量 step**。
- 阶段标记（NVTX，见 `nvtx_ranges.py`/`profile_one_step.py`）：`profile/warmup`、`profile/measure`、`forward`、`backward`、`optimizer`、`attention/scores`、`attention/softmax`、`attention/value`。
- 复现（六个配置 × batch=1；`profile_one_step.py` 入口会先调用 `patch_attention()` 安装 attention 子区间标注）：
```bash
for M in small medium; do for C in 256 512 1024; do
  N=run_${M}_ctx${C}
  nsys profile --output results/profile/${N} --force-overwrite true \
      python profiling/profile_one_step.py --model-size $M --batch-size 1 \
      --context-length $C --dtype fp32 --warmup 5
  rm -f results/profile/${N}.sqlite
  nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --force-export true --format csv \
      results/profile/${N}.nsys-rep > results/profile/${N}_stats.csv
done; done
python profiling/summarize.py     # → results/profile/{trace_summary.csv, run_metadata.json}
```

每个配置的模型/context/train_step/dtype/batch/tool/命令/本地 trace 文件名见 `results/profile/run_metadata.json`（六条均 `batch_size=1, dtype=fp32, tool=nsys, status=success`）。

### Kernel、Calls 与时间线

`results/profile/trace_summary.csv` 用一个 `kind` 列承载三类机器可读结果（列：`model_size,context_length,kind,name,calls,cpu_time_us,cuda_time_us,phase`）：
- `kind=kernel`：GPU kernel 的 Calls 与**累计 CUDA 时间**（每配置 top5，只取 kernel，不混入 CUDA API 项）；
- `kind=stage`：NVTX 阶段范围（`forward`/`backward`/`optimizer`，以及 `attention/scores|softmax|value`）的 Calls 与时长；
- `kind=cpu_api` / `kind=cpu_api_total`：CUDA runtime API 的 Calls 与**累计 CPU 时间**（含该配置的 CPU API 总时间）。

| 配置 | top kernel（Calls × 累计CUDA时间，phase） |
| --- | --- |
| small/ctx256 | vectorized_elementwise(4296×25.7ms, elem) · vectorized_elementwise(3978×16.8ms, elem) · elementwise(942×12.1ms, elem) |
| small/ctx512 | vectorized_elementwise(4296×25.7ms, elem) · sgemm_64x64_nn(438×24.3ms, matmul) · vectorized_elementwise(3978×18.3ms, elem) |
| small/ctx1024 | sgemm_128x64_tn(510×43.0ms, matmul) · cutlass_Kernel2(222×31.6ms, matmul) · sgemm_128x64_nn(432×29.8ms, matmul) |
| medium/ctx256 | vectorized_elementwise(8472×86.9ms, elem) · vectorized_elementwise(7866×51.1ms, elem) · elementwise(1878×38.0ms, elem) |
| medium/ctx512 | vectorized_elementwise(8472×88.0ms, elem) · sgemm_128x64_tn(438×54.9ms, matmul) · vectorized_elementwise(7866×53.9ms, elem) |
| medium/ctx1024 | sgemm_128x64_tn(1014×134.0ms, matmul) · vectorized_elementwise(8472×88.1ms, elem) · cutlass_Kernel2(438×81.5ms, matmul) |

CPU 侧（`kind=cpu_api_total`，CUDA runtime API 累计 CPU 时间）：small/ctx256 1262.0ms、small/ctx512 872.1ms、small/ctx1024 911.1ms、medium/ctx256 1420.9ms、medium/ctx512 1299.2ms、medium/ctx1024 1444.9ms。

代表配置（medium, context=512, batch=1）timeline（measure step 387.09 ms，6799 个 kernel）：

![medium/ctx512 GPU timeline（forward/backward/optimizer 阶段 + 按 kernel 类别着色）](assets/medium512_timeline.png)

各阶段耗时（来源：`results/profile/trace_summary.csv` 的 `kind=stage` 行）：

| 阶段 | Calls | 耗时 (ms) | 占比 |
| --- | ---: | ---: | ---: |
| forward | 1 | 140.73 | 36.4% |
| backward | 1 | 80.39 | 20.8% |
| optimizer | 1 | 165.26 | 42.7% |
| attention/scores | 24 | 7.26 | — |
| attention/softmax | 24 | 0.94 | — |
| attention/value | 24 | 3.32 | — |

**阶段标记的实现说明**：`attention/*` 子区间由 `nvtx_ranges.py` 的 annotated_sdpa（基于 `einsum` 与 `softmax` 的实现）产生，其耗时高于融合实现，因此本测量 step 为 387 ms，并包含 softmax kernel；阶段占比随该实现而变，kernel 级归因不受影响。

kernel 家族计数：elementwise 5849 · matmul 578 · other 171 · reduce 152 · softmax 48 · embedding 1。

小模型代表配置（small/ctx512, batch=1）timeline（measure step 105.69 ms，3451 kernels：forward 28.09 / backward 40.67 / optimizer 36.39 ms；elementwise 2969 · matmul 290 · reduce 80 · softmax 24 · other 87 · embedding 1），同样呈现 elementwise 占大多数 kernel。

![small/ctx512 GPU timeline](assets/small512_timeline.png)

### 工具边界

本实验用 **nsys**：`nsys stats --report cuda_gpu_kern_sum,cuda_api_sum` 同时给出 GPU kernel 汇总与 CUDA API 汇总，能在 CPU 侧 CUDA API（`cudaLaunchKernel`/`cudaMemcpyAsync` 等）与 GPU kernel 之间做关联；NVTX 区间用于把 warm-up 与测量 step、以及 `forward/backward/optimizer` 和 `attention/*` 子区间分开。本报告 `trace_summary.csv` 用 `kind` 区分 **kernel / stage / cpu_api**：kernel 行只统计 GPU kernel，CUDA runtime API 单列为 `cpu_api` / `cpu_api_total` 行，NVTX 阶段时长列为 `stage` 行。

### 关键结论（对应 PDF §2.1.4 的 (a)–(e)）

- (a) benchmark.py 只实测了 small/ctx512（forward≈0.023s），未实测 medium/ctx512；profiler 中 medium/ctx512 的 forward 为 140.73 ms（见“阶段标记的实现说明”），与标准库的数值不可直接比较，仅用于归因。
- (b) forward 内累计 GPU 时间最高的 kernel 是 **elementwise**（SiLU/LayerNorm/cast 等）；context 增大到 1024 时 **matmul(sgemm)** 反超成为最高；而 matmul 的数量始终远少于 elementwise。
- (c) 除矩阵乘外，elementwise/reduce（LayerNorm、softmax 的 max/sum、激活、cast、梯度）与 **AdamW 逐元素更新**占用可观 CUDA 时间——optimizer 区间几乎全是 elementwise。
- (d) 完整训练 step 中 matmul 占比相对纯推理(forward) 下降，elementwise+optimizer 占比上升。
- (e) 用 `attention/*` 子区间直接对比：attention 内 **matmul 部分**（scores 7.26ms + value 3.32ms）远大于 **softmax 0.94ms**，与 FLOPs 的差距一致——softmax 访存受限、matmul 计算密集，softmax 相对“便宜”。

## 3. Mixed Precision

### 四种累加实验

对同一求和 `Σ 0.01 × 1000 = 10.0`，四种写法（`profiling/mixed_precision.py --run accumulation`）：

| 写法 | 结果 | 误差 | 说明 |
| --- | ---: | ---: | --- |
| fp32 累加器 + fp32 加数 | 10.0001335 | +0.0001 | 基准，几乎精确 |
| **fp16 累加器 + fp16 加数** | **9.953125** | **-0.0469** | 累加器低精度，误差最大 |
| fp32 累加器 + fp16 加数 | 10.0021362 | +0.0021 | 输入量化 |
| fp32 累加器 + cast(fp16→fp32) 加数 | 10.0021362 | +0.0021 | 与上一行同值 |

**两类误差来源**：
- **累加器精度**占主导：fp16 累加器每次加法都发生舍入、误差不断累积，最终比真值少约 0.047（约 -0.5%）；而 fp32 累加器几乎不受累加累积影响。
- **输入量化**是次要的：`0.01` 被量化成 fp16（约 0.009998…，实际偏 +0.0021），但用 fp32 累加器后误差被限制在加数的量化偏置上（+0.0021）；第 3、4 段等价（PyTorch 会把 fp16 加数提升到 fp32 再相加）。

结论：**应在高精度(f32)下完成累加/reduction**，即使参与运算的加数被降精度；否则低精度累加器会带来远大于输入量化的误差。

### FP32 与 BF16 autocast

`ToyModel`（fc1→ReLU→LayerNorm→fc2）用 `profiling/mixed_precision.py --run all` 在 GPU 上运行，数据在 `results/mixed_precision.json`。运行于 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 内；记录各组件 dtype，并在同 batch/warmup/steps 下对比 FP32 与 BF16 autocast 的 forward+backward 时间、峰值显存与数值趋势。

**组件 dtype（实测）**：

| 组件 | FP16 autocast | BF16 autocast |
| --- | --- | --- |
| 参数 (`fc1.weight`) | fp32 | fp32 |
| `fc1` 输出 | fp16 | bf16 |
| ReLU 输出 | fp16 | bf16 |
| LayerNorm 输出 | **fp32** | **fp32** |
| logits | fp16 | bf16 |
| loss | fp32 | fp32 |
| 梯度 (fc1/ln/fc2) | **fp32** | **fp32** |

要点：参数不被 autocast 改变（保持 fp32）；`Linear`（matmul）被降到 autocast dtype 用 Tensor Core；ReLU 跟随输入；**LayerNorm 保持 fp32**（norm/reduction 需要 fp32 动态范围）；loss 与**梯度均为 fp32**（autocast 反向安全策略）。换 BF16 后 LayerNorm 依旧 fp32——BF16 指数位与 fp32 相同（不缩水动态范围，不像 FP16 易溢出/下溢），但尾数仍 8 位，归一化/累积仍宜在 fp32 里做。

**FP32 vs BF16（batch=32, warmup=5, steps=10）**：

| 项 | FP32 | BF16 autocast |
| --- | --- | --- |
| 平均时间 (ms) | **1.202** (±0.042) | **1.508** (±0.049) |
| 峰值显存 (MiB) | 16.28 | 16.29 |
| loss | 1.9449 | 1.9456 |
| logits 平均绝对值 | 0.4703 | 0.4707 |

趋势：**小模型上 BF16 反而更慢**（1.20 vs 1.51 ms）且**显存几乎不变**（16.28 vs 16.29 MiB）——因为几乎没有吃到 Tensor Core 的大矩阵乘，autocast 还引入类型转换与启动开销；数值上 BF16 与 FP32 很接近（无溢出/NaN）。**结论：混合精度收益在大模型/大矩阵乘上才明显**，这解释了为何要在 XL/长 context 下对比（见第 4 节）。

## 4. Memory Profiling

### 配置、峰值与 fallback

矩阵：XL(batch=1) × {128,1024,2048} 的 fp32 forward/train_step，外加混合精度（autocast bf16：参数保持 fp32、forward/loss 在 `torch.autocast(bf16)` 下执行，即 `--dtype fp32 --amp bf16`）的 forward/train_step{128,1024}，以及纯 BF16（`--dtype bf16`）的 forward/train_step{128,1024}。用 `profiling/memory_profiler.py`：初始化+warm-up 后 `torch.cuda.memory._record_memory_history`+`_dump_snapshot`。

峰值口径（`results/memory/peaks.csv` 的列）：
- `peak_active_mib` = `torch.cuda.max_memory_allocated()`：PyTorch 口径的**在用(active)峰值**，**包含**启用 memory history 之前就已存活的权重/优化器 allocation；
- `peak_reserved_mib` = `torch.cuda.max_memory_reserved()`：预留(reserved)峰值（≥ active）；
- `traced_delta_mib` = 仅由启用 history 之后的 alloc/free 事件重建的**增量**（不含权重），仅作诊断，**不代表在用显存**。

失败配置见 `results/memory/failures.jsonl`（含 stage/exception/峰值）。

峰值（GiB，`peak_active_mib` = PyTorch `max_memory_allocated`）：

| 配置 | fp32 forward | fp32 train_step | mixed(autocast bf16) forward | mixed train_step | **pure BF16** forward | **pure BF16** train_step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| xl/128 | 12.85 | **51.40** | 19.13 | 51.40 | **6.36** | **25.65** |
| xl/1024 | 13.39 | **56.36** | 19.35 | 57.01 | **6.63** | **28.76** |
| xl/2048 | 14.95 | OOM(>80) | — | — | — | — |

三种精度口径：
- **fp32**：基线。
- **混合精度（autocast bf16）**：参数保持 fp32，`torch.autocast(bf16)` 将 matmul 等算子降为 bf16；实测峰值不降（forward 升高，见 (c)）。
- **纯 BF16**：`model.to(torch.bfloat16)`，参数、梯度与优化器状态均为 bf16；实测峰值约为 fp32 的一半（forward 12.85→6.36、train_step 51.40→25.65 GiB）。

residual stream 理论大小：1.25 / 10 / 20 MiB（ctx128/1024/2048，=batch×seq×d_model×4，d_model=2560）。

**OOM（如实记录于 `failures.jsonl`）**：XL 的 `train_step` 在 **24GB** 必 OOM（权重≈12.8GiB + **AdamW 两个动量≈2×12.8GiB** + 反向激活 >24GiB，与 context 无关），故 train_step 在 **A800(80GB)** 上做；**`xl/2048` 的 fp32 train_step 连 80GB 也不够**（失败前 `max_memory_allocated` 已达 **77.52 GiB**、reserved 78.53 GiB），`failures.jsonl` 记录 stage=warmup、`OutOfMemoryError`。本次混合精度（autocast bf16）只到 1024，未扩展到 2048。

### Timeline、allocation 与 residual/gradient

- **Active / Reserved 显存时间线（PyTorch memory_viz 官方工具）**：
  - forward（在用 active）：峰值≈权重基线 12.85 GiB + 激活增量 21 MiB。

    ![XL/ctx128 forward 在用（active）显存时间线（PyTorch memory_viz）](assets/xl_ctx128_forward_active_mem.png)
  - full train_step（在用 active）：峰值≈**51.40 GiB**，出现在**反向/优化器**阶段（权重+梯度+2×AdamW 动量）。

    ![XL/ctx128 full train_step 在用显存时间线（PyTorch memory_viz）](assets/xl_ctx128_train_step_active_mem.png)
  - reserved（Cached Segment）：forward≈12.1 GiB、train_step≈51.2 GiB，说明 **reserved ≥ allocated ≥ active**。

    ![XL/ctx128 forward reserved（Cached Segment）时间线](assets/xl_ctx128_forward_active_cached.png)
    ![XL/ctx128 train_step reserved（Cached Segment）时间线](assets/xl_ctx128_train_step_active_cached.png)
- **口径**：`peak_active_mib` = PyTorch `max_memory_allocated()`（在用/active 峰值，含启用 history 前已存活的权重与优化器状态）；`peak_reserved_mib` = `max_memory_reserved()`（预留池，reserved ≥ active）；`traced_delta_mib` 仅为启用 history 之后的分配**增量**（不含权重），**不等于在用显存**。报告峰值一律用 `peak_active_mib / peak_reserved_mib`。
- **最大 allocation**（见 plot 输出）：xl/128 forward 单次 5 MiB、xl/1024 forward 128 MiB、xl/2048 forward 512 MiB；来源为前向激活/梯度（该工具的 Python 栈只能归到最外层脚本，无法逐层归因，为已知限制）。
- **residual/gradient**：单层 residual stream 张量＝`[B,seq,d_model]×4B`（XL d_model=2560）：ctx128=1.25、1024=10、2048=20 MiB。train_step 要为每层保存该 residual 供反向，同时反向产生等量级**梯度**；XL 共 32 层 → 残差流合计≈32×[seq,d_model]。结合 **train_step≈W(权重)+W(梯度)+2W(AdamW 动量)+激活 ≈ 4×权重**，可解释为何 train_step 峰值远超 forward（xl/128 实测 51.40 GiB≈12.85×4）。
- **任务四 (c) 混合精度**：`torch.autocast(bf16)` 下，forward 峰值升高（xl/128 12.85→19.13 GiB、xl/1024 13.39→19.35 GiB），train_step 基本持平（51.40→51.40、56.36→57.01 GiB）。原因：参数仍为 fp32，autocast 另需缓存 bf16 权重副本（≈+6.4 GiB）；train_step 峰值则由 fp32 的权重、梯度与优化器状态主导。因此在该设定下，混合精度对峰值显存没有明显收益。
- **纯 BF16 的对比**：将模型整体转为 BF16（`model.to(bf16)`）时，峰值约为 fp32 的一半（forward 12.85→6.36、train_step 51.40→25.65 GiB），因为权重、梯度与优化器状态均为 bf16。

  ![XL/ctx128 混合精度(autocast bf16) forward 在用显存时间线](assets/xl_ctx128_forward_ampbf16_mem.png)

  ![XL/ctx128 混合精度(autocast bf16) full train_step 在用显存时间线](assets/xl_ctx128_train_step_ampbf16_mem.png)

## 5. 限制与复现

- 代码同步命令：`python3 scripts/sync_a2p_submission.py --name '马万里'`
- 轻量结果目录：`results/`（`benchmark.csv`、`profile/{trace_summary.csv, run_metadata.json}`、`mixed_precision.json`、`memory/{peaks.csv, run_metadata.json, failures.jsonl}`）。
- 未提交的本地大型原始文件：`.nsys-rep`、`.sqlite`、`.qdstrm` 等 profiler 原始文件，以及 memory 的 `*.pickle` snapshot，仅保留在本地工作仓库（`results/profile/`、`results/memory/snapshots/`），不进入提交。
- 已知限制：集群无 GUI，compute/memory 时间线在集群用离线脚本成图（matplotlib）；memory 快照另在本地用 PyTorch memory_viz 打开作官方核实（本报告 memory 时间线即来自 memory_viz）。driver：RTX 4090 节点 560.28.03、A800 节点 580.159.03；CUDA 12.6、PyTorch 2.11.0+cu126。profiling 六条 trace 统一 batch=1、同一张 RTX 4090。memory 部分：XL 的 train_step 在 24GB(RTX 4090) 必 OOM，故 train_step 在 A800(80GB) 上做；`xl/ctx2048` 的 fp32 train_step 连 80GB 也不够（失败前 `max_memory_allocated` ≈ 77.5 GiB），已在 `failures.jsonl` 记录；混合精度（autocast bf16）只到 1024。
- 最小复现步骤：
  - 任务一：见第 1 节四条 `python profiling/benchmark.py ...` 命令；
  - 任务二：见第 2 节 `nsys profile` / `nsys stats` 循环，随后 `python profiling/summarize.py`；出图 `python profiling/plot_nsys_timeline.py --name run_medium_ctx512 --out assets/medium512_timeline.png`；
  - 任务三：`python profiling/mixed_precision.py --run all`；
  - 任务四（A800）：对每个配置执行
    `python profiling/memory_profiler.py --model-size xl --context-length C --batch-size 1 --dtype fp32 [--amp bf16] --mode forward 或 train_step --warmup 3 --tag TAG`（C 取 128/1024/2048，TAG 自定），
    然后 `python profiling/memory_profiler.py --finalize`，出图 `python profiling/plot_memory_timeline.py --tag TAG --out results/figures/TAG_mem.png`；
    快照再用本地 PyTorch memory_viz 打开核对。

## 飞书补充文档

https://fudan-nlp.feishu.cn/wiki/Won8wo5rriiyS3kQBlzcWxw2nvd

## 自检

- [x] 本 PR 只包含我本人本次 A2-P 的文件。
- [x] `README.md` 是 Markdown 主报告，所有图片使用相对路径和有意义的 alt text。
- [x] 每个关键数字都能回到命令、`results/` 或 metadata。
- [x] 引用仓库外源码或资料时使用固定 commit 的 GitHub HTTPS 绝对 URL，未写入本机路径或 `file://` 链接。
- [x] 已用 nsys 完成六个 `train_step` trace，并提交轻量汇总。
- [x] 已提交 1 张 Compute Profile 关键图和至少 2 张 Memory Timeline，均已裁剪、压缩并被报告引用。
- [x] `results/` 与 `assets/` 公开附件合计不超过 2 MiB。
- [x] 未提交 `.nsys-rep`、snapshot、完整 trace、权重、数据、压缩包或依赖环境。
- [x] GitHub 内容不含内部主机名、IP、账号、路径、UUID、进程或未公开项目。
- [x] GitHub 和飞书正文都不含 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档为组织内公开，且未开启互联网公开访问。
