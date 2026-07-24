# A2-K 公开提交：王扬

> 本文件和同目录代码、汇总、图片公开可见。只提交允许公开且已经脱敏的内容；上游仓库、
> 编译缓存、完整 trace 和大型原始文件留在个人工作目录。密钥和访问凭据不进入任何提交
> 材料。

> 正式要求见
> [`assignments/A2-K/README.md`](../../../../assignments/A2-K/README.md)，评分说明见
> [`assignments/A2-K/EVALUATION.md`](../../../../assignments/A2-K/EVALUATION.md)。

## 基本信息

- 作业题面版本：`26.1.4-k-rc.3`
- 完成范围：<填写>
- 未完成项：<填写；没有则写“无”>
- 上游 starter commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`
- 本地工作仓库：`../assignment2-systems`

## 环境与工具

| 项目 | 公开、脱敏的信息 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 24GB |
| 开跑前显存 | <填写 memory.total / memory.free；free 必须不少于 22 GiB> |
| Driver / CUDA | <填写> |
| PyTorch | <填写> |
| Triton | <填写> |
| power limit / P-state | <填写；使用默认设置> |
| TF32 | <填写 performance 与 FP32 correctness 的设置> |
| compile 配置 | <填写> |
| allocator limit / fraction | 23552 MiB / <填写实际 fraction> |
| 其他限制 | <填写；没有则写“无”> |

## 1. Activation Checkpointing

### 理论与代码骨架

<填写 checkpoint 安排、渐近峰值显存、计算量和不超过 20 行的代码骨架。>

### 固定实验

<引用 `results/checkpointing.csv`：先比较 context 1024 下无 checkpoint 与 block size
1/2/4/8，再报告 context 2048 的 baseline 和最低显存 checkpoint；填写 p50、peak
allocated、peak reserved、OOM/fallback。>

### 分析

<解释显存与重计算权衡，以及最佳配置为什么出现。>

## 2. PyTorch Attention 与 `torch.compile`

### 显式 PyTorch 基线

<说明显式 QK^T、mask、softmax、PV 实现和测量边界；引用
`results/attention_baseline.csv`。>

### Compile 对照

<引用 `results/compile_comparison.csv`，分开 cold-start 与 steady-state，并说明 graph
break、shape specialization 和完整 Transformer 对照。>

## 3. FlashAttention-2 Forward

### Pure PyTorch tiled reference

<说明 tile、保存的 O/L/Q/K/V、数值稳定性和 adapter 接口。>

### Triton kernel

<说明 launch grid、query/key tile、block pointer、online softmax、FP32 accumulator、
causal mask、num warps 和 num stages。>

## 4. Backward 与正确性

### 重计算式 backward

<说明 D 向量、dQ/dK/dV、torch.compile 边界，以及 PyTorch/Triton 两个 autograd path。>

### 官方 GPU tests

<引用 `results/unit_tests.txt`，明确填写 passed、failed、skipped；不能把 skip 写成 pass。>

### 扩展正确性

<引用 `results/correctness.json`，报告 O、L、dQ、dK、dV 的最大绝对/相对误差。>

## 5. 性能矩阵

### 配置与命令

<填写单张 RTX 4090 24GB、开跑前空闲显存、固定 batch、核心与 16384 边界 shape、dtype、
causal、`do_bench` warm-up/rep/quantiles 与命令。>

### 结果与图

<引用 `results/flash_benchmark.csv` 和至少两张 `assets/` 图片：核心矩阵比较 eager、
compiled 与学生 Triton，16384 边界至少比较 eager 与学生 Triton；报告
forward、backward、forward-backward、p20/p50/p80、显存和有效 speedup。>

### 分析

<解释短序列开销、长序列显存、tile 选择、OOM/compile 失败和不确定性。>

## 6. 限制与复现

- 代码同步命令：`python3 scripts/sync_a2k_submission.py --name '王扬'`
- 轻量结果目录：`results/`
- 24G 显存证据：<引用 `results/memory_evidence.json`，填写 peak allocated/reserved、
  allocator limit/fraction 与 `within_24gib`>
- 未提交的本地大型原始文件：<只写类型和保留策略，不写内部路径>
- 已知限制：<填写>
- 最小复现步骤：<填写>

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，不得开启互联网公开访问，只保存不能公开到 GitHub 但确有审核必要
的最小差量材料；不要机械复制公开报告，也不要上传编译缓存、完整 trace、binary 或凭据。

## 自检

- [ ] 本 PR 只包含我本人本次 A2-K 的文件。
- [ ] 正式结果全部来自单张 RTX 4090 24GB，且开跑前可用显存不少于 22 GiB。
- [ ] 每个正式脚本独立、串行执行，首次 CUDA allocation 前设置 23552 MiB allocator 上限。
- [ ] README 是 Markdown 主报告，所有图片使用相对路径和有意义的 alt text。
- [ ] checkpoint、baseline、compile、正确性与 Flash benchmark 的必交结果齐全。
- [ ] PyTorch baseline 没有调用已有 fused attention。
- [ ] 提交包含学生自己编写的真实 `@triton.jit` forward kernel。
- [ ] 官方 CUDA tests 的 pass/fail/skip 如实记录。
- [ ] 每个关键数字都能回到命令、`results/` 或 metadata。
- [ ] `results/` 与 `assets/` 附件合计不超过 2 MiB，README 和单文件均未超限。
- [ ] 未提交 compile cache、PTX/CUBIN、binary、完整 trace、上游仓库或依赖环境。
- [ ] GitHub 内容不含内部主机名、IP、账号、路径、UUID、进程或未公开项目。
- [ ] GitHub 和飞书正文都不含 Secret、Token、Cookie、密码或私钥。
- [ ] 飞书补充文档为组织内公开，且未开启互联网公开访问。
