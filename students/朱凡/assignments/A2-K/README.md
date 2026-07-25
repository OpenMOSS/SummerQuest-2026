# A2-K：单 GPU 显存与 GPU Kernel

## 范围与可复现性

本报告覆盖学生实现的 PyTorch 分块注意力参考实现、Triton 前向
kernel、重计算式反向传播、激活检查点、显式 eager attention、
`torch.compile`、正确性测试和性能矩阵。以下所有结论均来自
`results/` 中的文件，没有手工填写实验数值。

正式实验 metadata 报告的 GPU 为 **NVIDIA GeForce RTX 4090**，
总显存为 24564 MiB，PyTorch 版本为 2.11.0+cu126，CUDA 版本为
12.6，Triton 版本为 3.4.0。PyTorch allocator 上限为 23552 MiB。
公开报告不包含主机名、用户名、进程 ID、UUID 或内部路径。

实验记录的工作树 commit：`ec070562f565aa63495930ac9845e94d51b3fee9`。
Starter commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`。
课程补充文档：https://acnc6zeentra.feishu.cn/docx/D3omdgl6NocdKNxNvc5cW7KJnHd

## 激活检查点

对于由 `N` 个 Transformer block 组成的序列，每隔 `B` 个 block 设置
一个边界，只保存边界处的 hidden state。反向传播时，从前一个边界开始
重新计算当前区间，然后计算该区间的局部梯度。这里优先使用非嵌套
checkpoint，因为嵌套 checkpoint 会重复边界管理，并可能增加 kernel
启动开销。对于简单的均匀调度，激活显存上界为
`O((N/B + B) * activation_per_block)`；重计算带来的总工作量为
`O(N)` 的前向工作加上 `O(N)` 的反向重计算工作。实际最佳的 `B`
还取决于重计算 FLOPs、边界张量、kernel 启动开销、allocator 碎片、
block 内部峰值和显存带宽。

```python
hidden = embedding(tokens)
for start in range(0, N, B):
    hidden = checkpoint(run_blocks[start:start + B], hidden)
loss = head(norm(hidden))
loss.backward()
```

Checkpoint 结果状态：**ok=10**。

| 配置 ID | 上下文长度 | checkpoint block 大小 | 嵌套 | p50_ms | 峰值已分配显存 MiB | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| medium_L24_c1024_g0 | 1024 | 0 | False | 143.60785111784935 | 10064.2431640625 | ok |
| medium_L24_c1024_g1 | 1024 | 1 | False | 218.64781063050032 | 8116.52783203125 | ok |
| medium_L24_c1024_g2 | 1024 | 2 | False | 199.27192572504282 | 8117.46533203125 | ok |
| medium_L24_c1024_g4 | 1024 | 4 | False | 195.8522815257311 | 8117.46533203125 | ok |
| medium_L24_c1024_g8 | 1024 | 8 | False | 199.5816770941019 | 8117.46533203125 | ok |
| medium_L24_c2048_g0 | 2048 | 0 | False | 379.8650400713086 | 19661.5595703125 | ok |
| medium_L24_c2048_g1 | 2048 | 1 | False | 483.5942564532161 | 8136.78564453125 | ok |
| medium_L24_c2048_g2 | 2048 | 2 | False | 485.21011415869 | 8568.2939453125 | ok |
| medium_L24_c2048_g4 | 2048 | 4 | False | 485.9895845875144 | 9575.1845703125 | ok |
| medium_L24_c2048_g8 | 2048 | 8 | False | 486.6200825199485 | 11594.2470703125 | ok |

## Attention 与 Compile 对比

显式 eager 实现使用普通 PyTorch 操作依次完成 `QK^T`、缩放、
因果 mask、softmax 和 `PV`。Compile 对比会单独记录首次编译时间和
稳定态时间，并保留编译错误。分块 PyTorch 路径保存形状为
`[batch, num_queries]` 的 LSE 张量；Triton 路径为每个 query tile
启动一个 program，循环处理 key tile，使用 FP32 的 online max/sum
状态，并同时处理 query 和 key 边界 mask。

Attention baseline 状态：**ok=18**。

| 实现 | 序列长度 | head 维度 | 阶段 | p50_ms | 峰值已分配显存 MiB | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| eager | 512 | 64 | forward | 0.18943999707698822 | 9.6328125 | ok |
| eager | 512 | 64 | backward | 0.5928959846496582 | 18.875 | ok |
| eager | 512 | 64 | forward_backward | 0.6041600108146667 | 18.875 | ok |
| eager | 512 | 128 | forward | 0.19046400487422943 | 18.0078125 | ok |
| eager | 512 | 128 | backward | 0.5898240208625793 | 19.25 | ok |
| eager | 512 | 128 | forward_backward | 0.6031360030174255 | 19.25 | ok |
| eager | 2048 | 64 | forward | 0.19046400487422943 | 37.28125 | ok |
| eager | 2048 | 64 | backward | 0.5888000130653381 | 53.75 | ok |
| eager | 2048 | 64 | forward_backward | 0.6021119952201843 | 53.75 | ok |
| eager | 2048 | 128 | forward | 0.18534399569034576 | 38.28125 | ok |
| eager | 2048 | 128 | backward | 0.5847039818763733 | 55.25 | ok |
| eager | 2048 | 128 | forward_backward | 0.5959680080413818 | 55.25 | ok |

_这里只展示 18 行中的 12 行，完整表格见 `results/`。_

Compile 对比状态：**error=5，ok=37**。

| 实验 | 实现 | 模式 | 是否 compiled | p50_ms | cold_start_ms | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
|  | eager |  |  | 0.18432000279426575 |  | ok |
|  | eager |  |  | 0.5713919997215271 |  | ok |
|  | eager |  |  | 0.5806080102920532 |  | ok |
|  | eager |  |  | 0.18636800348758698 |  | ok |
|  | eager |  |  | 0.5713919997215271 |  | ok |
|  | eager |  |  | 0.5826560258865356 |  | ok |
|  | eager |  |  | 0.18636800348758698 |  | ok |
|  | eager |  |  | 0.5703679919242859 |  | ok |
|  | eager |  |  | 0.5826560258865356 |  | ok |
|  | eager |  |  | 0.18227200210094452 |  | ok |
|  | eager |  |  | 0.5652480125427246 |  | ok |
|  | eager |  |  | 0.5754879713058472 |  | ok |

_这里只展示 42 行中的 12 行，完整表格见 `results/`。_

## 正确性

正确性矩阵覆盖 seed 0、1、2，head 维度 32、64、128，因果和非因果
attention，以及输出、LSE、`dQ`、`dK` 和 `dV`。PyTorch 与 Triton
行分别报告；GPU 测试被 skip 时不会被改写为 pass。

正确性状态：**pass=36**。

| 实现 | seed | head 维度 | causal | 状态 |
| --- | --- | --- | --- | --- |
| pytorch | 0 | 32 | False | pass |
| pytorch | 0 | 32 | True | pass |
| pytorch | 0 | 64 | False | pass |
| pytorch | 0 | 64 | True | pass |
| pytorch | 0 | 128 | False | pass |
| pytorch | 0 | 128 | True | pass |
| pytorch | 1 | 32 | False | pass |
| pytorch | 1 | 32 | True | pass |
| pytorch | 1 | 64 | False | pass |
| pytorch | 1 | 64 | True | pass |
| pytorch | 1 | 128 | False | pass |
| pytorch | 1 | 128 | True | pass |

_这里只展示 36 行中的 12 行，完整表格见 `results/`。_

## 正式性能矩阵与长序列边界

核心矩阵使用 batch size 1、BF16、causal attention，序列长度
512/2048/8192，维度 64/128，并分别测量 forward、backward 和
forward-backward 阶段。即使 eager 在 16384 上 OOM，也会保留对应行；
Triton 会独立尝试。

Flash benchmark 状态：**error=7，ok=65**。

| 实现 | 序列长度 | head 维度 | 阶段 | p50_ms | 相对 eager 的加速比 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| eager | 512 | 64 | forward | 0.1812479943037033 | 1.0 | ok |
| eager | 512 | 64 | backward | 0.5693439841270447 | 1.0 | ok |
| eager | 512 | 64 | forward_backward | 0.5775359869003296 | 1.0 | ok |
| eager | 512 | 128 | forward | 0.1812479943037033 | 1.0 | ok |
| eager | 512 | 128 | backward | 0.5693439841270447 | 1.0 | ok |
| eager | 512 | 128 | forward_backward | 0.579584002494812 | 1.0 | ok |
| eager | 2048 | 64 | forward | 0.18227200210094452 | 1.0 | ok |
| eager | 2048 | 64 | backward | 0.5683199763298035 | 1.0 | ok |
| eager | 2048 | 64 | forward_backward | 0.5785599946975708 | 1.0 | ok |
| eager | 2048 | 128 | forward | 0.17715199291706085 | 1.0 | ok |
| eager | 2048 | 128 | backward | 0.5642240047454834 | 1.0 | ok |
| eager | 2048 | 128 | forward_backward | 0.5744640231132507 | 1.0 | ok |

_这里只展示 72 行中的 12 行，完整表格见 `results/`。_

## 证据与复现实验命令

- `results/run_metadata.json`
- `results/checkpointing.csv`
- `results/attention_baseline.csv`
- `results/compile_comparison.csv`
- `results/flash_benchmark.csv`
- `results/correctness.json`
- `results/unit_tests.txt`
- `results/memory_evidence.json`
- ![checkpoint timing](assets/checkpoint_time.svg)
- ![attention latency](assets/attention_latency.svg)
- ![peak memory](assets/memory_peak.svg)

```bash
python scripts/run_a2k_checkpoint_matrix.py
python scripts/a2k_correctness.py --output results/correctness.json
python scripts/attention_benchmark.py --output results/flash_benchmark.csv
```
