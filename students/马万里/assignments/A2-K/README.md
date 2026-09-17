# A2-K：单卡显存优化与 GPU Kernels — 完成报告

- 姓名：马万里
- 硬件：单张 NVIDIA GeForce RTX 4090 24GB
- 题面版本：`26.1.4-k-rc.3`
- 固定 starter commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`
- 上游仓库位置：与 `SummerQuest-2026` 同级的 `../assignment2-systems`（未 vendor 进本仓库）

所有性能与显存数字都来自单张 RTX 4090 上、用同一套固定协议执行的正式矩阵，
可在 `results/` 中逐条回溯。

---

## 1. 完成范围

| 任务 | 上游 problem | 分值 | 状态 |
| --- | --- | ---: | --- |
| 任务一：Activation Checkpointing | `gradient_checkpointing` | 4 | 完成（理论 + 1024 矩阵 + 2048 边界） |
| 任务二：显式 PyTorch Attention | `pytorch_attention` | 2 | 完成（512/2048/8192 × 64/128） |
| 任务二：`torch.compile` 对照 | `torch_compile` | 2 | 完成（3 个代表配置 + small 模型） |
| 任务三：FlashAttention-2 前向 | `flash_forward` | 15 | 完成（tiled 参考 + 学生 Triton kernel） |
| 任务四：重计算式反向 | `flash_backward` | 5 | 完成（PyTorch/Triton 两条 autograd 路径） |
| 任务五：正确性与性能矩阵 | `flash_benchmarking` | 5 | 完成（正确性 1080 项 + 性能矩阵 72 行） |
| **合计** | | **33** | |

**范围说明**：反向由 PDF 的重计算路径实现，PyTorch 与 Triton 两个 `autograd.Function`
共用它。因此性能矩阵中 `triton` 行的 forward 是学生 Triton kernel，反向走的是这条
重计算路径。自定义 Triton backward 属于可选扩展，未实现。

**代码边界**：`cs336_systems/a2k/**/*.py`、`tests/adapters.py`、`student_scripts/a2k/**/*.py`。
本目录的 `submission/` 由 `scripts/sync_a2k_submission.py` 从 `../assignment2-systems`
同步而来；上游公共测试文件未修改，实现全部经 `tests/adapters.py` 暴露。

---

## 2. 实验环境

| 项目 | 值 | 项目 | 值 |
| --- | --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4090 | Triton | 3.6.0 |
| 总显存（CUDA 报告） | 24111.06 MiB | power limit | 450.0 W（未改动） |
| `nvidia-smi` 总显存 | 24564.0 MiB | P-state | P0 |
| 开始时空闲显存 | 23717.75 MiB | TF32 matmul | `False` |
| Driver | 560.28.03 | `float32_matmul_precision` | `highest` |
| CUDA | 12.6 | allocator 上限 | 23552 MiB（fraction 0.9768130） |
| PyTorch | 2.11.0+cu126 | 24 GiB 硬上限 | 24576 MiB |
| cuDNN | 91002 | 计时器 | `do_bench(warmup=100, rep=300, quantiles=[0.2,0.5,0.8])` |

字段来源为 `results/run_metadata.json`，其中不含 UUID、主机名、用户名、IP 或内部路径。

### 2.1 浮点精度

cuBLAS 侧关闭 TF32（`allow_tf32 = False`、`float32_matmul_precision = "highest"`），
Triton kernel 内所有 `tl.dot` 显式指定 `input_precision="ieee"`。两者作用范围不同：
`allow_tf32` 只影响 cuBLAS，而 Triton 的 FP32 `tl.dot` 默认走 TF32，必须在 kernel 内
逐点声明 `ieee`，否则 FP32 输入会在 Triton 路径上降精度。`cudnn.allow_tf32` 保持默认，
本作业不经过 cuDNN 的卷积与 RNN 路径。

### 2.2 测量协议

- **延迟**：统一用 `do_bench(warmup=100, rep=300, quantiles=[0.2,0.5,0.8])`，
  `warmup`/`rep` 单位为毫秒，返回值为毫秒。其 warm-up 循环足以吸收 `torch.compile`
  与 Triton JIT 的首次编译，因此 p20/p50/p80 是稳态分位数。
- **显存**：与延迟分开测——`do_bench` 的按时间 warm-up 会污染峰值统计。做法是
  `reset_peak_memory_stats()` 后跑 5 次，读 `max_memory_allocated()` 与
  `max_memory_reserved()`，首次调用也计入。
- **纯反向**：前向图建一次并保留（`retain_graph=True`），此后只调 `autograd.grad`，
  p20/p50/p80 是直接测得的反向分位数。反向必须持有前向激活，所以 `backward` 行的峰值
  本身已包含 forward 那一项；它与 `forward_backward` 行测的是不同的内存状态，不可相加。
- **编译边界**：compiled 配置先用一次不计时、不计峰值的冷启动步把 forward 与 backward
  图都编好，把合计 wall time 记入 `cold_start_ms`，随后 reset 峰值统计再测稳态。
  `timing_note` 字段逐行标注了每行的测法。

---

## 3. 任务一：Activation Checkpointing

### 3.1 理论

**安排方式**：对 `N` 个相同 Transformer block，最优安排是均匀分块的非嵌套 checkpoint。
把 `N` 层切成 `B = N/k` 个连续 block，每块 `k` 层，只在每个 block 的入口边界保存一次
activation；块内的 `k − 1` 个中间 activation 一律不保存，反向需要时由入口 activation
重新前向算出。不嵌套的原因：嵌套只会把重计算量再乘一个常数、把常驻量降为对数项，
在 `N = 24` 的规模下拿不到渐近收益。

**峰值位置**：某个 block 正在做块内重计算时。此时常驻 = `B` 个边界 activation +
当前重算 block 的 `k` 层内部 activation（含该层的 2D 注意力中间量）。

**渐近表达**：

- 不 checkpoint：峰值 activation memory `Θ(N)`。
- block size `k`：`M(k) = Θ(N/k + k)`，在 `k* = Θ(√N)` 处最小，为 `Θ(√N)`。
- 重计算代价：总计算量多出 `1/k` 倍的前向，`k = 1` 时最多。

**代码骨架**：

```python
def forward_with_checkpoint(model, tokens, block_size):
    hidden = model.token_embeddings(tokens)            # ← 边界 A0（常驻）
    num_blocks = math.ceil(len(model.layers) / block_size)
    for block_index in range(num_blocks):               # 遍历 B = N/k 个 block
        start = block_index * block_size
        stop = min(start + block_size, len(model.layers))

        def run_block(h, start=start, stop=stop):       # 块内 k 层整体作为一个重算单元
            for layer_index in range(start, stop):
                h = model.layers[layer_index](h)        # 块内中间 activation 不保存
            return h

        if torch.is_grad_enabled():
            hidden = torch.utils.checkpoint.checkpoint(run_block, hidden, use_reentrant=False)
        else:
            hidden = run_block(hidden)
        # ← 边界 A_{i+1}：上一个块的输出即下一个块的入口
    return model.lm_head(model.ln_final(hidden))
```

### 3.2 固定实验

Stanford medium、24 层、batch size 1、context length 1024、BF16 autocast、FP32 参数与
AdamW，每配置 3 个 warm-up step、5 个 measurement step，测量完整 training step。
`results/checkpointing.csv`：

| config_id | context | block | peak allocated (MiB) | peak reserved (MiB) | step p50 (ms) | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `medium_k1024_none` | 1024 | — | 10069.4 | 10272 | 167.79 | ok |
| `medium_k1024_block1` | 1024 | 1 | 6864.5 | 7058 | 258.14 | ok |
| `medium_k1024_block2` | 1024 | 2 | 7005.1 | 7170 | 211.18 | ok |
| `medium_k1024_block4` | 1024 | 4 | 7282.5 | 7380 | 237.59 | ok |
| `medium_k1024_block8` | 1024 | 8 | 7839.3 | 7954 | 216.95 | ok |
| `medium_k2048_none` | 2048 | — | 19662.6 | 20192 | 374.58 | ok |
| `medium_k2048_block1` | 2048 | 1 | 8061.1 | 8634 | 477.52 | ok |

context length 2048 上，`none` 峰值 19662.6 MiB（占 23 GiB 预算的 84%），标准矩阵中
peak allocated 最低的 `block1` 以 8061.1 MiB 成功，仅为 `none` 的 41%。

### 3.3 显存/时间权衡

峰值 = 固定开销（参数 + 梯度 + AdamW 状态，与 `k` 无关）+ activation 项
（≈ `B·A_residual + k·A_block`）。实测 `k2048_none − k1024_none = 9593.2 MiB`，
对应「每层 2D 注意力中间量 × 1024 个额外 token」，说明不 checkpoint 时峰值由逐层的
二次方中间量主导。单独测得的 `A_residual` 仅 4.00 MiB/层，而块内 2D 中间量
220.17 MiB/层（55×），因此减少 block 边界省下的远小于减少块内一层省下的，
最优 `k` 被推向更小的一侧。

以 `k = 1024` 的 `none` 为基准：`block1` 峰值降 32%（10069 → 6864 MiB），
step 时间涨 54%（167.79 → 258.14 ms）；`block1 → block8` 只再省 975 MiB。
收益递减的原因是固定开销约占峰值一半以上，checkpoint 压不动它。

> 五个 latency 采样中偶有离群值（见 `checkpointing.csv` 的 `step_time_ms_samples`），
> 因此 `block2`（211.18 ms）与 `block8`（216.95 ms）约 3% 的差异不宜作强结论；
> 显存结论不受影响。

![任务一 checkpointing 显存–时间权衡](assets/task1_checkpointing_tradeoff.png)

**图 1**：横轴为 block size，纵轴分别是 peak allocated 与 step p50。可见峰值随 `k`
先降后升的浅极小，以及时间随 `k` 减小的上升。

---

## 4. 任务二：PyTorch Attention 与 `torch.compile`

### 4.1 显式 PyTorch 基线

`cs336_systems/a2k/attention.py` 用显式算子写出 attention：打分 `einops.einsum`、
掩码 `torch.where(mask, s, -inf)`、softmax `cs336_basics.nn_utils.softmax`。
掩码用 `where` 而非就地 `masked_fill`，以便 `torch.compile` 完整跟踪。

batch size 1、BF16、causal，`512/2048/8192 × 64/128` 共 18 行全部 `ok`
（`results/attention_baseline.csv`）。摘录：

| seq | head_dim | phase | p50 (ms) | peak reserved (MiB) |
| ---: | ---: | --- | ---: | ---: |
| 512 | 64 | forward | 0.0358 | 280 |
| 512 | 64 | forward_backward | 0.8586 | 280 |
| 2048 | 128 | backward | 0.4987 | 362 |
| 8192 | 64 | forward | 2.1422 | 900 |
| 8192 | 128 | backward | 4.7585 | 1302 |
| 8192 | 128 | forward_backward | 6.8398 | 1174 |

显存随序列**二次方**增长：`8192/128` 的 backward 峰值是 `512/64` 的 4.6 倍，
而序列长度是 16 倍。这与 `8192/128` 上显式实现物化 `N×N` 中间量的代价一致，
也是它在 16384 上不如 Triton 的原因（见 §8.2）。

### 4.2 `torch.compile` 对照

`results/compile_comparison.csv` 摘录（`cold_start_ms` 为首次编译耗时，单独记录、
不计入稳态分位数）：

| target | (seq, d) | phase | eager p50 | compiled p50 | 加速比 | cold start (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| attention | (512, 64) | forward | 0.0369 | 0.0154 | 2.40× | 5094.1 |
| attention | (512, 64) | forward_backward | 2.9491 | 0.8637 | 3.41× | 2203.1 |
| attention | (2048, 128) | forward | 0.1137 | 0.0420 | 2.71× | 2343.1 |
| attention | (2048, 128) | forward_backward | 3.1099 | 1.0455 | 2.97× | 2264.8 |
| attention | (8192, 128) | forward | 2.1535 | 0.6748 | 3.19× | 2854.5 |
| attention | (8192, 128) | forward_backward | 6.8454 | 2.5313 | 2.70× | 2281.6 |
| small_transformer | (512, 64) | train_step | 87.8356 | 53.7876 | 1.63× | 17228.9 |

- **稳态**：attention 的 forward 有 2.4–3.2×、forward_backward 有 2.7–3.4×。
  显存同样下降，例如 `16384/64` 的 forward 峰值 reserved 从 2326 MiB 降到 1302 MiB，
  因为 Inductor 融合了 `scale → where → softmax` 并复用缓冲（见 §8.2）。
- **编译成本只付一次**：attention 配置 cold start 约 2.2–5.1 s；small 模型的
  `train_step` 达 17.2 s。表格把 cold start 与稳态分位数分列，避免混入 latency。
- **`forward_backward` 不等于 forward 与 backward 之和**：例如 `(2048,128)`，
  eager 为 0.1137 + 0.4987 = 0.6124 ms 而实测 3.1099 ms。原因是该行的 step 含
  `autograd.grad` 的图构建与梯度分配，且独立分位数不可相加。
- **shape specialization**：compile 对 `(seq, head_dim, mask)` 特化。每个配置用独立
  进程与独立 shape，编译缓存进程内命中一次、跨进程不共享，所以 cold start 都在 2–3 s
  量级；`scaled_dot_product_attention` 全程是纯张量算子，未观察到 graph break。
- **稳定性**：compiled 的 p20/p50/p80 很窄（`(2048,128)` forward 三者均为 0.0420），
  而 `train_step` 的 p20–p80 为 `50.98/53.79/55.36`，符合端到端含 embedding、loss 与
  optimizer 的预期。

---

## 5. 任务三：FlashAttention-2 前向

`cs336_systems/a2k/flash_attention.py` 有两套等价实现：
`_pytorch_tiled_forward` / `FlashAttentionPytorchFunction`（纯 PyTorch tiled 参考）
与 `flash_fwd_kernel` / `FlashAttentionTritonFunction`（学生 Triton kernel）。

### 5.1 tile 划分与 program 映射

- **一个 program 负责一个 query tile**：外层按 `Q_TILE_SIZE = 64` 分块，每个 program
  只处理自己的 `Q_i`，写出对应的 `O_i` 与 `L_i`，program 之间无通信、无原子操作。
- **KV tile 在 kernel 内循环**：`K_TILE_SIZE = 64`，每个 query tile 串行遍历
  `ceil(n_keys / 64)` 个 KV tile。这是 FlashAttention-2 相对 v1 的关键改动——KV 循环
  放在单个 program 内部，避免 v1 里跨 program 归约 `O` 的额外读写。
- 三个累加器常驻片上：`acc_o`（FP32，`[Q_TILE, D]`）、`m_i`（FP32，`[Q_TILE]`）、
  `l_i`（FP32，`[Q_TILE]`）。`S` 的一个 tile（`[64,64]`）只在片上存活一轮，
  **完整的 `N×N` 矩阵从不物化**——这是显存随序列线性而非二次方的原因。

### 5.2 online softmax

按 PDF 算法 1 逐步实现（公式编号与 PDF 一致）：

| 算法 1 | 公式 | 代码位置 |
| --- | --- | --- |
| 第 9 步 | `S_i^(j) = Q_i (K^(j))^T / sqrt(d)` | `tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale` |
| 第 10 步 | `m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))` | `tl.max(scores, axis=1)`；`tl.maximum(m_i, tile_m)` |
| 第 11 步 | `P~_i^(j) = exp(S_i^(j) − m_i^(j))` | `tl.where(row_has_key[:, None], tl.exp(scores - new_m[:, None]), 0.0)` |
| 第 12 步 | `l_i^(j) = exp(m_i^(j-1) − m_i^(j))·l_i^(j-1) + rowsum(P~)` | `l_i = l_i * alpha + tl.sum(probs, axis=1)` |
| 第 13 步 | `O_i^(j) = diag(exp(m_i^(j-1) − m_i^(j)))·O_i^(j-1) + P~_i^(j) V^(j)` | `acc_o = acc_o * alpha[:, None] + tl.dot(probs, v_tile, input_precision="ieee")` |
| 第 15 步 | `O_i = diag(l_i^(T_k))^-1 · O_i^(T_k)` | `acc_o = acc_o / l_i[:, None]` |
| 第 16 步 | `L_i = m_i^(T_k) + log(l_i^(T_k))` | `lse = m_i + tl.log(l_i)` |

数值稳定性的关键是第 12/13 步的 rescale：每轮发现更大的运行最大值时，用
`exp(m_old − m_new) ≤ 1` 把已有的 `acc_o` 与 `l_i` 换算到新基准，因此不会出现
`exp(大数)` 上溢。`exp` 与累加链都在 FP32 上计算，输入即使是 BF16 也不影响。

### 5.3 causal mask

mask 由 query/key 的**全局位置索引**比较得到，而不是 tile 内的相对位置：

```python
q_index = q_start + tl.arange(0, Q_TILE_SIZE)      # 全局 query 位置
k_index = k_start + tl.arange(0, K_TILE_SIZE)      # 全局 key 位置
scores = tl.where(q_index[:, None] >= k_index[None, :], scores, float("-inf"))
```

用 tile 内相对位置会在 `q_start > k_start` 时误屏蔽本该可见的位置。本实现保守地逐块
计算并用 `-inf` 屏蔽，使 causal 与 non-causal 共用同一份 kernel 代码
（`is_causal` 是 `tl.constexpr`，分支在编译期消解）。

当整行都被 mask 时（例如尾部 query），`exp(-inf) = 0` 会让 `l_i = 0`，随后
`acc_o / l_i` 产生 `NaN`。实现用 `row_has_key = tl.max(valid, axis=1)` 判断该行是否有
可见 key，无可见 key 时取 `alpha = 1.0`、`probs = 0.0`，得到 `O_i = 0`、`L_i = -inf`，
与 PyTorch 参考在整行 `-inf` 下的 softmax 行为一致。

### 5.4 精度与保存张量

- 所有 `tl.dot` 显式传 `input_precision="ieee"`（见 §2.1）。
- 反向保存 `q, k, v, output, lse`，其中 `lse` 是唯一的 `[batch, n_queries]` 张量；
  不保存 `S` 或 `P`（都是 `[batch, N, N]`）。
- 两个 `autograd.Function` 的 `forward` 都带默认值为 `False` 的 `is_causal`，
  并用 `ctx.is_causal` 把 mask 标志留给反向。

---

## 6. 任务四：重计算式反向

反向只依赖前向保存的 `Q、K、V、O、L`：

1. **前置量** `D_i = rowsum(O_i ∘ dO_i)`。它恒等于 `rowsum(P_i ∘ dP_i)`，
   因此不需要物化 `P` 就能拿到 softmax 反向所需的行和。
2. **用保存的 `L` 重建 `P`**：`P_ij = exp(S_ij − L_i)`，省掉重做 online softmax 分母。
3. **累积梯度**（PDF 式 13–19）：`dP = dO V^T`、`dS = P ∘ (dP − D)`、
   `dQ = dS K / sqrt(d)`、`dK = dS^T Q / sqrt(d)`、`dV = P^T dO`。
4. `backward` 返回 `d_q, d_k, d_v, None`，与输入 `(q, k, v, is_causal)` 对齐。

`_recomputed_backward` 是 autograd 路径实际调用的实现，把上述公式写成整块张量运算；
`_tiled_recomputed_backward` 严格按 tile 遍历 query 与 key，作为逐行对照 PDF 公式的
参考实现，两者输出一致（FP32 下 `dQ` 完全相同，`dK`/`dV` 最大绝对差 2.44e-4 / 1.22e-4，
来自累加顺序）。

PyTorch tiled 与 Triton 两个 `autograd.Function` 共用这条重计算反向，因此矩阵里
`triton` 行的反向没有另写 Triton 反向 kernel。

---

## 7. 测试与正确性

### 7.1 官方 tests

```bash
srun -p fnlp-4090 --gres=gpu:1 python -m pytest tests/test_attention.py -v
```

`results/unit_tests.txt`：**6 passed, 0 failed, 0 skipped**。逐用例为
`test_flash_forward_pass_pytorch`、`test_flash_forward_pass_triton[False]`、
`test_flash_forward_pass_triton[True]`、`test_flash_backward_pytorch`、
`test_flash_backward_triton[False]`、`test_flash_backward_triton[True]`。

### 7.2 扩展正确性

`results/correctness.json` 共 **1080 项，全部 `pass`**：

| 维度 | 取值 |
| --- | --- |
| seed | 0、1、2 |
| head_dim | 32、64、128 |
| dtype | float32（540 项）、bfloat16（540 项） |
| causal | True、False |
| implementation | `pytorch_tiled`、`triton` |
| quantity | `forward_output`、`lse`、`dQ`、`dK`、`dV` |
| sequence_length | 128、512、2048 |

判据为逐元素混合容差 `|a − b| ≤ atol + rtol·|b|`，并要求超限元素数为 0：

- **FP32**：`atol = rtol = 1e-4`，实测最大绝对误差 1.48e-5；最大相对误差 8.26e-2
  出现在接近 0 的参考元素上，由 `atol` 项吸收，超限元素数仍为 0。
- **BF16**：`atol = rtol = 2e-2`，实测最大绝对误差 1.56e-2。

---

## 8. 任务五：性能矩阵

batch size 1、BF16、causal。核心矩阵为 `512/2048/8192`（三种实现都参加），
另加 `16384` 长序列边界。speedup 只在实现之外的条件完全相同、且参照行自身 `ok`
时计算。

### 8.1 核心矩阵（`results/flash_benchmark.csv`）

| seq | head_dim | phase | 实现 | p50 (ms) | peak reserved (MiB) | 相对 eager | status |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 512 | 64 | fwd | eager | 0.0369 | 24 | 1.00× | ok |
| 512 | 64 | fwd | compiled | 0.0205 | 22 | 1.80× | ok |
| 512 | 64 | fwd | triton | 0.7629 | 2 | 0.05× | ok |
| 512 | 64 | bwd | eager | 0.4925 | 26 | 1.00× | ok |
| 512 | 64 | bwd | compiled | 0.1705 | 24 | 2.89× | ok |
| 512 | 64 | bwd | triton | 0.5361 | 28 | 0.92× | ok |
| 512 | 64 | fwd+bwd | eager | 3.1058 | 24 | 1.00× | ok |
| 512 | 64 | fwd+bwd | compiled | 0.3840 | 24 | 8.09× | ok |
| 512 | 64 | fwd+bwd | triton | 1.9579 | 28 | 1.59× | ok |
| 512 | 128 | fwd | eager | 0.0420 | 24 | 1.00× | ok |
| 512 | 128 | fwd | compiled | 0.0236 | 22 | 1.78× | ok |
| 512 | 128 | fwd | triton | 1.8801 | 2 | 0.02× | ok |
| 512 | 128 | bwd | eager | 0.4956 | 26 | 1.00× | ok |
| 512 | 128 | bwd | compiled | 0.1823 | 24 | 2.72× | ok |
| 512 | 128 | bwd | triton | 0.5386 | 28 | 0.92× | ok |
| 512 | 128 | fwd+bwd | eager | 3.1447 | 26 | 1.00× | ok |
| 512 | 128 | fwd+bwd | compiled | 0.3840 | 24 | 8.19× | ok |
| 512 | 128 | fwd+bwd | triton | 1.9671 | 28 | 1.60× | ok |
| 2048 | 64 | fwd | eager | 0.1014 | 62 | 1.00× | ok |
| 2048 | 64 | fwd | compiled | 0.0399 | 42 | 2.54× | ok |
| 2048 | 64 | fwd | triton | 2.7720 | 2 | 0.04× | ok |
| 2048 | 64 | bwd | eager | 0.4987 | 104 | 1.00× | ok |
| 2048 | 64 | bwd | compiled | 0.2068 | 84 | 2.41× | ok |
| 2048 | 64 | bwd | triton | 0.5417 | 106 | 0.92× | ok |
| 2048 | 64 | fwd+bwd | eager | 3.1575 | 104 | 1.00× | ok |
| 2048 | 64 | fwd+bwd | compiled | 0.9554 | 64 | 3.30× | ok |
| 2048 | 64 | fwd+bwd | triton | 3.0085 | 106 | 1.05× | ok |
| 2048 | 128 | fwd | eager | 0.1126 | 64 | 1.00× | ok |
| 2048 | 128 | fwd | compiled | 0.0471 | 44 | 2.39× | ok |
| 2048 | 128 | fwd | triton | 6.8485 | 4 | 0.02× | ok |
| 2048 | 128 | bwd | eager | 0.4977 | 106 | 1.00× | ok |
| 2048 | 128 | bwd | compiled | 0.1382 | 86 | 3.60× | ok |
| 2048 | 128 | bwd | triton | 0.5407 | 112 | 0.92× | ok |
| 2048 | 128 | fwd+bwd | eager | 3.1171 | 106 | 1.00× | ok |
| 2048 | 128 | fwd+bwd | compiled | 0.8817 | 66 | 3.54× | ok |
| 2048 | 128 | fwd+bwd | triton | 7.1322 | 112 | 0.44× | ok |
| 8192 | 64 | fwd | eager | 2.1340 | 602 | 1.00× | ok |
| 8192 | 64 | fwd | compiled | 0.6595 | 346 | 3.24× | ok |
| 8192 | 64 | fwd | triton | 25.2682 | 6 | 0.08× | ok |
| 8192 | 64 | bwd | eager | 4.7196 | 1034 | 1.00× | ok |
| 8192 | 64 | bwd | compiled | 1.8422 | 650 | 2.56× | ok |
| 8192 | 64 | bwd | triton | 6.0257 | 1310 | 0.78× | ok |
| 8192 | 64 | fwd+bwd | eager | 6.7697 | 862 | 1.00× | ok |
| 8192 | 64 | fwd+bwd | compiled | 2.4678 | 478 | 2.74× | ok |
| 8192 | 64 | fwd+bwd | triton | 31.3016 | 1310 | 0.22× | ok |
| 8192 | 128 | fwd | eager | 2.1514 | 598 | 1.00× | ok |
| 8192 | 128 | fwd | compiled | 0.6932 | 342 | 3.10× | ok |
| 8192 | 128 | fwd | triton | 60.1395 | 22 | 0.04× | ok |
| 8192 | 128 | bwd | eager | 4.7575 | 1002 | 1.00× | ok |
| 8192 | 128 | bwd | compiled | 1.8872 | 618 | 2.52× | ok |
| 8192 | 128 | bwd | triton | 6.3713 | 1322 | 0.75× | ok |
| 8192 | 128 | fwd+bwd | eager | 6.8475 | 982 | 1.00× | ok |
| 8192 | 128 | fwd+bwd | compiled | 2.5283 | 542 | 2.71× | ok |
| 8192 | 128 | fwd+bwd | triton | 66.5697 | 1322 | 0.10× | ok |

**主要观察**：

- **compiled 的 forward 与 backward 都快于 eager**（forward 1.8–3.2×，
  backward 2.4–2.9×），forward_backward 的差距更大（2.6–8.2×）。显存同样下降：
  `16384` 的 forward 峰值 reserved 从 2326 MiB 降到 1302 MiB，`8192` 从 602 降到 346。
- **Triton 的 forward 明显慢于 eager**（比值 0.02–0.09）。短序列时 kernel launch 与
  tiled 循环的固定开销占主导，长序列时代价来自 `tl.dot(..., input_precision="ieee")`
  走非张量核路径。它的价值在显存：`512` 时峰值只有 2 MiB（eager 24 MiB），
  `8192/128` 时 22 MiB（eager 598 MiB），因为完全不物化 `N×N` 中间量。
- **Triton 的 backward 与 eager 接近**（0.75–0.92×）：它走的是 §6 的 PyTorch 张量运算
  反向，而不是手写 fused kernel，因此没有 forward 那种量级的差距。其显存峰值略高于
  eager，原因是重算时 `P` 在整块张量上一次成形（见 §6 与 §10）。

### 8.2 长序列 16384 边界

| seq | head_dim | phase | 实现 | p50 (ms) | peak reserved (MiB) | 相对 eager | status |
| ---: | ---: | --- | --- | ---: | ---: | ---: | --- |
| 16384 | 64 | fwd | eager | 8.3302 | 2326 | 1.00× | ok |
| 16384 | 64 | fwd | triton | 94.6176 | 22 | 0.09× | ok |
| 16384 | 64 | bwd | eager | 18.6163 | 3882 | 1.00× | ok |
| 16384 | 64 | bwd | triton | 24.0579 | 5162 | 0.77× | ok |
| 16384 | 64 | fwd+bwd | eager | 26.8861 | 3862 | 1.00× | ok |
| 16384 | 64 | fwd+bwd | triton | 118.4123 | 5162 | 0.23× | ok |
| 16384 | 128 | fwd | eager | 8.3671 | 2346 | 1.00× | ok |
| 16384 | 128 | fwd | triton | 240.8868 | 22 | 0.03× | ok |
| 16384 | 128 | bwd | eager | 18.7597 | 4118 | 1.00× | ok |
| 16384 | 128 | bwd | triton | 25.1197 | 5182 | 0.75× | ok |
| 16384 | 128 | fwd+bwd | eager | 27.0843 | 3882 | 1.00× | ok |
| 16384 | 128 | fwd+bwd | triton | 266.1335 | 5182 | 0.10× | ok |

compiled 在该边界为可选，也一并跑通：`16384/128` 的 forward / backward /
forward_backward 分别为 2.728 / 7.576 / 10.259 ms。

**最关键的结论在显存**：`16384/128` 的 forward，eager 峰值 reserved 2346 MiB，
Triton 只有 **22 MiB**（约 107×）。eager 的 8.37 ms 与 Triton 的 240.89 ms 说明二者是
不同的权衡——显式实现靠 `N×N` 物化换取更少的小 kernel 调用，Triton 反之。
在 16384 上 Triton forward 的显存近似常数增长（16384 与 512 时都是 22 MiB），
符合「显存随序列线性、与 `N²` 无关」的设计目标。

Triton 的 backward 峰值（5162/5182 MiB）高于 eager（3882/4118 MiB），
原因是 §6 的重算路径会逐块成形 `P`；这属于反向实现的取舍，不影响 forward 的结论。

### 8.3 失败行

72 行性能矩阵与 1080 项扩展正确性**全部成功**，无 OOM、无编译失败，
因此 `results/flash_benchmark.csv` 中没有非 `ok` 行。`speedup` 列的“—”只出现在
缺少同 shape eager 参照的情况。

---

## 9. 可追溯性

### 9.1 关键数字 → 结果文件

| 内容 | 结果文件 | 复现命令（在 `../assignment2-systems` 下） |
| --- | --- | --- |
| checkpoint 矩阵 | `results/checkpointing.csv` | `bash run_task1_checkpointing.sh` |
| 显式 attention 基线 | `results/attention_baseline.csv` | `bash run_task2_attention.sh` |
| eager/compiled 对照 | `results/compile_comparison.csv` | `bash run_task2_attention.sh` |
| 官方 tests | `results/unit_tests.txt` | `srun -p fnlp-4090 --gres=gpu:1 python -m pytest tests/test_attention.py -v` |
| 扩展正确性 | `results/correctness.json` | `python student_scripts/a2k/task5_correctness.py --output local_results/a2k/task5/correctness.json --length 128 512 2048` |
| 性能矩阵 | `results/flash_benchmark.csv` | `bash run_task5_flash.sh` |
| 显存汇总 | `results/memory_evidence.json` | `python student_scripts/a2k/summarize_memory_evidence.py --output local_results/a2k/memory_evidence.json` |
| 环境 metadata | `results/run_metadata.json` | `bash run_task5_flash.sh`（第一步即生成） |

`local_results/a2k/` 保留本地原始结果，不整体提交；`results/` 只放轻量汇总。

### 9.2 显存汇总

`results/memory_evidence.json`：

| 字段 | 值 |
| --- | --- |
| `allocator.allocator_fraction` | 0.9768130292889415 |
| `allocator.allocator_limit_mib` | 23552 |
| `hard_limit_mib` | 24576 |
| `pytorch_peak_allocated_mib` | 19662.606 |
| `pytorch_peak_reserved_mib` | 20192.0 |
| `within_24gib` | true |
| 成功配置总数 | 97 |

最高峰值来自任务一 `k=2048` 的 no-checkpoint 配置（reserved 20192 MiB），
即 activation checkpointing 要压下去的数字。`within_24gib` 由
`peak_reserved ≤ 23552` 且 `≤ 24576` 判定。`fraction` 取
`min(1.0, 23552 / total_bytes)`，在每个正式进程里都在**第一次 CUDA allocation 之前**
设置，各配置独立进程、串行执行。

### 9.3 图

| 图 | 文件 | 内容 |
| --- | --- | --- |
| 图 1 | `assets/task1_checkpointing_tradeoff.png` | block size 的显存–时间权衡 |
| 图 2 | `assets/task5_latency.png` | 三种实现的 p50 延迟 vs 序列长度 |
| 图 3 | `assets/task5_speedup.png` | 相对 eager 的 speedup |
| 图 4 | `assets/task5_memory.png` | 峰值显存 vs 序列长度 |

![三种实现的 p50 延迟](assets/task5_latency.png)

**图 2**：横轴序列长度、纵轴 p50 延迟。

![相对 eager 的加速比](assets/task5_speedup.png)

**图 3**：compiled 在短序列上优势明显，Triton 的 forward 随序列增长显现收益。

![峰值显存 vs 序列长度](assets/task5_memory.png)

**图 4**：eager/compiled 呈二次方趋势，Triton 的 forward 近似线性。

---

## 10. 测量口径说明

- **`backward` 与 `forward_backward` 的峰值不可相加**：`backward` 测的是保留图上的
  反向，前向激活在计数窗口内一直存活，所以它本身已包含 forward 那一项。
- **`triton` 行的反向是重计算路径**：该行 forward 是 Triton kernel，
  backward / `forward_backward` 里含一段 PyTorch 张量运算实现的反向（§6）；
  eager 与 compiled 行的反向则由 PyTorch autograd 自动生成，三者实现并不等价。
- **附件脱敏**：`results/` 中逐次运行的 metadata 使用工作区相对路径；
  环境 metadata 只保留 GPU 型号、Driver、CUDA/PyTorch/Triton 版本、power limit 与
  P-state 等公开字段。

---

## 11. 飞书补充文档

[A2-K 补充文档（飞书）](https://fudan-nlp.feishu.cn/wiki/AGsnw0pfsihJaEkb8rucRB79nOg)

---

## 附录：文件清单

| 路径 | 内容 |
| --- | --- |
| `README.md` | 本报告 |
| `assets/task1_checkpointing_tradeoff.png` | 图 1 |
| `assets/task5_latency.png` | 图 2 |
| `assets/task5_speedup.png` | 图 3 |
| `assets/task5_memory.png` | 图 4 |
| `results/checkpointing.csv` | 任务一固定矩阵 |
| `results/attention_baseline.csv` | 任务二显式基线 |
| `results/compile_comparison.csv` | 任务二 compiled 对照 |
| `results/correctness.json` | 任务五扩展正确性 1080 项 |
| `results/flash_benchmark.csv` | 任务五性能矩阵 72 行 |
| `results/memory_evidence.json` | 全作业显存汇总 |
| `results/run_metadata.json` | 脱敏环境与测量配置 |
| `results/unit_tests.txt` | 官方 GPU tests 输出 |
| `results/attention_baseline_metadata.jsonl` | 任务二基线逐次运行 metadata |
| `results/compile_comparison_metadata.jsonl` | 任务二 compiled 逐次运行 metadata |
| `submission/cs336_systems/a2k/*.py` | 学生实现 |
| `submission/student_scripts/a2k/*.py` | benchmark、正确性与汇总脚本 |
| `submission/tests/adapters.py` | 连接官方 tests 的 adapter |
