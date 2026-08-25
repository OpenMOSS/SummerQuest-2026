# A2-K：单卡显存与 GPU Kernel（田一贤）

## 完成范围与环境

本目录从固定 starter interface 独立重写，代码、结果和图表均由本次任务新鲜生成，未复用吴家兴、章之禹或其他同学的提交、结果、图片和 metadata。题面版本为 `26.1.4-k-rc.3`，固定 starter commit 为 [`ca8bc81a59b70516f7ebb2da4808daade877c736`](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736)。

本次 review 正式复跑运行在田一贤本人提交的 1×NVIDIA H100（80 GB）上，PyTorch `2.9.0+cu128`、CUDA `12.8`、Triton `3.5.0`、Python `3.12.3`。attention/Flash 使用 BF16、batch 1、causal；warm-up 5、measurement 20；checkpoint 使用 medium/24 layers、batch 1、BF16、warm-up 3、measurement 5。此前 H200 分片目录不作为本次复跑的正式证据，也不用于 H100 结果的硬件归因。

## 1. 独立实现

`submission/cs336_systems/a2k/attention.py` 的纯 PyTorch 路径只保留 `[batch, query, dim]`、`[batch, key, dim]` 和 `[batch, key, value_dim]`，逐 query tile 遍历 key/value tile。每个 tile 用 FP32 的 `m`、`l` 和输出累加器更新：

```text
m' = max(m, rowmax(S_tile))
l' = exp(m-m')*l + sum(exp(S_tile-m'))
O' = (exp(m-m')*l*O + exp(S_tile-m')*V_tile) / l'
```

causal mask 在 score tile 上应用，保存的 LSE 是唯一的 `[batch, n_queries]` 张量；反向保存 `Q/K/V/O/L` 后重建 PyTorch 图，返回 `dQ/dK/dV`。Triton kernel 由一个 query-tile program 循环 key/value tile，online-softmax 状态和 accumulator 使用 FP32，支持 causal 与 non-causal。

![原创 tiled attention schematic](assets/attention_tiles.svg)

`checkpoint_blocks` 保存 block 边界 activation，反向时重算 block 内部。block 越小通常降低 activation 峰值但增加重算和 launch 开销；最佳点由时间、临时张量和 allocator 峰值共同决定，而不是 checkpoint 数量单独决定。

![原创 checkpoint trade-off schematic](assets/checkpoint_tradeoff.svg)

## 2. Activation Checkpointing

标准 context 1024 的完整矩阵和 context 2048 边界矩阵见 [`results/checkpointing.csv`](results/checkpointing.csv)。下表列出 p50 step time / peak reserved：

| context | block 0 | block 1 | block 2 | block 4 | block 8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 83.86 ms / 10276 MiB | 142.34 / 8194 | 134.62 / 8246 | 132.43 / 8212 | 130.35 / 8258 |
| 2048 | 157.60 ms / 20212 MiB | 203.46 / 9230 | 204.12 / 9996 | 204.10 / 10528 | 204.61 / 12122 |

block 1 在两个 context 上给出最低 reserved，但 block 8 的重算代价较小，因此 1024 上 latency 逐渐回落；这正是显存—计算交换而非“checkpoint 越多越好”。

## 3. Explicit Attention 与 `torch.compile`

`results/attention_baseline.csv` 含 `S∈{512,2048,8192}`、`d∈{64,128}`、forward/backward/forward_backward 的 18/18 pass 行，每行保留 p20/p50/p80、20 个 raw samples、peak allocated/reserved。以 forward、head dim 128 为例，eager/compiled/Triton 的 p50（ms）如下：

| sequence | eager | compiled | Triton |
| ---: | ---: | ---: | ---: |
| 512 | 0.103 | 0.101 | 0.089 |
| 2048 | 0.110 | 0.097 | 0.190 |
| 8192 | 0.900 | 0.916 | 0.632 |

`results/compile_comparison.csv` 含三个代表 attention shape 的 cold-start/steady-state 对照，以及 Stanford small、B1/S512 的 eager/compiled forward、forward-backward、train-step；24/24 行 pass。H100 复跑中 S512/D64 compiled forward p50 约 0.099 ms、cold-start 约 1.55 s；small model compiled forward cold-start 约 12.0 s、steady-state 约 2.98 ms，forward-backward cold-start 约 25.8 s、steady-state 约 11.8 ms。因此不能把不同 phase 的 cold-start 或 steady-state 数字混写。

## 4. FlashAttention-2 正确性与性能

`results/flash_benchmark.csv` 的核心矩阵为 54/54 pass（3 implementations × 3 sequence × 2 head dimensions × 3 phases）；独立的 [`results/flash_boundary.csv`](results/flash_boundary.csv) 保存 16384 边界的 eager/Triton 12/12 pass 行，合计 66/66 pass。每行记录 q/k tile、warps、stages、p20/p50/p80、峰值和 speedup。Triton 配置为 `q_tile=k_tile=64, num_warps=4, num_stages=2`。H100 复跑中 Triton forward 在 S8192/D128 为约 0.632 ms；其必做重计算式 backward 会重新执行 tiled PyTorch graph，这个代价是代码设计的重计算成本，而不是把慢路径伪装成失败。

16384 边界为资源受限的 witness（warm-up 1、measurement 1，compiled 为题面允许的 optional skip）：

| head dim | implementation | forward p50 ms | backward p50 ms | forward-backward p50 ms |
| ---: | --- | ---: | ---: | ---: |
| 64 | eager | 2.937 | 3.058 | 5.819 |
| 64 | Triton | 0.899 | 55977.557 | 52389.647 |
| 128 | eager | 2.920 | 3.040 | 5.875 |
| 128 | Triton | 1.799 | 52348.577 | 48640.963 |

Triton 的边界 backward 仍然完成并标记 pass；高延迟直接反映重计算 tile 数随序列长度平方增长。

扩展正确性 [`results/correctness.json`](results/correctness.json) 覆盖 seed `7/19/41`、head dim `32/64/128`、causal/non-causal，共 18/18 pass；同时检查 output、LSE、`dQ`、`dK`、`dV`，最大误差均远低于 `rtol=atol=1e-2`。这是 `synthetic_proxy` 数学参考（代码中的 dense attention），不是数据集 ground truth；该复跑的 device metadata 为 H100。

固定 commit 的官方 attention tests 输出见 [`results/unit_tests.txt`](results/unit_tests.txt)：`6 passed in 3.79s`，包含 PyTorch 与 Triton forward/backward causal/non-causal。Triton witness 不依赖 Triton 不可用时的静默 eager fallback。

## 5. 显存证据与复现

`results/memory_evidence.json` 汇总所有核心矩阵的 allocator 口径：118/118 个正式 pass 行均记录运行时 `torch.cuda.set_per_process_memory_fraction`，峰值 allocated `19710.3 MiB`、reserved `20212.0 MiB`，allocator limit `23552 MiB`、硬限制 `24576 MiB`，`within_24gib=true`。每个正式 benchmark 子进程都在首次 CUDA allocation 前设置该 guard；`memory_evidence.py` 自身也记录 `runtime_guard_applied=true`。其中 `nvidia_smi.source` 明确写为 `pytorch_peak_reserved_proxy; nvidia-smi not collected`，不等同于整卡 `nvidia-smi` 峰值。

```bash
PYTHONPATH=students/田一贤/assignments/A2-K/submission \
python -m student_scripts.a2k.correctness --device cuda \
  --output students/田一贤/assignments/A2-K/results/correctness.json

uv run pytest tests/test_attention.py -q  # 在固定 commit 的外部 assignment2-systems 工作树中运行
```

## 飞书补充文档

组织内公开的 A2 Systems 实验指南入口（正文仍保持组织内可见）：[飞书补充文档](https://acnc6zeentra.feishu.cn/docx/D3omdgl6NocdKNxNvc5cW7KJnHd)
