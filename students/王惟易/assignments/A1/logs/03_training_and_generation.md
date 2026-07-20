# A1 学习记录 03：训练基础设施与生成

- 日期：2026-07-16
- 状态：训练与生成基础设施完成，进入正式实验

## 1. 里程碑结论

这一阶段已经从“模型能完成前向传播”推进到了“模型可以被训练、保存、恢复和采样”：

- 实现了数值稳定的交叉熵、AdamW、余弦学习率调度、全局梯度裁剪和批数据采样。
- 实现了训练步、验证、checkpoint、JSONL 指标记录，以及从 checkpoint 继续训练的完整训练入口。
- 实现了 temperature、top-p 和自回归生成，并提供了加载 tokenizer、模型配置与 checkpoint 的生成入口。
- 单批次过拟合实验将 loss 从 `3.9891` 降至 `0.00196`，证明从数据、模型、损失、反向传播到优化器更新的主链路能够工作。
- 官方测试结果为 `47 passed, 1 xfailed`；唯一的 xfail 是预期中的 `Tokenizer.encode` 内存测试。

这只是“训练系统可以工作”的里程碑，而不是“模型已经训练好”。正式语料训练、超参数比较、消融实验和生成质量分析仍是下一阶段的核心任务。

## 2. 学习路径

```text
数值稳定的 cross entropy
└── AdamW
    └── cosine learning-rate schedule
        └── global gradient clipping
            └── mmap-aware batch sampling
                └── checkpoint save/load
                    └── train/evaluate loop
                        └── JSONL logging 与 resume
                            └── temperature 与 top-p sampling
                                └── autoregressive generation
                                    └── single-batch overfit
                                        └── 正式训练与实验
```

## 3. 关键理论与实现理解

### 3.1 数值稳定的交叉熵

直接计算 softmax 再取对数会遇到指数溢出和小概率下溢。稳定实现先从每行 logits 中减去最大值，等价地使用 `logsumexp`：

$$
\operatorname{CE}(x, y) = \log \sum_j e^{x_j} - x_y.
$$

目标 token 的 logit 可以通过 `torch.gather` 沿最后一维取出。若 logits 形状为 `(..., vocab_size)`，targets 形状为 `(...)`，先把 targets 扩成 `(..., 1)`，gather 后再去掉最后那个长度为 1 的维度。这样实现不依赖固定的 batch 或 sequence 维数。

### 3.2 AdamW 的状态与更新

AdamW 为每个可训练参数维护两个同形状状态：

- `m`：梯度的一阶指数滑动平均，即 momentum-like first moment。
- `v`：梯度平方的二阶原始矩指数滑动平均，并不是 velocity。

因此，仅看参数和优化器状态，AdamW 状态额外占参数本身的两倍。若参数、梯度、`m`、`v` 都使用相同精度且暂不计算激活值，总量约为参数存储的四倍。

AdamW 的 `W` 表示 weight decay 与梯度更新解耦。权重衰减直接作用于参数，而不是先把 `\lambda \theta` 加入梯度再交给 Adam 的自适应归一化。没有梯度的参数应跳过更新。

这一实现也让我第一次系统使用 PyTorch 的原地运算。优化器更新通常位于 `torch.no_grad()` 语境中，原地修改参数与状态可以避免为每一步创建不必要的新张量；但在仍需 autograd 追踪的普通前向计算中，原地运算需要谨慎使用。

### 3.3 学习率调度的 step 语义

余弦调度分为 warmup、cosine decay 和最终学习率三个区间。最容易出错的不是公式，而是 step 的含义：训练循环中的 `start_step` 表示已经完成的 optimizer update 数量，因此恢复训练时，下一个循环 step 与学习率查询必须延续同一套计数约定，不能重复或跳过一次更新。

### 3.4 全局梯度裁剪

全局梯度裁剪先把所有参数梯度视为一个长向量，计算共同的 $L_2$ 范数。若范数超过阈值，所有梯度乘同一个缩放系数：

$$
g \leftarrow g \cdot \frac{\text{max\_norm}}{\lVert g \rVert_2}.
$$

重点是“全局”和“同一个系数”。若逐参数分别裁剪，会改变不同参数梯度之间的相对大小，从而不再是对整体梯度方向的等比例缩放。

### 3.5 数据采样与 mmap

语言模型训练样本由连续 token 序列构成。随机起点 `i` 对应：

- 输入：`tokens[i : i + context_length]`
- 目标：`tokens[i + 1 : i + context_length + 1]`

使用 `np.memmap` 时，语料不必整体进入内存；每一步只切出当前 batch，再转换成 PyTorch 张量并移动到训练设备。笔记本没有 GPU 不影响实现和单元测试，正式训练则可以放到 RTX 4060 Ti 16GB 上完成。

### 3.6 Checkpoint 与恢复训练

一个可继续训练的 checkpoint 至少需要保存：

- 模型参数。
- 优化器参数与 `m`、`v`、step 等状态。
- 已完成的训练迭代数。

训练入口还维护了日志追加和 wall-clock offset，使恢复后的 JSONL 记录在 step 与累计时间上保持连续。当前作业要求的恢复功能已经满足；如果追求严格复现完全相同的随机轨迹，还需要额外保存 Python、NumPy、PyTorch 和 CUDA 的随机数生成器状态。

### 3.7 训练与验证模式

`model.train()` 和 `model.eval()` 决定模块处于训练还是评估语义；`torch.no_grad()` 决定是否构建 autograd 图。这两件事彼此独立。即使当前 Transformer 没有 dropout，也应保留清晰的模式切换，因为模型结构以后可能改变。

验证只聚合标量 loss，不需要保留计算图。频繁调用 `.item()` 会触发设备同步，因此训练日志不宜每个细小操作都同步；在指定日志间隔取一次标量即可。

### 3.8 Temperature 与 top-p sampling

Temperature 在 softmax 前缩放 logits：

$$
p_i = \operatorname{softmax}(z_i / T).
$$

- 较小的 temperature 使分布更尖锐，采样更确定。
- 较大的 temperature 使分布更平坦，随机性更强。

Top-p sampling 先按概率从大到小排序，再保留累计概率达到阈值所需的最短前缀。越过阈值的那个 token 必须保留，否则留下的概率质量会小于目标阈值。采样是在排序后的坐标中进行的，因此最终 token 位置还必须通过 `sorted_indices` 映射回原 vocabulary 坐标。

### 3.9 自回归生成

每一步生成只使用最后一个位置的 logits 来采样下一个 token，再把新 token 追加到上下文中。若序列超过模型的 context length，只把最近的窗口送入模型，但已经生成的完整 token 序列仍需保留用于最终解码。遇到 EOS 后可以提前停止。

当前实现没有 KV cache，所以每生成一个 token 都重新计算整个可见上下文。它足以验证正确性和完成 A1 实验，但生成复杂度高于带缓存的生产实现。

### 3.10 单批次过拟合的意义

单批次过拟合是训练代码的集成测试：固定一个很小的 batch，反复更新同一批数据，模型应能把 loss 压到非常低。如果做不到，数据错位、梯度链路、优化器、模型容量或训练模式中至少有一处存在问题。

它是正式训练前非常有价值的必要检查，但不是充分条件：能够记住一个 batch 不代表数据管线吞吐、泛化能力、长时间训练稳定性或生成质量没有问题。

## 4. 当前代码结构

- `cs336_basics/model.py`：数值稳定的交叉熵与 Transformer 模型组件。
- `cs336_basics/optimizer.py`：AdamW、余弦学习率调度和梯度裁剪。
- `cs336_basics/training.py`：batch 采样、checkpoint、单步训练和验证。
- `scripts/train.py`：配置读取、训练循环、JSONL 日志、验证、保存与恢复。
- `cs336_basics/generation.py`：temperature/top-p 采样和自回归生成。
- `scripts/generate.py`：加载 tokenizer、模型配置和 checkpoint 后生成文本。

单批次过拟合使用临时检查脚本完成；验证通过后已经清理，没有把一次性诊断代码留在正式脚本目录中。

## 5. 验证证据

### 5.1 官方测试

```text
47 passed, 1 xfailed
```

唯一的 xfail 是作业预期的 `Tokenizer.encode` 内存限制测试。

### 5.2 训练恢复

最小端到端实验已经验证：

- 首次运行完成 step 1、2 并写入 checkpoint。
- 恢复后继续完成 step 3、4，没有重复先前更新。
- JSONL 中训练 step 连续为 1、2、3、4。
- validation 记录出现在 step 2、4。
- wall-clock 时间在恢复前后保持单调。
- 最终 checkpoint 的 iteration 为 4。

### 5.3 生成

采样函数通过了 temperature 和 top-p 的针对性检查；自回归循环通过了假模型 smoke test；生成 CLI 能完整加载 BPE、配置与 checkpoint。同一 seed 的输出可复现。随机初始化模型输出乱码或无意义文本是正常现象，因为代码链路正确不等于模型已经学到语言分布。

### 5.4 单批次过拟合

```text
initial 3.989149332046509
50      0.0244273878633976
100     0.009255800396203995
150     0.005432933568954468
200     0.003616225440055132
250     0.002595508936792612
300     0.0019610982853919268
final   0.0019610982853919268
```

loss 平稳下降约三个数量级，没有出现 NaN、停滞或异常振荡。

## 6. 资源核算中的主要认识

矩阵乘法 `(m, n) @ (n, p)` 可以看作执行 `m × p` 个长度为 `n` 的点积。若一次乘法和一次加法分别记作一个 FLOP，总计算量约为 `2mnp` FLOPs。

在已核算的 GPT-2 XL 风格配置、context length 为 1024 时，主要矩阵乘法的一次 forward 约为 `3.52 TFLOPs`：

- Attention projections：约 `28.62%`。
- Attention mixing：约 `9.16%`。
- SwiGLU：约 `57.53%`。
- LM head：约 `4.68%`。

固定 context length 和 vocabulary size、增大模型宽度与层数时，Attention projections 和 SwiGLU 主要按 $L d^2$ 增长；Attention mixing 主要按 $L T^2 d$ 增长；LM head 主要按 $T d V$ 增长。头数本身不会改变主要 attention FLOPs，因为 `num_heads × d_head = d_model`。

在同一 GPT-2 XL 风格设置下，把 context length 从 1024 增至 16384，总 FLOPs 从约 `3.52 TFLOPs` 上升至约 `133.58 TFLOPs`，增长 `37.98` 倍；Attention mixing 的占比从 `9.16%` 上升到 `61.73%`，因为它含有随 $T^2$ 增长的项。

这些估算忽略了 SiLU、逐元素乘法、归一化等低阶操作，因为它们相对大型矩阵乘法通常不占主导，而且其精确成本取决于 FLOP 约定和硬件实现。

## 7. 下一阶段：以实验问题为主线

后续不再逐项拆解参数解析、文件搬运和简单封装，而只在会影响训练结论的地方停下来。主线是：

```text
准备可训练语料与 tokenizer artifacts
└── 在 RTX 4060 Ti 上做吞吐和显存 smoke test
    └── 建立 TinyStories baseline
        └── 学习率实验（包含至少一个不稳定或发散设置）
            └── batch size 实验
                └── 模型或训练策略消融
                    └── 生成样例与定量结果分析
                        └── OpenWebText 实验与最终报告
```

每个节点只保留三件事：实验问题、最小实现或配置变化、结果解释。纯工程胶水默认一次性处理，不再占用逐模块手敲的学习时间。
