# A1 学习记录 02：模型组件

日期：2026-07-15

状态：模型组件代码里程碑完成

## 1. 里程碑结论

本阶段从最小的线性层与 embedding 开始，逐层完成了 RMSNorm、SiLU/SwiGLU、RoPE、数值稳定 softmax、scaled dot-product attention、带因果遮罩和 RoPE 的 multi-head self-attention、pre-norm Transformer block，以及完整 Transformer LM 前向计算。

当前已经满足的核心合同：

- 所有可学习权重都通过 `nn.Parameter` 或注册子模块管理，参数形状和 state-dict 层级与作业参考权重一致。
- 所有位置级算子保留任意前导 batch-like 维度，并只对最后的特征维执行对应运算。
- RMSNorm 在平方前把输入提升到 float32，归一化后恢复输入 dtype。
- RoPE 对每个 head 的 Q/K 相邻维度对执行多频率旋转，不旋转 V，并通过非持久 buffer 缓存 sin/cos。
- Softmax、mask 和 attention 的数值稳定性与广播规则正确。
- MHA 使用三个整体投影并把 head 当作 batch-like 维度，没有逐 head Python 循环。
- Transformer block 使用 pre-norm、两条 residual connection 和因果注意力。
- Transformer LM 支持不超过 `context_length` 的任意实际序列长度，并返回未归一化 logits。
- 模型组件相关公开测试、Ruff、ty 和额外行为不变量全部通过。

这部分明显快于 tokenization。原因不是内容更少，而是已有较完整的 Transformer 概念基础；主要学习增量集中在数学公式如何落实为可靠的 PyTorch 张量合同，而不是像 BPE 那样同时设计全局状态、增量索引、性能优化和序列化协议。

## 2. 学习路径

```text
Linear
└── Embedding
    └── RMSNorm
        └── SiLU
            └── SwiGLU
                └── RoPE
                    └── Stable Softmax
                        └── Scaled Dot-Product Attention
                            └── Causal Multi-Head Self-Attention
                                └── MHA + RoPE
                                    └── Pre-Norm Transformer Block
                                        └── Transformer LM
```

与 tokenization 阶段相同，每个节点采用“数学合同与形状 → 最小实现 → 定向测试 → 概念纠偏 → 静态检查”的节奏。不同之处在于，本阶段大部分模块只依赖前一节点，局部测试反馈很快，因此主干几乎没有因额外代码实验产生分叉。

## 3. 核心理论与张量合同

### 3.1 Linear：权重方向与批量维

Linear 保存的权重形状是 `(d_out, d_in)`，对应列向量记号中的线性映射：

\[
y = Wx, \qquad W \in \mathbb{R}^{d_{out}\times d_{in}}.
\]

在 PyTorch 中，输入把特征放在最后一维，形状为 `(..., d_in)`，因此前向可以写成：

```text
x:       (..., d_in)
weight:  (d_out, d_in)
y:       (..., d_out)
```

存成 `(d_in, d_out)` 在纯数学上也能工作，但那实际保存的是 \(W^\top\)，会偏离作业、PyTorch 和参考 state dict 的共同约定。当前实现使用 `torch.einsum("...i,oi->...o", x, weight)`，清楚标出了输入和输出轴。

初始化使用均值 0、方差 \(2/(d_{in}+d_{out})\) 的截断正态分布，截断范围是 `[-3σ, 3σ]`。这种初始化公式对转置是对称的，所以权重存储方向不是由初始化决定，而是由线性映射和接口约定决定。

### 3.2 Embedding：高级整数索引与梯度累加

Embedding 权重是一个查找表：

```text
weight:     (vocab_size, d_model)
token_ids:  (...)
output:     (..., d_model)
```

`torch.Tensor.__getitem__` 已实现多维高级整数索引，因此 `weight[token_ids]` 会把 `token_ids` 中每个整数替换为对应的整行向量，并保留 token ID 张量的全部维度。

若同一个 token ID 在输入中出现多次，autograd 会把这些位置的上游梯度累加到同一 embedding 行：

\[
\frac{\partial L}{\partial W[j]}
=
\sum_{p:\,token\_ids[p]=j}
\frac{\partial L}{\partial output[p]}.
\]

在 `loss = output.sum()` 的 smoke test 中，一个出现两次的 token 对应权重行得到全 2 梯度；未被索引的行梯度为 0。

### 3.3 RMSNorm：只缩放，不中心化

RMSNorm 对每个 token 的最后一维独立计算：

\[
\operatorname{RMS}(x)=\sqrt{\operatorname{mean}(x^2)+\epsilon},
\]

\[
y=\frac{x}{\operatorname{RMS}(x)}\odot g.
\]

其中 `g` 是形状 `(d_model,)` 的可学习 gain，初始化为全 1。与 LayerNorm 相比，RMSNorm 不减均值，也不包含 bias。

`mean(..., dim=-1, keepdim=True)` 将形状 `(..., d_model)` 变为 `(..., 1)`，使除法只沿最后一维广播。不保留该维度时，PyTorch 会从右侧对齐剩余轴；一般会失败，也可能在尺寸偶合时错误地沿另一轴广播。

低精度输入必须在平方之前转换为 float32。`Tensor.to()` 不是原地操作，所以必须接住返回值。若先在 float16 中执行平方，溢出已经发生，事后再转换无法恢复信息。最终结果再转回原输入 dtype。

### 3.4 SiLU 与 SwiGLU：非线性门控

SiLU 定义为：

\[
\operatorname{SiLU}(x)=x\sigma(x).
\]

它在大正数区域近似恒等映射，在大负数区域从负侧趋近 0，并在 0 附近平滑过渡。

SwiGLU 的前向为：

\[
\operatorname{FFN}(x)=W_2\left(\operatorname{SiLU}(W_1x)\odot W_3x\right).
\]

形状链是：

```text
x:                 (..., d_model)
W1x, W3x:          (..., d_ff)
gate * value:      (..., d_ff)
W2(...):           (..., d_model)
```

`W1` 和 `W3` 必须是独立投影。若共享同一个结果 `z`，隐藏表示会退化为 `SiLU(z) * z = z² sigmoid(z)`，让 gate 与 value 无法独立学习，并显著限制单层表达形式。共享计算会减少而不是增加计算量；真正的问题是表达约束，而非计算冗余。

### 3.5 RoPE：用绝对旋转实现相对位置交互

RoPE 把相邻维度 `(0, 1), (2, 3), ...` 看作二维平面。第 `k` 个维度对的角频率是：

\[
\omega_k=\theta^{-2k/d_k}=\frac{1}{\theta^{2k/d_k}},
\]

位置 `i` 的旋转角是：

\[
\phi_{i,k}=i\omega_k.
\]

代码中的 `theta ** (dimension_indices / d_k)` 是频率分母或尺度，真正的频率是它的倒数。指数在 `[0, 1)` 内等间隔，因此频率在对数尺度上从快到慢分布，用有限的维度覆盖局部到长程的多种位置尺度。

以 `d_k = 8, theta = 10000` 为例：

| channel 对 | 每 token 角度 | 近似完整周期 |
|---|---:|---:|
| `(0, 1)` | 1 rad | 6.28 tokens |
| `(2, 3)` | 0.1 rad | 62.8 tokens |
| `(4, 5)` | 0.01 rad | 628 tokens |
| `(6, 7)` | 0.001 rad | 6283 tokens |

这可以理解为一组转速不同的时钟：快时钟分辨局部距离，慢时钟保留长距离差异。单个频率会周期性绕回，多频率联合能显著减少短范围内的碰撞。

RoPE 使用绝对位置 `i` 和 `j` 分别旋转 query 与 key，但注意力内积只依赖相对距离：

\[
q_i'=R_iq_i, \qquad k_j'=R_jk_j,
\]

\[
(q_i')^\top k_j'=q_i^\top R_i^\top R_jk_j=q_i^\top R_{j-i}k_j.
\]

因此更准确的说法是“使用绝对下标执行旋转，让注意力交互自然依赖相对位置”，而不是因为角度能绕回就同时获得绝对与相对编码。

sin/cos 表不需要训练，使用 `register_buffer(..., persistent=False)`：buffer 会随模块移动设备，但不会成为参数，也不会进入 state dict。由于静态分析器无法从动态注册中可靠推断属性类型，类中显式声明了对应 buffer 为 `torch.Tensor`。

### 3.6 Stable Softmax 与 Scaled Dot-Product Attention

Softmax 对加上任意公共常数保持不变，因此可以先减去指定维度的最大值：

\[
\operatorname{softmax}(x)_i
=
\frac{e^{x_i-m}}{\sum_j e^{x_j-m}},
\qquad m=\max_jx_j.
\]

平移后最大指数输入为 0，所以所有指数值都不超过 1，避免上溢。极小值下溢成 0 通常只丢弃本来就可忽略的概率质量；上溢则容易产生 `inf / inf = NaN`，破坏整行分布。

Scaled dot-product attention 为：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
\]

其形状合同是：

```text
Q:        (..., queries, d_k)
K:        (..., keys, d_k)
V:        (..., keys, d_v)
scores:   (..., queries, keys)
weights:  (..., queries, keys)
output:   (..., queries, d_v)
```

若 Q/K 各分量近似独立、均值 0、方差 1，则点积是 `d_k` 项之和，方差约为 `d_k`，标准差约为 `sqrt(d_k)`。除以 `sqrt(d_k)` 恰好把 score 的尺度恢复到常数量级；除以 `d_k` 会缩放过度，使维度越大时 softmax 越接近均匀分布。

mask 中 `True` 表示允许注意，`False` 表示禁止。必须在 softmax 之前把禁止位置填成 `-inf`。如果先 softmax 再置零，禁止位置虽然不会直接贡献 V，却已经占据分母中的概率质量，允许位置的权重和会小于 1；除非再归一化，否则结果不同。

### 3.7 Multi-Head Self-Attention：并行的多套路由

Q/K/V 分别用一个 `(d_model, d_model)` 权重完成整体投影，再把最后一维拆成：

```text
(..., sequence, d_model)
→ (..., sequence, num_heads, d_head)
→ (..., num_heads, sequence, d_head)
```

其中 `d_head = d_model // num_heads`。`unflatten` 负责拆分最后一维，`transpose(-3, -2)` 把 head 移到 sequence 前，使它成为 SDPA 支持的 batch-like 维度。所有 heads 因而可以并行计算，不需要逐 head 循环。

因果 mask 是下三角布尔矩阵：query `i` 只能关注 key `j <= i`。改变未来 token 的输入不会影响更早位置的输出，这一性质通过额外 smoke test 验证。

多头注意力不是简单地“强迫每个头学习不同内容”。一层单头注意力只有一张 softmax 权重矩阵，所有 value channels 共用同一种路由；多头注意力允许同一个 query 同时使用多张独立权重矩阵。架构提供了形成分工的归纳偏置，但没有多样性约束，头之间仍可能冗余或塌缩。

RoPE 在拆分 heads 后应用，旋转维度是 `d_head` 而不是 `d_model`。token positions 在 head 轴处增加一个单例维度，从 `(..., sequence)` 变成 `(..., 1, sequence)`，同一位置表即可广播到所有 heads。

RoPE 只作用于 Q/K，因为它们决定注意力路由。V 承载被取回的内容；若不同位置的 V 按各自绝对相位旋转，它们会处在不同坐标系，却被直接加权相加，使输出同时混合内容与位置相位。

### 3.8 Pre-Norm Transformer Block 与 Transformer LM

Pre-norm block 的两步更新是：

\[
y=x+\operatorname{MHA}(\operatorname{RMSNorm}_1(x)),
\]

\[
z=y+\operatorname{FFN}(\operatorname{RMSNorm}_2(y)).
\]

归一化只作用于送进子层的分支，残差流本身保持不被 norm 截断。若 `y = x + f(x)`，则：

\[
\frac{\partial y}{\partial x}=I+\frac{\partial f}{\partial x}.
\]

即使子层分支的梯度很小或不稳定，恒等项仍保留直接通路。这也是 pre-norm 深层模型训练更稳定的直观原因之一。

完整 Transformer LM 的形状链是：

```text
token IDs:          (batch, sequence)
token embeddings:  (batch, sequence, d_model)
blocks:             (batch, sequence, d_model)
final RMSNorm:      (batch, sequence, d_model)
LM head logits:     (batch, sequence, vocab_size)
```

每个 block 内的 RMSNorm 只规范化子层输入，block 输出的 residual stream 仍持续累加，因此在 LM head 前还需要 final RMSNorm 控制最终尺度。

模型返回 logits 而不是 probabilities。Cross-entropy 可以直接基于 logits 用稳定的 log-sum-exp 计算，避免先形成概率、再取对数造成额外计算和数值损失。生成阶段也可能根据 temperature、top-k 或 top-p 在外部修改 logits 后再归一化。

`nn.ModuleList` 用于注册独立 Transformer blocks。普通 Python list 不会完整参与模块注册、设备迁移和 state dict；`[block] * num_layers` 还会让所有层共享同一个对象。实际 token positions 根据输入当前的 `sequence_length` 构造，而不是固定使用最大 `context_length`，所以截短输入能够正确前向。

## 4. 实现结构

### `cs336_basics/model.py`

- `Linear`：无 bias 的线性变换和截断正态初始化。
- `Embedding`：整数 token ID 到 embedding 行的高级索引。
- `RMSNorm`：float32 内部归一化、可学习 gain 和 dtype 恢复。
- `silu`：逐元素 SiLU。
- `SwiGLU`：三个 Linear 组成的门控前馈网络。
- `RoPE`：多频率 sin/cos buffer、按 token position 索引和相邻维度旋转。
- `softmax`：指定轴上的数值稳定 softmax。
- `scaled_dot_product_attention`：缩放、mask、softmax 和 value 汇总。
- `MHA`：整体 Q/K/V 投影、head 拆分、因果 mask、可选 RoPE 和输出投影。
- `TransformerBlock`：两组 pre-norm 子层和 residual connection。
- `TransformerLM`：token embedding、`ModuleList` blocks、final norm 和 LM head。

### `tests/adapters.py`

- adapter 只负责构造模块、按参考权重的 device/dtype 分配参数、加载 state dict，并调用学生实现。
- 参数命名与参考权重保持一致，例如 `attn.output_proj.weight`、`layers.0.ffn.w1.weight` 和 `ln_final.weight`。
- `load_state_dict` 会复制数值，但不会把已经创建的模块自动迁移到来源权重的 device/dtype，因此构造时必须显式传递二者。

## 5. 验证证据

### 公开模型与 softmax 测试

```text
tests/test_model.py::test_linear PASSED
tests/test_model.py::test_embedding PASSED
tests/test_model.py::test_swiglu PASSED
tests/test_model.py::test_scaled_dot_product_attention PASSED
tests/test_model.py::test_4d_scaled_dot_product_attention PASSED
tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED
tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED
tests/test_model.py::test_transformer_block PASSED
tests/test_model.py::test_rmsnorm PASSED
tests/test_model.py::test_rope PASSED
tests/test_model.py::test_silu_matches_pytorch PASSED
tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED

14 passed in 0.18s
```

### 额外不变量

- Embedding 对重复 token ID 的梯度正确累加，未选中行梯度为 0。
- RMSNorm 对 float16 `[300, 400]` 输入保持有限输出，证明在平方前提升到 float32，避免了低精度平方溢出。
- Softmax 在非末维和约 1000 量级 logits 上与 PyTorch 参考实现一致，归一化结果有限且指定轴求和为 1。
- SDPA 的单一允许 key mask 返回该 key 对应的 value；无 mask 路径保持形状和有限数值。
- MHA 省略 token positions 与显式传入 `arange(sequence_length)` 得到相同结果。
- 改变未来 token 不会改变更早位置的 MHA 或 Transformer block 输出。
- Transformer block 的 state dict 恰好包含 9 个可学习权重层级；RoPE 的非持久 buffers 不进入其中。

### 静态检查

```text
ruff: passed
ty: passed
```

检查范围为 `cs336_basics/model.py` 和 `tests/adapters.py`。

## 6. 典型错误与调试经验

### PyTorch 语义

- `Tensor.to(dtype)` 不是原地操作；忽略返回值等于没有转换。
- `weight[token_ids]` 已支持任意维整数索引，不需要手工循环拼装 embedding 输出。
- `keepdim=True` 不只是为了“形状好看”，而是为了让广播严格发生在被归约的轴上。
- `transpose` 后的维度含义必须靠形状注释跟踪；MHA 中 head 和 sequence 的交换决定了 SDPA 在哪个轴上工作。
- `register_buffer` 动态创建属性，静态分析器可能需要类级 `torch.Tensor` 类型声明。
- `nn.ModuleList` 才能可靠注册层列表；普通 list 和对象重复列表都不满足独立层合同。

### 数学公式落地

- RMSNorm 的分母是均方根，不是均方；遗漏平方根会让几乎所有 snapshot 元素偏离。
- RMSNorm 必须先提升精度再平方，不能在溢出后补救。
- RoPE 中 `theta ** exponent` 是频率分母，真正的角频率是其倒数；学习注释里曾出现过负号重复，代码虽正确但文字公式错误。
- RoPE 的相对性来自 `R_i^T R_j = R_{j-i}`，不是来自单个角度的周期绕回。
- SDPA 除以 `sqrt(d_k)` 是匹配点积标准差，而不是任意选择一个随维度增长的缩放量。
- mask 在 softmax 后置零会丢失归一化；必须在 softmax 前将禁止 score 设为 `-inf`。

### 接口与参数组织

- MHA 的输出投影最终统一命名为 `output_proj`，避免 isolated adapter 通过但完整 Transformer state dict 无法直接加载。
- RoPE 的维度是每个 head 的 `d_head`，不是总 `d_model`。
- token positions 在 MHA 内插入 head 单例轴，才能同时支持一维、按 batch 给定的位置和多头广播。
- adapter 构造模块时必须沿用参考权重的 device/dtype；仅调用 `load_state_dict` 不足以改变模块的分配位置和精度。
- Transformer LM 的实际位置长度来自当前输入，而不是最大 context length；否则截短输入会得到错误形状或错误位置表。

## 7. 学习反思

本阶段的主观难度低于 tokenization，主要原因有四点：

1. 对 Transformer 的模块结构已有先验，知道每个组件在整体网络中的职责。
2. 依赖关系近似线性，每个组件都有局部 snapshot 或 PyTorch 参考实现，错误能快速定位。
3. 没有 BPE 那种需要长期保持一致的多份全局可变状态，也没有必须 profile 后才能发现的算法级瓶颈。
4. 大多数错误属于“数学概念正确，但张量轴、dtype、广播或注册语义不精确”，修复范围较小。

不过，这次仍补齐了若干过去容易交给 agent 而没有真正内化的实现细节：

- PyTorch 高级整数索引如何自然扩展 token ID 形状。
- 归约轴与 `keepdim` 如何决定广播语义。
- 低精度计算中“在危险操作之前提升精度”的原则。
- RoPE 的频率尺度、旋转群性质和相对位置推导。
- multi-head attention 相比同宽单头的一层为何确实具有多张独立路由矩阵。
- PyTorch 子模块、buffer、ModuleList 和 state dict 的注册边界。
- 参考测试通过之后仍要检查 device/dtype、因果性和参数层级等行为合同。

本阶段全部验证都在笔记本 CPU 上完成。模型组件的正确性测试不依赖 GPU；GPU 会在后续训练吞吐、显存与规模实验中变得重要。

## 8. 已知边界与暂缓事项

模型前向组件已经完成，但这不代表 A1 已全部完成：

- 尚未完成 Transformer FLOPs 和内存 resource accounting。
- 尚未实现 cross-entropy、AdamW、cosine learning-rate schedule 和 gradient clipping。
- 尚未实现随机 batch sampling、checkpointing 和完整 training loop。
- 尚未把 tokenizer 输出序列化为训练所需的数据数组，也尚未进行实际 LM 训练。
- 尚未完成生成、训练曲线、超参数比较和后续 ablation 报告。
- 当前实现以作业要求的清晰正确为目标，没有实现 fused QKV、FlashAttention、KV cache 或 GPU kernel 优化；这些不是本里程碑的完成条件。

## 9. 下一里程碑：训练基础设施

按讲义与代码依赖，下一条学习主干暂定为：

```text
Transformer FLOPs / memory accounting
└── Cross-Entropy
    └── AdamW
        └── Cosine LR Schedule
            └── Gradient Clipping
                └── Data Loading
                    └── Checkpointing
                        └── Training Loop
```

其中 resource accounting 是理论节点，其余节点继续使用“数学合同 → 最小实现 → 定向测试 → 不变量检查”的节奏。训练 loop 只有在损失、优化器、调度器、数据和 checkpoint 各自通过验证之后再组装。
