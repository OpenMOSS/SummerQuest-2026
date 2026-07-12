# A1：Basics——从零实现语言模型

> 状态：已发布。题面版本 26.0.3，适用于 OpenMOSS 暑期集训 2026。
>
> 本页提供中文精简讲解和实验室提交规范。公式、逐题要求与实现细节请参考
> [Version 26.0.3 原 PDF](https://github.com/stanford-cs336/assignment1-basics/blob/a158843b20107949f1a8d7df1b05cd33b9166712/cs336_assignment1_basics.pdf)；
> 提交目录、必交文件、Markdown 格式、测试接口与 PR 流程以本仓库为准。两者冲突时，
> 以本仓库的实验室版要求为准。原版工作仓库固定放在本仓库的兄弟目录
> `../assignment1-basics`，并基于 Stanford CS336 `assignment1-basics` 的
> `a158843b20107949f1a8d7df1b05cd33b9166712` commit。

原 PDF 中 Stanford 课程专属的提交平台、leaderboard、外部 PR、AI 使用和评分要求不适用
于本集训；本集训的相关要求以本仓库和课程通知为准。

## 发布信息

- 上游来源为 [stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)，不复制进本仓库。
- 原版仓库、依赖锁、公共测试、数据和本地训练产物统一放在兄弟目录
  `../assignment1-basics`。
- 本作业沿用上游 21 个 adapter 函数作为稳定代码接口；真实实现必须放在
  `cs336_basics/`，不能写进 adapter，也不能修改公共测试。
- 本作业满分为 100 分。`README.md` 报告内容要求、`logs/` 日志格式与评分标准见下文
  [第 8 节](#8-实验日志格式-logs)、[第 9 节](#9-readme-报告内容要求)、[第 10 节](#10-评分标准100-分)。

开始前请阅读[公开性与提交规则](../../docs/submission-rules.md)。本仓库公开可见，内部
服务器、数据、路径、凭据和未公开实验信息不得进入 GitHub 或 Git 历史。

## 本地目录与下载

两份仓库必须保持下面的同级结构：

```text
<父目录>/
├── SummerQuest-2026/
└── assignment1-basics/
```

在 SummerQuest 仓库根目录下载并切到固定版本：

```bash
git clone https://github.com/stanford-cs336/assignment1-basics.git ../assignment1-basics
git -C ../assignment1-basics switch -c "a1/<GitHub ID>" \
  a158843b20107949f1a8d7df1b05cd33b9166712
git -C ../assignment1-basics rev-parse HEAD
```

最后一条命令必须输出上述固定 commit。不要把 `assignment1-basics/` 放进
`SummerQuest-2026/`，也不要把公共 tests、fixtures、数据、模型权重或虚拟环境复制到
SummerQuest PR。

## 提交方式

先同步最新的上游 `main`，再运行：

```bash
python3 scripts/create_assignment.py --name '<同学真名>' --assignment A1
```

脚手架会从固定相对路径 `../assignment1-basics` 读取当前工作区，只复制
`cs336_basics/`、`tests/adapters.py`、`scripts/` 和 `configs/`，并创建下面的固定目录。
PR 只提交个人 A1 目录：

```text
students/<同学真名>/assignments/A1/
├── README.md
├── submission/
│   ├── cs336_basics/
│   │   └── *.py
│   ├── tests/
│   │   └── adapters.py
│   ├── scripts/
│   │   └── *.py                 # 训练、编码、生成入口
│   └── configs/
│       └── *.{json,toml,yaml}   # 可选：公开且可复现的轻量配置
├── logs/                         # 必交：文件名与字段格式见第 8 节
└── assets/
    └── *.{png,jpg,jpeg,webp,svg} # 可选：README.md 引用的压缩图表
```

## 提交文件

- `README.md`：必交，作为公开 Markdown 报告，替代原作业的 `writeup.pdf`。
- `submission/cs336_basics/**/*.py`：必交，保存真实实现。
- `submission/tests/adapters.py`：必交，保留原作业 adapter 的函数名和签名，只负责把
  参数转交给真实实现。
- `submission/scripts/**/*.py`：必交，保存学生自己编写的训练、数据编码和生成脚本。
- `logs/`：必交，保存供作业批改使用的实验日志。
- `submission/configs/`：可选，保存轻量、公开且可复现的配置。
- `assets/`：可选，保存 `README.md` 引用的压缩图表。

> `README.md` 报告的分节要求见[第 9 节](#9-readme-报告内容要求)；`logs/` 的文件名、格式与
> 字段见[第 8 节](#8-实验日志格式-logs)；满分 100 分的评分标准见[第 10 节](#10-评分标准100-分)。

书面题、公式、表格和实验分析统一使用 Markdown；不提交 PDF、Office 文档或 notebook
导出文件。依赖由 `../assignment1-basics/uv.lock` 固定，个人提交中不添加 `pyproject.toml`、
`requirements.txt` 或 lock file。

## 文件规则

- 沿用仓库现有规则：学生目录内单个文件不得超过 5 MiB；日志格式见[第 8 节](#8-实验日志格式-logs)。
- GitHub 与飞书的公开范围继续遵循仓库统一的
  [公开性与提交规则](../../docs/submission-rules.md)。

## 代码接口

公共 ABI 是 `../assignment1-basics/tests/adapters.py` 中的 21 个函数：

| 模块 | adapter 函数 |
| --- | --- |
| Transformer | `run_linear`、`run_embedding`、`run_swiglu`、`run_scaled_dot_product_attention`、`run_multihead_self_attention`、`run_multihead_self_attention_with_rope`、`run_rope`、`run_transformer_block`、`run_transformer_lm`、`run_rmsnorm`、`run_silu` |
| 训练 | `run_get_batch`、`run_softmax`、`run_cross_entropy`、`run_gradient_clipping`、`get_adamw_cls`、`run_get_lr_cosine_schedule`、`run_save_checkpoint`、`run_load_checkpoint` |
| Tokenizer | `get_tokenizer`、`run_train_bpe` |

签名、默认参数、参数语义和返回值以该文件为准。不得改名或删减。在兄弟仓库中完成
实现并按上游方式运行测试：

```bash
cd ../assignment1-basics
uv sync --frozen
uv run pytest
cd ../SummerQuest-2026
python3 scripts/sync_a1_submission.py --name '<同学真名>'
```

每次修改实现、adapter、脚本或配置后都重新运行同步命令，再检查 SummerQuest 中的
`git diff`。同步脚本不会复制公共 tests、fixtures、数据或依赖文件。

一个 PR 只能提交一名同学的一次 A1，标题使用 `[A1] 姓名 - 简短说明`；PR 只能修改
自己的 `students/<姓名>/assignments/A1/`。完整 Git 流程见
[公开性与提交规则](../../docs/submission-rules.md)。

## 1. 作业全景

这份作业要求从零搭建一条小型语言模型流水线：

```text
Unicode 文本
-> byte-level BPE tokenizer
-> token ID 序列
-> decoder-only Transformer
-> cross-entropy loss
-> AdamW 训练
-> checkpoint / validation
-> temperature + top-p 生成文本
```

### 要实现

1. BPE tokenizer 的训练、编码和解码；
2. Transformer LM 的全部基本模块；
3. cross-entropy、AdamW、学习率调度和梯度裁剪；
4. data loader、checkpoint、training loop；
5. 自回归文本生成器。

### 要实际运行

- 在 TinyStories 和 OpenWebText 上训练 tokenizer；
- 把数据编码为 token IDs；
- 训练 TinyStories LM，生成文本并评估；
- 在 TinyStories 上做架构消融，并在 OWT 上训练；

### 代码限制

核心组件必须 from scratch。除以下项目外，不能直接使用 `torch.nn`、`torch.nn.functional`、`torch.optim` 中的现成实现：

- `torch.nn.Parameter`；
- `Module`、`ModuleList`、`Sequential` 等容器；
- `torch.optim.Optimizer` 基类。

例如不能直接使用 `nn.Linear`、`nn.Embedding` 或现成 AdamW。

`adapters.py` 只是测试接口，不能把真实逻辑写在里面；测试文件不要修改。

### 提交与总分

提交内容和目录以本页前面的“提交文件”为准，不沿用上游的 PDF + ZIP
打包方式。本作业总分固定为 100，具体评分标准由作业批改助教完善。

---

## 2. Tokenizer：从文本到整数

### 2.1 Unicode 和 UTF-8 的区别

可以分成三层理解：

```text
人看到的字符 -> Unicode 码点 -> UTF-8 bytes
“牛”          -> U+725B       -> E7 89 9B
```

- **Unicode** 给文本符号分配抽象整数编号，即 code point。
- **UTF-8** 规定怎样把码点编码成 1–4 个 byte，以便写入文件或网络传输。
- 一个 byte 是 8 bit，所以取值只能是 `0..255`。

Python 示例：

```python
text = "牛"
encoded = text.encode("utf-8")

len(text)       # 1 个字符
len(encoded)    # 3 个 byte
list(encoded)   # [231, 137, 155]
```

“迭代 bytes”就是用 `for` 或 `list()` 逐个访问其中的 byte value。每次得到的是 `0..255` 的整数，不一定是完整字符。

因为任何 UTF-8 文本最终都由这 256 种 byte value 组成，所以 byte-level tokenizer 不会遇到 OOV。但纯 byte 序列太长，模型训练会变慢，因此需要 BPE。

### 2.2 BPE 在做什么

BPE 用更大的词表换更短的 token 序列。它不断把语料中最常见的相邻 token pair 合并：

```text
t + h -> th
th + e -> the
```

如果 `the` 很常见，它最终可以从 3 个 byte token 压成 1 个 token。

### 2.3 BPE 训练流程

1. **初始化词表**：加入全部 256 个单 byte，以及 special tokens。
2. **预分词**：使用题目给定的 GPT-2 风格正则，把文本切成较粗的 pre-tokens。
3. **统计 pair**：只在每个 pre-token 内统计相邻 pair，不能跨边界。
4. **选择最高频 pair**：并列时选择字典序更大的 pair，保证结果确定。
5. **执行 merge**：生成新 token、加入 vocabulary，并按顺序记录 merge rule。
6. 重复直到达到最大 `vocab_size`。

`<|endoftext|>` 是 special token：

- 自身始终是一个完整 token；
- 是文档之间的硬边界；
- 不参加普通 pair 统计；
- 边界两侧不能发生 merge。

性能上，pre-tokenization 可以使用 multiprocessing；merge 有前后依赖，通常通过增量维护 pair counts 加速。

### 2.4 编码和解码

**编码：**

```text
字符串
-> 分离 special tokens
-> 相同正则预分词
-> UTF-8 bytes
-> 按训练得到的 merge 优先级合并
-> token IDs
```

编码阶段不能根据当前输入重新统计频率，必须使用训练好的 merge 顺序。

**解码：**

```text
token IDs
-> 查询每个 ID 对应的 bytes
-> 拼接全部 bytes
-> 整体做 UTF-8 decode
```

不能逐 token 单独 UTF-8 decode，因为一个字符可能跨多个 token。非法 UTF-8 要替换为 `U+FFFD`，而不是直接崩溃。

大文件使用 `encode_iterable` 流式处理，不能先把整个文件载入内存，也要避免任意 chunk boundary 改变 tokenization。

### 2.5 本部分任务

- `unicode1`、`unicode2`：理解码点和 Unicode 编码；
- `train_bpe`：实现 BPE 训练函数；
- TinyStories：10K 词表，加入 `<|endoftext|>`；
- OWT：32K 词表，并与 TinyStories tokenizer 比较；
- BPE 资源上限：TinyStories 为无 GPU、30 分钟、30 GB RAM；OWT 为无 GPU、12 小时、100 GB RAM；
- `tokenizer`：实现 `encode`、`decode`、`encode_iterable`；
- 比较 compression ratio（bytes/token）和 throughput；
- 把 train/dev 编码成 NumPy token ID 文件，32K 以内词表可使用 `uint16`。

---

## 3. Transformer：从 ID 到下一 token logits

### 3.1 输入输出

```text
token IDs       (B, T)
embedding       (B, T, d_model)
Transformer     (B, T, d_model)
LM head logits  (B, T, vocab_size)
```

位置 `i` 的 logits 用来预测 `x_{i+1}`。训练时一次 forward 并行预测所有位置；生成时只取最后一个位置。

模型应返回 **logits**，不要在 `TransformerLM.forward` 中提前 softmax。

### 3.2 整体结构

```text
Token Embedding
-> L 个 Pre-Norm Transformer Blocks
-> Final RMSNorm
-> Linear LM Head
-> Logits
```

每个 block：

$$
z=x+\mathrm{MHA}(\mathrm{RMSNorm}(x)),
$$

$$
y=z+\mathrm{FFN}(\mathrm{RMSNorm}(z)).
$$

residual connection 让输入可以绕过主分支直接传到后面。Pre-Norm 通常比把 norm 放在残差之后的 Post-Norm 更稳定。

### 3.3 基础模块

#### Linear 和 Embedding

- Linear 权重存成 `(d_out, d_in)`，没有 bias；
- Embedding 是 `(vocab_size, d_model)` 的可学习查表；
- 两者都不能使用现成 `nn.Linear`、`nn.Embedding`。

#### RMSNorm

RMSNorm 根据最后一维的均方根缩放激活，不减均值：

$$
\mathrm{RMSNorm}(x)
=\frac{x}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}\odot g.
$$

平方和归一化前先转成 float32，避免低精度溢出，最后再转回原 dtype。

#### SwiGLU FFN

$$
\mathrm{FFN}(x)
=W_2\left(\mathrm{SiLU}(W_1x)\odot W_3x\right).
$$

其中 `SiLU(x) = x * sigmoid(x)`，且 $\sigma(x)=\frac{1}{1+e^{-x}}$。推荐 `d_ff`
约为 `(8/3) * d_model`，并取附近的 64 倍数。

#### RoPE

RoPE 根据 token position 成对旋转每个 attention head 的 `Q` 和 `K`：

- 让 dot product 包含相对位置信息；
- 不旋转 `V`；
- 没有可学习参数；
- sin/cos 可以提前缓存并跨层复用。

### 3.4 Attention

Scaled dot-product attention：

$$
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
$$

直觉：query 与所有 key 计算相似度，softmax 变成权重，再对 values 加权求和。

softmax 前要减去该维最大值，避免 `exp` 溢出。

#### Mask

本作业规定：

- mask 为 `True`：允许注意；
- mask 为 `False`：禁止注意，在 softmax 前设为负无穷。

Causal mask 是包含主对角线的下三角矩阵：位置 `i` 只能看到 `j <= i`，防止训练时偷看未来答案。

#### Multi-head

把 `d_model` 分成 `h` 个 head：

```text
(B, T, d_model)
-> Q/K/V
-> (B, h, T, d_k)
-> 每个 head 独立 attention
-> 拼回 (B, T, d_model)
```

其中 `d_k = d_v = d_model / h`。

### 3.5 参数量和算力

矩阵乘法 `(m,n) @ (n,p)` 需要约 `2mnp` FLOPs。

要理解两种增长：

- Linear/FFN 的计算通常随序列长度 `T` 线性增长；
- attention matrix 的计算和内存随 `T^2` 增长。

因此 context 从 1K 增长到 16K 时，attention 会迅速成为主要成本。

### 3.6 本部分任务

- 实现 `Linear`、`Embedding`、`RMSNorm`、SwiGLU、RoPE；
- 实现稳定 softmax、masked attention 和 causal MHA；
- 组装 Transformer block 和完整 LM；
- 对 GPT-2 small/medium/large/XL 形状计算参数、内存和 forward FLOPs。

建议严格按以上依赖顺序完成，每个模块立即运行对应 `pytest -k ...`。

---

## 4. 训练：让模型学会预测下一 token

### 4.1 Cross-entropy 和 perplexity

对正确 token `y`，单位置 loss 为：

$$
\ell=-\log\mathrm{softmax}(logits)[y].
$$

Perplexity 是平均 cross-entropy 的指数：

$$
\mathrm{PPL}=\exp(\mathrm{mean\ loss}).
$$

越低通常越好，但不同 tokenizer 或数据集的 per-token loss 不宜直接比较。

### 4.2 AdamW

AdamW 为每个参数保存：

- 一阶矩 `m`：梯度移动平均；
- 二阶矩 `v`：梯度平方移动平均；
- step `t`，用于 bias correction。

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

因为初始值为 0，需要进行 bias correction：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

最终参数更新：

$$
\theta_t=(1-\eta\lambda)\theta_{t-1}-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

它会把参数按 weight decay 向 0 收缩。关键点是 **decoupled weight decay**：decay 单独应用，而不是简单把 `lambda * parameter` 加进梯度，也就是等号右侧的第一项。

### 4.3 稳定训练的两个辅助机制

- **Warmup + cosine schedule**：训练初期从小 LR 升到最大值，之后按余弦下降到最小值。
- **Global gradient clipping**：所有参数共同计算一个 L2 norm；超过阈值时统一缩小，而不是逐层分别裁剪。

### 4.4 Data loader

整个 tokenized corpus 看成一条长序列。随机选择起点 `s`：

```text
input  = x[s : s+T]
target = x[s+1 : s+T+1]
```

输入和目标只相差一位。大数据使用 `np.memmap` 或 `np.load(..., mmap_mode="r")`，避免加载整个文件。

### 4.5 Checkpoint 和训练循环

Checkpoint 至少保存：

1. model state；
2. optimizer state；
3. iteration。

只存模型权重不能无缝恢复 AdamW 和学习率调度。

完整循环：

```text
get batch
-> zero gradients
-> model forward
-> cross-entropy
-> backward
-> gradient clipping
-> optimizer step
-> logging / validation / checkpoint
```

验证时关闭梯度，并记录 step、wall-clock time、training loss 和 validation loss。

### 4.6 本部分任务

- 实现 cross-entropy、AdamW、cosine schedule、gradient clipping；
- 实现随机 batch、checkpoint save/load；
- 写可配置、支持 mmap、验证、日志和恢复训练的脚本；
- 完成 AdamW 显存、FLOPs 和训练时间核算。

---

## 5. 文本生成与实验

### 5.1 自回归生成

给定 prompt：

1. 模型 forward；
2. 取最后位置 logits；
3. 转成采样分布；
4. 采一个 token 并追加；
5. 遇到 `<|endoftext|>` 或达到最大长度时停止。

**Temperature：** 越低越确定，越高越随机。

**Top-p：** 从高概率 token 开始，保留累计概率至少达到 `p` 的最小集合，重新归一化后采样。它与固定保留 `k` 个 token 的 top-k 不同。

### 5.2 TinyStories baseline

| 参数 | 值 |
|---|---:|
| vocab size | 10,000 |
| context length | 256 |
| `d_model` | 512 |
| `d_ff` | 1,344 |
| layers / heads | 4 / 16 |
| RoPE theta | 10,000 |

目标：以`batch_size=128,training_steps=10000`为例，`total_tokens=327680000`，把 per-token validation loss 调到不高于 **1.45**。

低资源 CPU/MPS 方案可降到约 40M tokens，并把目标放宽到 2.00。

生成实验要提交至少 256 tokens，除非先遇到 `<|endoftext|>`，并评价流畅度及至少两个影响因素。

### 5.3 必做实验

1. 记录完整 experiment log 和 loss curves；
2. 扫 learning rate，必须包含至少一个发散 run；
3. batch size 从 1 试到显存上限，包括 64 和 128；
4. 四个架构消融：
   - 删除 RMSNorm；
   - Pre-Norm 改成 Post-Norm；
   - RoPE 改成 NoPE；
   - SwiGLU 对比参数量近似匹配的 SiLU FFN；
5. 在 OWT 上用相同模型架构和训练 iterations 训练并生成文本。

实验比较要尽量只改一个变量，并记录 processed tokens、step 和 wall-clock time。

---

## 6. 推荐实现顺序

```text
1. BPE training
2. Tokenizer encode/decode
3. Linear / Embedding / RMSNorm / SwiGLU / RoPE
4. Softmax / attention / causal MHA
5. Transformer block / full LM
6. Cross-entropy / AdamW / schedule / clipping
7. Data loader / checkpoint / training loop
8. Decoder
9. TinyStories -> ablations -> OWT
```

每完成一个模块就运行对应测试，不要等整个系统完成后一起排错。

## 7. 容易错的地方

1. 把 Unicode 码点和 UTF-8 byte 当成同一件事。
2. BPE 跨 pre-token 或 `<|endoftext|>` merge。
3. 编码时重新按频率 merge，而不是使用训练 merge 顺序。
4. Linear 权重方向、head/sequence 维度放反。
5. attention mask 中误把 `True` 当成屏蔽。
6. 对 `V` 也应用 RoPE，或忘记 causal mask。
7. RMSNorm 平方前未转 float32。
8. 模型内部提前 softmax。
9. input 和 target 没有右移一位。
10. AdamW 没保存 optimizer state，checkpoint 无法正确续训。
11. gradient clipping 对每层单独做，而不是全局 norm。
12. 实验预算和 tokenizer 不同，却直接比较 loss 数值。

---

## 8. 实验日志格式 logs

本节规定 `logs/` 的固定目录、文件名和字段。**批改会用脚本直接解析这些文件**，格式不符
的日志无法得分，请严格按此产出。所有日志为纯文本，单文件不超过 5 MiB。

### 8.1 目录与文件

```text
logs/
├── train_tinystories.jsonl        # 必交：TinyStories 主训练 run
├── lr_sweep/                       # 必交：学习率扫，>=3 个 run，至少 1 个发散
│   ├── lr_1e-3.jsonl
│   ├── lr_3e-3.jsonl
│   └── lr_1e-2.jsonl
├── batch_size/                     # 必交：>=3 个 run，必须含 batch size 64 和 128
│   ├── bs_64.jsonl
│   ├── bs_128.jsonl
│   └── bs_256.jsonl
├── ablation_no_rmsnorm.jsonl       # 必交：删除 RMSNorm
├── ablation_post_norm.jsonl        # 必交：Pre-Norm 改 Post-Norm
├── ablation_nope.jsonl             # 必交：RoPE 改 NoPE
├── ablation_silu.jsonl             # 必交：SwiGLU 对比 SiLU
├── train_owt.jsonl                 # 必交：OpenWebText 训练 run
└── summary.json                    # 必交：汇总指标，见 8.3
```

低资源方案（CPU/MPS）仍需交齐上述文件，但可以缩小 token 数，并在 `summary.json` 里把
`low_resource` 设为 `true`（评分阈值会相应放宽，见第 10 节）。

### 8.2 训练日志行格式（所有 `*.jsonl`）

每行一个 JSON 对象，一行代表一次记录点。字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `step` | int | 是 | 梯度步，从小到大 |
| `wall_clock_sec` | float | 是 | 自训练开始的墙钟秒数，单调不减 |
| `train_loss` | float | 是 | 该步的 per-token 训练 loss |
| `lr` | float | 是 | 该步学习率 |
| `val_loss` | float\|null | 否 | 验证时填数值，否则 `null`；每个 run 至少 2 行非 null |
| `grad_norm` | float | 否 | 裁剪前的全局梯度 L2 norm |
| `tokens` | int | 否 | 累计已处理 token 数 |

示例（`train_tinystories.jsonl` 节选）：

```json
{"step": 0,    "wall_clock_sec": 0.0,   "train_loss": 9.21, "val_loss": null, "lr": 0.0,     "grad_norm": 3.1}
{"step": 200,  "wall_clock_sec": 24.7,  "train_loss": 4.05, "val_loss": null, "lr": 0.0006,  "grad_norm": 1.4}
{"step": 1000, "wall_clock_sec": 121.3, "train_loss": 2.11, "val_loss": 2.03, "lr": 0.0005,  "grad_norm": 0.9}
{"step": 10000,"wall_clock_sec": 1487.0,"train_loss": 1.33, "val_loss": 1.42, "lr": 1e-05,   "grad_norm": 0.6}
```

发散 run（LR 扫里至少要有一个）：`train_loss` 明显上升或出现 `NaN`/`Inf`，如实记录即可，
不要删掉发散的行。

### 8.3 `summary.json` 字段

一个 JSON 对象，供批改直接读取关键指标。`null` 表示未做/未测：

```json
{
  "student": "张三",
  "low_resource": false,
  "config": {
    "vocab_size": 10000, "context_length": 256, "d_model": 512, "d_ff": 1344,
    "num_layers": 4, "num_heads": 16, "rope_theta": 10000,
    "batch_size": 128, "total_steps": 10000
  },
  "tinystories": {
    "final_val_loss": 1.42, "total_wall_clock_sec": 1487.0, "total_tokens": 327680000
  },
  "tokenizer": {
    "ts_longest_token_bytes": 12,
    "owt_longest_token_bytes": 20,
    "ts_compression_ratio": 4.10,
    "owt_compression_ratio": 4.35,
    "throughput_bytes_per_sec": 2500000,
    "cross_tokenizer_comment": "用 TinyStories tokenizer 编码 OWT 时压缩率下降到约 ..."
  },
  "ablations": {
    "no_rmsnorm": {"final_val_loss": 5.20},
    "post_norm":  {"final_val_loss": 1.90},
    "nope":       {"final_val_loss": 1.71},
    "silu":       {"final_val_loss": 1.55}
  },
  "owt": {"final_val_loss": 3.80},
  "generation": {
    "tinystories_sample_tokens": 300,
    "sample_text": "Once upon a time ...",
    "fluency_comment": "整体通顺，但偶尔重复；影响因素：temperature 与训练步数。"
  }
}
```

### 8.4 建议：用统一 logger 产出日志

为保证格式一致，建议在自己的训练脚本里用下面这个极简 logger（复制到
`submission/scripts/` 即可，无需额外依赖）：

```python
# a1_logging.py —— 产出符合第 8 节规范的日志
import json, time, pathlib

class RunLogger:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")
        self.t0 = time.time()

    def log(self, step, train_loss, lr, val_loss=None, grad_norm=None, tokens=None):
        row = {"step": int(step), "wall_clock_sec": round(time.time() - self.t0, 3),
               "train_loss": float(train_loss), "lr": float(lr), "val_loss": val_loss}
        if grad_norm is not None: row["grad_norm"] = float(grad_norm)
        if tokens is not None: row["tokens"] = int(tokens)
        self.f.write(json.dumps(row) + "\n"); self.f.flush()

    def close(self):
        self.f.close()
```

---

## 9. README 报告内容要求

`students/<姓名>/assignments/A1/README.md` 是公开 Markdown 报告，替代原作业的
`writeup.pdf`。除模板已有的“基本信息/复现说明/代码与脚本/实验日志/飞书补充文档”外，
报告正文必须包含下列小节（标题请照抄，便于批改定位）：

1. **书面题**：`unicode1`、`unicode2` 的简答，以及 AdamW 显存/最大 batch size/单步
   FLOPs/训练时间核算（`adamw_accounting`）。
2. **Tokenizer 实验**：TinyStories 与 OWT tokenizer 的最长 token、compression ratio、
   throughput，以及交叉编码对比。
3. **TinyStories 训练**：贴出 loss 曲线（横轴分别按 step 和 wall-clock 各一张，放
   `assets/` 引用），报告最终 val loss。
4. **学习率扫**、**batch size 实验**：曲线 + 结论，学习率扫需说明哪个 run 发散。
5. 四个消融，各用**下列固定标题**（批改按标题匹配）：
   - `### 消融：删除 RMSNorm`
   - `### 消融：Post-Norm`
   - `### 消融：NoPE`
   - `### 消融：SiLU`
6. **OWT 实验**：曲线 + 与 TinyStories 的对比分析。
7. **文本生成**：至少 256 token 的生成样本 + 流畅度评价 + 至少两个影响因素。

书面题的数值答案除写进正文外，**还要在报告中放一个 ```json 代码块**，字段如下，供批改
自动核对（无需额外文件）：

````markdown
```json answers
{
  "unicode1_chr0": "空字符 NUL (U+0000)",
  "unicode2_utf8_reason": "UTF-8 变长、兼容 ASCII、词表仅 256，...",
  "adamw_peak_memory_gpt2xl_bytes": 1.2e11,
  "adamw_max_batch_size_80gb": 3,
  "adamw_step_flops": "约 2 * 参数量 的更新 FLOPs + ...",
  "gpt2xl_train_hours_h100": 120
}
```
````

---

## 10. 评分标准100 分

总分固定 100。**代码分与指标分由脚本自动判定**；带“人工确认”的分析类小项由批改助教
在自动初评基础上快速复核。

| 板块 | 分值 | 判定方式 |
|---|---:|---|
| **代码正确性**（21 个 adapter，`uv run pytest`） | 40 | 自动：按 JUnit 结果分组计分。BPE 训练 8、Tokenizer 8、Transformer 14、训练工具 10；组内按通过测试比例给分。两个 memory-usage 测试不计分。 |
| **书面题**（`unicode`、`adamw_accounting`） | 6 | 自动：核对报告 `answers` 块的存在与数值（含容差）；推导过程人工确认 |
| **Tokenizer 实验** | 7 | 自动：核对 `summary.json.tokenizer` 各字段存在且在合理范围；交叉对比结论人工确认 |
| **TinyStories 训练** | 18 | 自动：日志完整（含 step 与 wall-clock）+ 最终 val loss 达标。达标阈值 ≤1.45 满分，≤1.60 部分分，≤2.00 少量分；低资源模式阈值放宽到 ≤2.00 满分 |
| **学习率扫**（含 ≥1 发散 run） | 5 | 自动：run 数 ≥3 且存在发散 run |
| **batch size 实验** | 3 | 自动：run 数 ≥3 且覆盖 64、128 |
| **四个消融** | 12 | 每个 3 分：日志有效 + 记录最终指标（自动 2 分）+ README 分析小节（人工确认 1 分） |
| **OWT 实验 + 文本生成** | 5 | 自动：OWT 日志有效 + 生成样本 ≥256 token；流畅度评价人工确认 |
| **日志完整性/一致性** | 4 | 自动：step 与 wall-clock 单调、loss 明显下降且非恒定（异常仅触发人工抽查，不等于判定作弊） |

说明：

- 达不到 val loss 阈值不代表零分，日志与实验完整仍按上表给分；只有相应达标项拿不到。
- 各板块的具体阈值、字段范围与 tokenizer 数值答案的参考值，最终以批改助教发布的
  评分脚本配置为准；本表用于让同学了解得分点与目标。
- 不同 tokenizer / 训练预算下的 per-token loss 不能横向比较，报告分析时请注意。
