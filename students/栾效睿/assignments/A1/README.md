# A1 公开提交：栾效睿

> 本文件和同目录代码公开可见。只提交允许公开且已经脱敏的内容；组织内材料放在下方
> 登记的飞书补充文档中，密钥和访问凭据不进入任何提交材料。

> 评分标准与评测方式见 [`assignments/A1/EVALUATION.md`](../../../assignments/A1/EVALUATION.md)；日志格式见 [`assignments/A1/README.md` 的《实验日志格式》](../../../assignments/A1/README.md#实验日志格式)。
> 本模板固定报告、代码、脚本、日志和图表的提交位置；各部分照下方占位填写即可。

## 基本信息

- 作业题面版本：26.0.4
- 完成范围：完成了所有要求的实验
- 未完成项：无
- 上游 starter commit：`a158843b20107949f1a8d7df1b05cd33b9166712`
- 本地工作仓库：`../assignment1-basics`（必须与 `SummerQuest-2026` 同级）

## Markdown 报告

### `unicode1`：理解 Unicode

1. `chr(0)` 返回 Unicode 码位 U+0000，即空字符 NUL。
2. 它的 `repr` 是可见的转义形式 `'\x00'`，直接 `print` 时则没有可见字形。
3. NUL 会作为一个长度为 1 的普通字符保留在 Python 字符串内部；它不会截断 Python
   字符串，但输出时通常看不见，某些依赖 C 风格 NUL 结尾字符串的外部系统可能把它误当作终止符。

### `unicode2`：Unicode 编码

1. UTF-8 对 ASCII 只使用 1 字节，英语和代码类语料通常明显小于 UTF-16/UTF-32；它还没有
   UTF-16/UTF-32 的字节序问题，并且是 Web 与现有文本工具链的事实标准。UTF-8 的变长前缀也让
   字节流可以自同步，而 UTF-16 还需要处理代理项对。
2. 例如输入 `"牛".encode("utf-8") == b"\xe7\x89\x9b"` 时，错误函数会在单独解码
   `b"\xe7"` 时抛出 `UnicodeDecodeError`。原因是一个 UTF-8 字符可能跨越多个字节，不能把
   每个字节独立解码后再拼接。
3. `b"\xc0\x80"` 不能解码成任何合法 Unicode 字符：它试图用被 UTF-8 明令禁止的
   overlong 形式表示 U+0000，因此 Python 会抛出 `UnicodeDecodeError`。

### BPE tokenizer 训练

#### 实验设置

- TinyStories 使用完整 `data/TinyStoriesV2-GPT4-train.txt`（2,227,753,162 字节），
  词表大小 10,000。
- OpenWebText 使用完整 `data/owt_train.txt`（11,920,511,059 字节），词表大小 32,000。
- 两者都把 `<|endoftext|>` 作为 special token，并使用 10 个预分词 worker。
- 实验机器为 10 核 Apple M5、16 GB 内存、Python 3.13.12。实验由
  `scripts/train_bpe_experiments.sh` 运行，日志写入 `logs/train_bpe/`；耗时用
  `time.perf_counter()` 记录，主进程峰值 RSS 用 Python `resource` 记录，因此 RSS 是统一、
  可复现的主进程口径，而不是所有 worker RSS 简单相加。

| 数据集 | 词表大小 | Merge 数 | 训练耗时 | 主进程峰值 RSS | 最长 token（按 UTF-8 字节） |
|---|---:|---:|---:|---:|---|
| TinyStories | 10,000 | 9,743 | 32.40 秒 | 1.52 GB | 15 字节：`␠accomplishment`、`␠disappointment`、`␠responsibility` |
| OpenWebText | 32,000 | 31,743 | 1,163.11 秒（19 分 23.11 秒） | 6.01 GB | 64 字节：64 个连字符；另一个并列 token 是重复的 mojibake `ÃÂ…` |

#### `train_bpe_tinystories`

TinyStories 的最长 token 都是带前导空格的常见故事词，符合儿童故事语料中高频完整单词被
逐步合并的直觉。

此前用于定位瓶颈的 TinyStories `cProfile` 结果显示，34.19 秒总时间中约 31.84 秒（约 93%）位于
`run_pipeline` 的并行预分词与结果汇总阶段，而 merge 选择阶段的累计时间约为 2.12 秒；
因此当前训练实现的主要瓶颈是预分词/跨进程结果汇总，而不是构造最后 9,743 个 merge。

#### `train_bpe_expts_owt`

OWT 的最长 token 来自网页分隔线和乱码，反映了真实 Web 语料中的格式噪声；它“有统计意义”，
但不一定有语义价值。相比之下，TinyStories 语料较窄、语言简单，因此词表集中在常见叙事词；
OWT 词表更大，覆盖专有名词、数字、代码、标点串、网页格式和乱码等更广泛的模式。

本次 `logs/train_bpe/` 中记录的 TinyStories 与 OWT 重跑所得 `vocab.json`、`merges.json`
都与仓库现有产物的 SHA-256 完全一致。

### `tokenizer_experiments`：Tokenizer 实验

压缩比实验从各自的完整训练集按文档做水塘抽样，固定 `seed=2026`，抽取 10 篇文档，并把
文档末尾的 `<|endoftext|>` 计入字节数和 token 数。压缩比定义为
`UTF-8 字节数 / token 数`，先汇总全部 10 篇文档再相除。

| 抽样语料 | Tokenizer | 字节数 | Token 数 | 压缩比（bytes/token） |
|---|---|---:|---:|---:|
| TinyStories | TinyStories 10K | 10,994 | 2,602 | 4.2252 |
| OpenWebText | OWT 32K | 32,864 | 6,919 | 4.7498 |
| OpenWebText | TinyStories 10K | 32,864 | 10,225 | 3.2141 |

在同一批 OWT 文档上，TinyStories tokenizer 比匹配语料的 OWT tokenizer 产生多 47.78% 的
token，bytes/token 下降 32.33%。这是因为 TinyStories 词表较小且领域单一，不能把 OWT 中的
长词、实体、数字与网页模式合并成较长 token。不同语料行之间的绝对压缩比不能直接用于判断
tokenizer 优劣；同一 OWT 样本上的交叉比较才控制住了文本内容。

吞吐实验由 `scripts/run_tokenizer_experiments.sh` 运行，日志写入 `logs/tokenizer_experiments/`。
它排除 tokenizer 加载和磁盘 I/O：从各自训练集读取第一个约 4 MiB、按文档边界结束的文本块，
先预热一次，再用单进程连续编码 3 次并取中位数。Pile 按十进制 825 GB 估算，未计入并行加速、
写盘和调度开销。

| Tokenizer | 基准字节数 | 中位耗时 | 单进程吞吐 | 编码 825 GB 的串行估计 |
|---|---:|---:|---:|---:|
| TinyStories 10K | 4,193,638 | 2.1559 秒 | 1.945 MB/s | 117.81 小时（4.91 天） |
| OWT 32K | 4,180,848 | 2.5745 秒 | 1.624 MB/s | 141.12 小时（5.88 天） |

完整训练集与验证集已经编码为以下数组；这些全量数据给出的比值与 10 文档抽样值处于相近范围：

| 数据集切分 | Token 数 | dtype | 全量 bytes/token |
|---|---:|---|---:|
| TinyStories train | 541,229,347 | `uint16` | 4.1161 |
| TinyStories valid | 5,465,883 | `uint16` | 4.1169 |
| OWT train | 2,727,120,452 | `uint16` | 4.3711 |
| OWT valid | 66,401,098 | `uint16` | 4.3674 |

`uint16` 能表示 0–65,535，而两个词表的最大 token ID 分别只有 9,999 和 31,999，所以它能
无损保存全部 token ID，同时只占 `uint32`/`int32` 一半的空间；token ID 不需要负数，因此
无符号类型也是自然选择。

### `transformer_accounting`：Transformer LM 资源核算

记词表大小为 $V$、上下文长度为 $S$、层数为 $L$、模型维度为 $d$、注意力头数为 $h$、
SwiGLU 隐层维度为 $f$。本题架构没有线性层 bias，输入 embedding 与输出 LM head 不共享权重；
RoPE 的正余弦缓存和 causal mask 都不是可训练参数。

#### (a) GPT-2 XL 参数量与加载内存

参数总数为

$$
P = 2Vd + L\left(4d^2 + 3df + 2d\right) + d.
$$

其中 $2Vd$ 来自输入 embedding 与输出 LM head；每层的 $4d^2$ 来自 Q、K、V、O 四个投影，
$3df$ 来自 SwiGLU 的三个矩阵，$2d$ 来自两个 RMSNorm，最后的 $d$ 是最终 RMSNorm。

代入 $V=50{,}257$、$S=1{,}024$、$L=48$、$d=1{,}600$、$h=25$、$f=4{,}288$：

- 输入 embedding + LM head：160,822,400 个参数；
- 48 个 Transformer block：1,479,628,800 个参数；
- 最终 RMSNorm：1,600 个参数；
- 总计 **1,640,452,800 个参数**。

全部使用 float32 时，仅加载参数需要 6,561,811,200 字节，即 **6.562 GB**（**6.111 GiB**）。

#### (b) GPT-2 XL 单序列前向矩阵乘法 FLOPs

矩阵乘法使用 $m\times n$ 与 $n\times p$ 相乘需要 $2mnp$ FLOPs 的口径。Embedding lookup、
RMSNorm、RoPE、softmax 和逐元素操作不计入本小问的矩阵乘法 FLOPs。

| 组件 | FLOPs 公式 | GPT-2 XL FLOPs |
|---|---:|---:|
| Q、K、V 投影 | $6LSd^2$ | 754,974,720,000 |
| 注意力分数 $QK^\top$ | $2LS^2d$ | 161,061,273,600 |
| 注意力加权 $AV$ | $2LS^2d$ | 161,061,273,600 |
| 注意力输出投影 | $2LSd^2$ | 251,658,240,000 |
| SwiGLU 的 $W_1,W_3,W_2$ | $6LSdf$ | 2,023,332,249,600 |
| 输出 LM head | $2SdV$ | 164,682,137,600 |
| **总计** | $L(8Sd^2+4S^2d+6Sdf)+2SdV$ | **3,516,769,894,400** |

因此，一个长度为 1,024 的 GPT-2 XL 单序列前向传播约需 **3.517 TFLOPs** 的矩阵乘法。

#### (c) 哪些组件 FLOPs 最多

SwiGLU 占总 FLOPs 的 57.53%，是最大项；全部注意力投影与两次注意力矩阵乘法合计约
37.78%，输出 LM head 占 4.68%。因此在该上下文长度下，FFN 而不是 $S^2$ 注意力是主要计算瓶颈。

#### (d) GPT-2 各尺寸的 FLOPs 构成

下表沿用仓库实现选择的 64 倍数 $f$：Small/Medium/Large/XL 分别为
2,048/2,752/3,456/4,288。比例把 QKV 与 O 合为“注意力线性投影”，把 $QK^\top$ 与 $AV$
合为“注意力核心”。

| 模型 | 总 FLOPs | 注意力线性投影 | 注意力核心 | SwiGLU | LM head |
|---|---:|---:|---:|---:|---:|
| GPT-2 Small | 0.29165 TFLOPs | 19.88% | 13.25% | 39.76% | 27.10% |
| GPT-2 Medium | 0.83017 TFLOPs | 24.83% | 12.42% | 50.05% | 12.70% |
| GPT-2 Large | 1.78665 TFLOPs | 27.04% | 10.82% | 54.76% | 7.37% |
| GPT-2 XL | 3.51677 TFLOPs | 28.62% | 9.16% | 57.53% | 4.68% |

随着 $d$ 和 $L$ 增大，FFN 与线性投影的 $d^2$ 级成本占比上升；固定 $S=1{,}024$ 时，
注意力核心的相对占比缓慢下降，未共享的 LM head 因只随 $d$ 线性增长而快速下降。

#### (e) 把 GPT-2 XL 上下文增至 16,384

总 FLOPs 从 3.51677 TFLOPs 增加到 **133.57773 TFLOPs**，是原来的 **37.98 倍**，而不是
简单的 16 倍。注意力核心因 $S^2$ 增长，其占比从 9.16% 上升到 61.73%；注意力线性投影、
SwiGLU 和 LM head 的占比分别降到 12.06%、24.24% 和 1.97%。

### `learning_rate_tuning`：学习率实验

实验严格使用题目中的 $10\times10$ 参数矩阵、`loss=(weights**2).mean()`、SGD 和 10 次迭代，
并固定 `torch.manual_seed(2026)`。表中的末值是第 10 次迭代反向传播前记录的 loss。

| 学习率 | 初始 loss |          第 10 次迭代 loss | 现象   |
|---:|---:|-----------------------:|------|
| 1 | 27.6607 |                22.8895 | 缓慢下降 |
| 10 | 27.6607 |                 3.7174 | 快速下降 |
| 100 | 27.6607 |             1.4678e-23 | 极速下降 |
| 1,000 | 27.6607 |             2.5637e+18 | 爆炸发散 |

$$w_{t+1} = w_t - \eta \cdot \frac{\partial L}{\partial w_i} = w_t - \eta \cdot (0.02 w_t) = (1 - 0.02\eta) w_t$$
而当采用 $$\eta = \frac{lr }{ math.sqrt(t + 1)}$$, 每次更新就会出现上述情景，可以直接推算。

### `adamw_accounting`：AdamW 训练资源核算

#### (a) 峰值内存表达式

沿用上面的参数量 $P$。设 batch size 为 $B$，并按照题目要求只把列出的中间结果各保留一次：
每层计两个 RMSNorm 输出、QKV、注意力分数、softmax、加权 value、输出投影、SwiGLU 的
$W_1/W_3$ 输出、SiLU、逐元素乘积与 $W_2$ 输出；层外再计最终 RMSNorm、logits 和每位置一个
交叉熵标量。激活元素数为

$$
A = B\left\{L\left(2hS^2+8Sd+4Sf\right)+Sd+SV+S\right\},
$$

其中题目要求 $f\approx\frac{8}{3}d$，实际模型取附近的 64 倍数。如果暂时忽略取整，
$P=2Vd+L(12d^2+2d)+d$，且
$A=B\{L(2hS^2+\frac{56}{3}Sd)+Sd+SV+S\}$，因此表达式只依赖题目列出的
$B,V,S,L,d,h$。全部张量使用 float32 时：

| 类别 | 内存 |
|---|---:|
| 参数 | $4P$ 字节 |
| 激活 | $4A$ 字节 |
| 梯度 | $4P$ 字节 |
| AdamW 一阶、二阶矩 | $8P$ 字节 |
| **总计** | **$16P+4A$ 字节** |

这是题目指定的简化核算；真实 PyTorch 峰值还会受到临时 kernel workspace、CUDA allocator、
激活释放时机和实现是否重计算激活等因素影响。

#### (b) GPT-2 XL 与 80 GB 上限

GPT-2 XL 有 $P=1{,}640{,}452{,}800$，代入上式得到

$$
M(B)=16.16754\,B+26.24724\ \text{GB},
$$

其中参数、梯度、optimizer state 分别占 6.56181 GB、6.56181 GB、13.12362 GB，激活占
$16.16754B$ GB；这里对 GPT-2 XL 使用题目给定的取整值 $f=4{,}288$。或用二进制单位写成
$15.05720\,B+24.44465$ GiB。按十进制 80 GB，$B=3$ 时约
74.75 GB，$B=4$ 时约 90.92 GB，因此理论最大 batch size 是 **3**；按 80 GiB 计算结论仍为 3。

#### (c) 一次 AdamW step 的 FLOPs

忽略只对标量执行一次的 bias-correction 计算，并把乘、加、减、除、平方根各记为 1 FLOP：
weight decay 为 $2P$，一阶矩更新为 $3P$，二阶矩更新为 $4P$，归一化参数更新为 $5P$，
总计 **$14P$ FLOPs**。对 GPT-2 XL，这等于 22,966,339,200 FLOPs，约 22.97 GFLOPs，
相对模型前向/反向计算很小。

#### (d) 单张 H100 训练 400K steps 的时长

GPT-2 XL 单序列前向为 $F=3.5167698944\times10^{12}$ FLOPs。batch size 1,024 时，按反向为
前向两倍，一步模型计算是 $3BF$；再加 $14P$ 的 AdamW 更新。50% MFU 下的有效吞吐为
$0.5\times495=247.5$ TFLOP/s，因此

$$
t=\frac{400{,}000\left(3\times1{,}024\times F+14P\right)}{247.5\times10^{12}}
=17{,}460{,}267\ \text{s}
\approx\mathbf{4{,}850.07\ \text{小时}}
\approx\mathbf{202.09\ \text{天}}.
$$

### TinyStories 主训练

正式 TinyStories run 使用 batch size 128、context length 256、4 层、`d_model=512`、16 heads、
SwiGLU `d_ff=1344`、RoPE、pre RMSNorm、10K TinyStories tokenizer 和 peak LR 0.003。

| 状态 | 最终 step | 已处理 token | 最终训练 loss | 训练期验证 loss | Full val loss | Wall-clock time | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| completed | 10,000 | 327,680,000 | 1.3027 | 1.3413 | 1.3380 | 1,052.5 秒 | 311,330 tok/s |

最终 checkpoint 在训练期验证和 full validation 上都达到稳定的低损失；full validation 略低于训练期
最后一次验证，说明训练期小批验证口径没有明显高估模型质量。

### 学习率 sweep

所有候选 run 使用同一模型、batch size 128、context length 256、8,192,000 token 预算和相同
warmup/cosine schedule，只改变 peak learning rate。发散 run 的验证 loss 只是停止前日志值，
主要判据是 `nonfinite_or_large_train_loss`。

| Peak LR | 状态 | 最终 step | 已处理 token | 最终训练 loss | 验证 loss | Wall-clock time |
|---:|---|---:|---:|---:|---:|---:|
| 1e-4 | completed | 250 | 8,192,000 | 4.0062 | 3.9554 | 39.4 秒 |
| 3e-4 | completed | 250 | 8,192,000 | 3.1614 | 3.1017 | 30.4 秒 |
| 1e-3 | completed | 250 | 8,192,000 | 2.6315 | 2.5594 | 30.8 秒 |
| 3e-3 | completed | 250 | 8,192,000 | 2.5731 | 2.5036 | 30.6 秒 |
| 1e-2 | completed | 250 | 8,192,000 | 3.5485 | 3.4920 | 31.1 秒 |
| 3e-2 | completed | 250 | 8,192,000 | 4.3064 | 4.2590 | 30.9 秒 |
| 1e-1 | diverged | 5 | 163,840 | 25.4723 | 7.1518 | 1.3 秒 |
| 3e-1 | diverged | 2 | 65,536 | 25.7844 | 6.9645 | 1.1 秒 |
| 1e1 | diverged | 1 | 32,768 | 350.8913 | 351.5475 | 0.9 秒 |

在相同短 token 预算下，peak LR 0.003 的验证 loss 最低，因此主训练和消融实验采用 0.003。
0.01 和 0.03 能完整跑完但损失明显更高；0.1 及以上迅速发散，说明这个模型和 batch 配置下的
稳定学习率上界在 0.03 与 0.1 之间。

### Batch-size 实验

Batch-size sweep 固定 peak LR 0.001、context length 256 和 8,192,000 processed tokens，
比较 batch size 1 到 256。由于 token 预算固定，较小 batch 会执行更多 optimizer steps。

| Batch size | Steps | 已处理 token | 最终训练 loss | 验证 loss | Wall-clock time | 吞吐 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32,000 | 8,192,000 | 2.9500 | 2.4938 | 600.8 秒 | 13,635 tok/s |
| 2 | 16,000 | 8,192,000 | 2.4074 | 2.3396 | 310.2 秒 | 26,405 tok/s |
| 4 | 8,000 | 8,192,000 | 2.1936 | 2.1797 | 158.2 秒 | 51,772 tok/s |
| 8 | 4,000 | 8,192,000 | 2.2220 | 2.0682 | 83.4 秒 | 98,180 tok/s |
| 16 | 2,000 | 8,192,000 | 2.3227 | 2.0490 | 49.9 秒 | 164,175 tok/s |
| 32 | 1,000 | 8,192,000 | 2.2588 | 2.1285 | 39.8 秒 | 205,838 tok/s |
| 64 | 500 | 8,192,000 | 2.3450 | 2.2883 | 35.0 秒 | 234,179 tok/s |
| 128 | 250 | 8,192,000 | 2.6315 | 2.5737 | 32.4 秒 | 253,048 tok/s |
| 256 | 125 | 8,192,000 | 3.0329 | 2.9684 | 31.1 秒 | 263,287 tok/s |

短预算下 batch size 16 的验证 loss 最低，batch size 8 非常接近；batch size 128/256 吞吐最高，
但 optimizer steps 太少，验证损失明显更差。batch size 1 虽然 steps 最多，但吞吐极低且噪声大，
没有换来最好的验证 loss。

### 消融实验

消融 run 与 baseline 共享 optimizer、schedule、batch size 128、context length 256、327,680,000
token 预算、验证协议和随机种子；每个 run 只改变表中列出的变量。消融 full validation 默认评估
`best.pt`，baseline 的 best 和 final validation step 重合。

| 变体 | 改变的变量 | 状态 | 最终 step | 训练期验证 loss | Full val loss | Wall-clock time | 解释 |
|---|---|---|---:|---:|---:|---:|---|
| Baseline | 无 | completed | 10,000 | 1.3413 | 1.3380 | 1,052.5 秒 | 作为对照 |
| 删除 RMSNorm | `norm_mode=none` | diverged | 242 | 37.3082 | 37.4277 | 35.8 秒 | 训练 loss 爆到 1634.3040，RMSNorm 对稳定训练是必要的 |
| Post-Norm | `norm_mode=post` | completed | 10,000 | 1.3819 | 1.3784 | 861.4 秒 | 比 pre-norm 差约 0.0404 full-val loss |
| NoPE | `use_rope=false` | completed | 10,000 | 1.4008 | 1.3969 | 774.1 秒 | 去掉位置编码后最差，说明位置信息对故事建模有明显帮助 |
| SiLU | `ffn_type=silu`, `d_ff=2048` | completed | 10,000 | 1.3483 | 1.3451 | 840.9 秒 | 接近 baseline，但仍高约 0.0071 full-val loss |

四个消融里，删除 RMSNorm 是唯一直接发散的设置；NoPE 和 post-norm 都能训练完，但验证损失稳定
劣于 baseline。参数量近似匹配的 SiLU FFN 与 SwiGLU 最接近，不过在 full validation 上仍略差。

### OpenWebText 主训练

OWT run 使用 32K OWT tokenizer、context length 512、batch size 128、8 层、`d_model=768`、
8 heads、SwiGLU `d_ff=2048`、RoPE、pre RMSNorm、peak LR 0.003，并开启 `torch.compile`
和 bf16 autocast。该 run 设置 5,400 秒 wall-clock 上限为了查看1.5 H100 hours的效果，因此停止原因是 `max_wall_clock_sec`。

| 状态 | 最终 step | 已处理 token | 最终训练 loss | 训练期验证 loss | Full val loss | Wall-clock time | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| time_limit | 22,822 | 1,495,662,592 | 3.3315 | 3.3907 | 3.3618 | 5,401.6 秒 | 276,892 tok/s |

OWT 的 full-val loss 明显高于 TinyStories，这主要来自语料更开放、词表更大和上下文长度更长，
不能直接解释为模型实现退化。该 run 在 1.5 小时限制下处理了约 1.50B token，停止时学习率仍为
0.00131，属于时间预算截断而非训练完全收敛。

### 文本生成

文本生成使用 TinyStories `best.pt`，采样设置为 temperature 0.8、top-p 0.9、seed 1337。
三条样本都以 `<eos>` 停止，生成长度为 133 到 217 tokens。

> **Prompt：** Once upon a time
>
> **生成文本节选（与 prompt 拼接后）：** Once upon a time, there was a little boy named Tim.
> Tim loved to help his mom in the kitchen. One day, his mom was making a cake. ... Tim and his mom
> shared the cake with their family, and everyone was very happy.

样本整体符合 TinyStories 风格：句子短、语法基本正确，有明确人物、行动、结果和温和结尾。主要
失败模式是局部重复和语义松散，例如同一故事里多次说 “the cake was done”，另一个样本里鸟的行为
和人物对话较不自然。这说明模型已经学到儿童故事的表面结构，但长程一致性和细节因果仍有限。


## 复现说明

- 环境与依赖：使用课程提供的 Python 项目环境，依赖由上游 `uv.lock` 固定；实验环境为 Python 3.13.12。Tokenizer/BPE 实验在本地 Apple M5 机器上复现，长时间语言模型训练实验在单卡 H100/H200 上运行。报告和日志中不依赖内部绝对路径。
- 数据准备：从课程公开数据源准备 TinyStoriesV2-GPT4 与 OpenWebText 的 train/valid 文本切分，放到 `data/` 下，文件名分别为 `TinyStoriesV2-GPT4-train.txt`、`TinyStoriesV2-GPT4-valid.txt`、`owt_train.txt`、`owt_valid.txt`。先训练 BPE tokenizer 到 `artifacts/TinyStories_train_bpe/` 和 `artifacts/owt_train_bpe/`，再编码 token-id 数组到 `artifacts/token_ids/`。原始数据、模型 checkpoint、虚拟环境和依赖锁不放入提交包。
- Tokenizer、训练与生成命令：
  ```bash
  bash scripts/train_bpe_experiments.sh
  bash scripts/run_tokenizer_experiments.sh --encode-arrays --strict

  bash scripts/search_lr.sh
  bash scripts/search_batch_size.sh
  bash scripts/train_tinystories.sh
  bash scripts/run_ablations.sh
  bash scripts/train_owt.sh

  bash scripts/evaluate_tinystories_full_val.sh
  bash scripts/evaluate_ablations_full_val.sh
  bash scripts/evaluate_owt_full_val.sh
  bash scripts/generate_text.sh
  ```
- 同步命令：`python3 scripts/sync_a1_submission.py --name '<姓名>'`
- 配置文件：`submission/configs/train_bpe_experiments.json`、`submission/configs/tokenizer_experiments.json`、`submission/configs/search_lr.json`、`submission/configs/search_batch_size.json`、`submission/configs/train_tinystories.json`、`submission/configs/ablations.json`、`submission/configs/train_owt.json`、`submission/configs/generate_text.json`

## 代码与脚本

- 真实实现：`submission/cs336_basics/`
- 测试 adapter：`submission/tests/adapters.py`
- 训练、数据编码与生成脚本：`submission/scripts/`
- 实现说明：核心实现包括 BPE tokenizer 训练、BPE tokenizer 编解码、Transformer LM、RoPE、causal multi-head self-attention、RMSNorm、SwiGLU/SiLU FFN、cross entropy、AdamW/SGD optimizer、学习率调度、checkpoint 保存与恢复、训练日志、full validation 评测和文本生成。实验脚本从 `configs/` 读取配置，训练日志写入 `logs/`，模型 checkpoint 写入 `checkpoint/`，BPE 产物写入 `artifacts/*_train_bpe/`，token-id 数组写入 `artifacts/token_ids/`。

真实实现先在兄弟目录 `../assignment1-basics` 中完成并通过官方测试，再使用同步命令复制
到本目录。不要手工复制公共 tests、fixtures、数据、模型权重、虚拟环境或依赖锁。

## 实验日志

- 日志目录：`logs/`
- 文件与格式：见 [`assignments/A1/README.md` 的《实验日志格式》](../../../assignments/A1/README.md#实验日志格式)
- 与报告中实验的对应说明：`logs/train_bpe/` 对应 BPE tokenizer 训练实验；`logs/tokenizer_experiments/` 对应压缩比、吞吐和全量 token-id 编码统计；`logs/lr_sweep/` 对应 TinyStories learning-rate search；`logs/batch_size_sweep/` 对应 batch size search；`logs/train_tinystories.jsonl`、`logs/train_tinystories.summary.json` 和 `logs/train_tinystories_full_val.json` 对应 TinyStories 主训练与完整验证集评测；`logs/ablations/` 和 `logs/ablations_full_val/` 对应四个消融实验及其 full validation；`logs/train_owt.jsonl`、`logs/train_owt.summary.json` 和 `logs/train_owt_full_val.json` 对应 OpenWebText 主训练与完整验证集评测；`logs/generation_samples.jsonl` 对应 TinyStories checkpoint 的文本生成样本。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/FtU9wIHJyijoJVktRu0cl8DQnlh

该文档设置为组织内公开，不得开启互联网公开访问，只保存不能公开到 GitHub 但确有
审核必要的最小差量材料。
