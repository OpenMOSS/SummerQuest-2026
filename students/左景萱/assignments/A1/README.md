# A1 公开提交：左景萱

> 本文件和同目录代码公开可见，只包含可公开、可复现且已经脱敏的内容。数据集、模型
> checkpoint、运行环境缓存和访问凭据均不进入提交。

> 评分标准见 [EVALUATION.md](../../../../assignments/A1/EVALUATION.md)，实验日志要求见
> [A1 README](../../../../assignments/A1/README.md#实验日志格式)。

## 基本信息

- 作业题面版本：SummerQuest 26.0.4；书面核算按 Stanford 原题 Version 26.0.3。
- 完成范围：byte-level BPE、Tokenizer、Transformer LM、AdamW 与训练工具、训练/编码/
  生成/实验编排脚本，以及 Unicode 和资源核算书面题。
- 实验状态：代码、tokenizer、数据编码、TinyStories/OWT 训练、学习率与 batch
  sweep、架构消融、文本生成、脱敏日志和 loss 曲线均已实际运行并回填。
- 上游 starter commit：`a158843b20107949f1a8d7df1b05cd33b9166712`
- 本地工作仓库：`../assignment1-basics`（与 SummerQuest-2026 同级）
- 默认随机种子：42

## 书面题

### Unicode 1：理解 Unicode

1. `chr(0)` 返回 Unicode 空字符 NUL，即 U+0000。
2. 它的 `repr` 显示为转义形式 `'\x00'`，直接打印时则不可见。
3. NUL 仍是 Python 字符串中的一个真实字符，会占据一个字符位置并参与拼接和长度计算；
   “不可见”不等于“不存在”（部分以 NUL 结尾的外部 C 接口可能另行把它解释为终止符）。

### Unicode 2：Unicode 编码

1. UTF-8 对 ASCII 使用单字节，英文占比较高的语料通常比 UTF-16/UTF-32 更紧凑；它还与
   ASCII 向后兼容、无端序问题，并且仍能表示全部 Unicode 码点。因此，以 UTF-8 bytes
   作为初始的 256 项词表既避免 OOV，又减少常见文本的基础序列长度。
2. 反例为 `"牛".encode("utf-8") == b"\xe7\x89\x9b"`。错误函数逐 byte 解码，
   但 UTF-8 的一个字符可能由多个 bytes 共同组成；单独解码第一个先导 byte
   `b"\xe7"` 就会抛出 `UnicodeDecodeError`。
3. `b"\x80\x80"` 无法解码为任何 Unicode 字符：两个 byte 都是 UTF-8
   continuation byte，却没有合法的 leading byte。

## Transformer 与 AdamW 资源核算

以下核算遵循原题 Version 26.0.3 的 untied input embedding / LM head、SwiGLU、每层两个
RMSNorm、无 bias 模型。记词表大小为 \(V\)，序列长度为 \(S\)，层数为 \(L\)，模型宽度为
\(d\)，SwiGLU 隐层宽度为 \(f\)，注意力头数为 \(h\)，microbatch size 为 \(B\)。
矩阵乘法中的一次乘法和一次加法计 2 FLOPs；embedding lookup、RoPE、归一化和逐元素操作
相对较小，在题目的主项 FLOPs 统计中忽略。

### 参数量

- 输入 embedding 与独立 LM head：\(2Vd\)。
- 每层 Q/K/V/O 四个投影：\(4d^2\)。
- 每层 SwiGLU 三个投影：\(3df\)。
- 每层两个 RMSNorm：\(2d\)；最终 RMSNorm：\(d\)。

因此总参数量为

\[
P = 2Vd + L(4d^2 + 3df + 2d) + d.
\]

取 \(V=50{,}257\)、\(S=1{,}024\)，结果如下。FP32 参数显存只计参数本身，即 \(4P\)
bytes，并用 \(2^{30}\) bytes/GiB 换算。

| 模型 | \(L\) | \(d\) | \(f\) | 参数量 \(P\) | FP32 参数显存 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 12 | 768 | 2,048 | 162,148,608 | 0.604 GiB |
| medium | 24 | 1,024 | 2,752 | 406,539,264 | 1.514 GiB |
| large | 36 | 1,280 | 3,392 | 833,591,040 | 3.105 GiB |
| XL | 48 | 1,600 | 4,288 | 1,640,452,800 | 6.111 GiB |

### 单样本 forward FLOPs

每层四个投影贡献 \(8Sd^2\)，注意力的 \(QK^\top\) 和 \(AV\) 合计贡献 \(4S^2d\)，
SwiGLU 三个投影贡献 \(6Sdf\)，LM head 贡献 \(2SdV\)。因此

\[
F = L(8Sd^2 + 4S^2d + 6Sdf) + 2SdV.
\]

| 模型 | forward FLOPs | QKV+O | \(QK^\top+AV\) | SwiGLU | LM head |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 0.291648 TFLOPs | 19.881% | 13.254% | 39.762% | 27.104% |
| medium | 0.830172 TFLOPs | 24.833% | 12.417% | 50.054% | 12.696% |
| large | 1.768531 TFLOPs | 27.321% | 10.928% | 54.301% | 7.449% |
| XL | 3.516770 TFLOPs | 28.624% | 9.160% | 57.534% | 4.683% |

XL 的 \(S\) 从 1,024 增加到 16,384 后，单样本 forward 为 133.5777 TFLOPs，是原来的
37.983 倍，而不是 16 倍。原因是注意力矩阵项随 \(S^2\) 增长；此时 FLOPs 构成为
QKV+O 12.057%、attention 61.734%、SwiGLU 24.236%、LM head 1.973%，attention 已成为
主导项。

### AdamW 训练显存

按题目要求，以 FP32 训练且不考虑 activation checkpointing。保存的 activation 元素数为

\[
A = L\left[BS(8d+4f)+2BhS^2\right] + BSd + 2BSV.
\]

每个参数需要参数值、梯度、一阶矩和二阶矩，共 \(4+4+4+4=16\) bytes；每个 activation
为 4 bytes。因此

\[
M = 16P + 4A \quad \text{bytes}.
\]

对 XL（\(h=25,S=1{,}024\)），参数、梯度与 Adam 状态固定占
\(16P=24.4447\) GiB；每增加一个 microbatch 样本，activation 增加
\(4A/B=15.2489\) GiB，所以

\[
M(B)=24.4447+15.2489B\quad\text{GiB}.
\]

80 GiB 显存下满足该估算的最大整数 microbatch size 为
\(\lfloor(80-24.4447)/15.2489\rfloor=3\)。这是按题目简化项得到的理论上限；实际运行还需
为 allocator、kernel workspace 和框架开销留余量。

### AdamW 更新 FLOPs 与 H100 训练时间

AdamW 对每个参数更新一阶矩、二阶矩并执行归一化与 decoupled weight decay，按原题的
逐元素核算约为 \(14P\) FLOPs/step。XL 因而需要

\[
14\times1{,}640{,}452{,}800=22.966\text{ GFLOPs/step},
\]

远小于模型 forward/backward 的矩阵乘法开销。

若 XL 使用全局 batch size 1,024 训练 400,000 steps，并近似 backward 为 forward 的
2 倍，则总训练计算量为 \(3F\times1{,}024\times400{,}000\)。以单张 H100 的
BF16 峰值 495 TFLOP/s、50% MFU 估计：

\[
t =
\frac{3\times3.5167698944\times10^{12}\times1{,}024\times400{,}000}
{0.5\times495\times10^{12}}
=4{,}850.1\text{ hours}
=202.1\text{ days}.
\]

这里使用的是原题 2026 版 H100 口径。

## 实现说明

### Byte-level BPE 与 Tokenizer

- 初始 vocabulary 包含 256 个单 byte token，并把 `<|endoftext|>` 当作
  不可拆分的文档边界；普通文本使用题目指定的 GPT-2 正则预分词。
- BPE 训练先统计唯一 pre-token 及频次，再增量维护 pair count 与 pair-to-word 倒排索引；
  最大堆采用惰性失效，频次相同时选择字典序最大的 pair，保证确定性并避免每轮全量重算。
- 大文件预分词支持按 special token 边界切分和多进程计数；编码、指标计算与二进制写出均
  提供流式路径，避免一次性加载完整语料。
- 编码严格按训练时 merge rank 合并；解码先拼接所有 token bytes，再以
  `errors="replace"` 整体 UTF-8 解码。词表与 merge 使用 GPT-2 兼容格式持久化。

### Transformer LM

- 从张量操作实现无 bias Linear、Embedding、RMSNorm、SiLU/SwiGLU、RoPE、scaled
  dot-product attention 和 causal multi-head self-attention。
- RMSNorm 的 reduction 与 attention softmax 在 FP32 中计算，以提升低精度训练稳定性；
  RoPE 的 sin/cos 是非持久 buffer，不计入可学习参数。
- 基线采用 Pre-Norm residual block、RoPE、SwiGLU、最终 RMSNorm，以及互不共享的 token
  embedding 和 LM head。实现还支持 no-RMSNorm、Post-Norm、NoPE 和参数量近似匹配的
  SiLU FFN 消融。
- 生成器裁剪到 context window，并支持 temperature、top-p nucleus sampling 和 EOS
  提前停止。

### 优化与训练

- 数值稳定的 softmax/cross-entropy 使用 max-shift 与 log-sum-exp；全局梯度裁剪按所有
  参数梯度的联合 L2 norm 缩放。
- AdamW 从 Optimizer 基类实现一、二阶矩、bias correction 和 decoupled weight decay；
  学习率为 linear warmup 后 cosine decay。
- 训练数据通过 NumPy memmap 读取随机连续窗口；验证使用固定种子且恢复外部 RNG 状态，
  便于跨 step 比较。
- 训练入口支持 BF16 autocast、可选 torch.compile、断点恢复、best/latest/final checkpoint；
  JSONL 逐点记录 step、wall_clock_sec、train_loss、val_loss、lr 和 processed tokens，
  summary.json 保存最终配置和汇总。
- 最终 GPU 验收 run 使用 BF16 eager（`compile=false`）；compile 独立通过预检，避免将
  编译器与低精度交互引入长训练变量。
- 多 GPU 实验编排是一卡一个独立 run，用于并行完成 baseline、学习率扫、四项消融和 OWT；
  batch-size benchmark 逐步增加 batch，记录吞吐、峰值显存并在 OOM 后停止。

## 实验结果

以下数值均由脚本和原始 JSONL 日志产生；脱敏后的逐点记录、summary 与总验收结果见
[`logs/`](logs/)。

### Tokenizer

| 训练语料 | 词表大小 | 训练时间 | 域内验证集 bytes/token | 最长普通 token | bytes/s | tokens/s |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| TinyStories | 10,000 | 10.352 s | 4.1169 | ` accomplishment`（15 bytes） | 567,261 | 137,788 |
| OpenWebText | 32,000 | 630.658 s | 4.3673 | `ÃÂ` 重复串（64 bytes） | 452,975 | 103,720 |

训练时间来自当前 54-core CPU 环境：TinyStories 使用 48 workers；OWT 也使用 48 workers。
完整验证集上的统一单进程测量如下，bytes/token 越高表示压缩越好。

| tokenizer | 验证语料 | token 数 | bytes/token | source MB/s | tokens/s |
| --- | --- | ---: | ---: | ---: | ---: |
| TinyStories 10K | TinyStories | 5,465,883 | 4.1169 | 0.5673 | 137,788 |
| TinyStories 10K | OWT | 91,369,966 | 3.1739 | 0.4810 | 151,537 |
| OWT 32K | TinyStories | 5,617,816 | 4.0056 | 0.5313 | 132,635 |
| OWT 32K | OWT | 66,402,215 | 4.3673 | 0.4530 | 103,720 |

域内优势很清楚：Tiny tokenizer 在 TinyStories 上比 OWT tokenizer 多压缩约 2.78%，而
OWT tokenizer 在 OWT 上比 Tiny tokenizer 多压缩约 37.60%。32K merge 表的单进程查找
开销也更高，因此相同语料上的 source bytes/s 略低。TinyStories 最长普通 token
` accomplishment` 是合理的高频自然片段；OWT 的 64-byte 最长 token 是网页数据中的
mojibake 重复模式，不是自然词，也没有跨越 `<|endoftext|>` 文档边界。

TinyStories 全量 BPE 的独立 profile 重跑耗时 10.877 s；每 50 ms 汇总父进程和所有 worker
的 RSS，50 个峰值进程的 aggregate RSS 为 6,547,410,944 bytes（6.10 GiB）。5 MiB 固定
样本的 cProfile 中，预分词计数 `_count_corpus_pretokens` 占 1.707/3.024 s（约 56%），
其中 Counter 更新和 byte tuple 构造占主要部分；实际 `_merge_pair` 仅占 0.192 s，说明
lazy heap 已把 merge 维护从主要瓶颈降为次要开销。

若按完整 OWT valid 的单进程 452,975 bytes/s 编码 825 GB Pile，需要约 21.1 天；实际
48-worker OWT train 编码达到 21.19 MB/s，对应理想化约 10.8 小时。后者已包含本实现的
分块、worker 启动和有序拼接开销，但跨机器估算仍会受 I/O 与 CPU 配额影响。

### TinyStories 基线

基线配置为 \(V=10{,}000,S=256,d=512,L=4,h=16,f=1{,}344\)，batch size 128，
训练 10,000 steps；AdamW 峰值学习率 \(1\times10^{-3}\)，500-step warmup，余弦退火到
\(3\times10^{-5}\)，weight decay 0.1，gradient clip 1.0。

| run | steps | processed tokens | final train loss | final/best val loss | PPL | wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyStories baseline | 10,000 | 327,680,000 | 1.365764 | **1.370364** | 3.9368 | 823.320 s |

final validation loss 比 1.45 门槛低 0.079636。它从 step 0 的 9.259990 持续降到
1.370364；step 6,000 时已达 1.445997，后续仍继续改善，未见过拟合回升。曲线见
[step 横轴](assets/tinystories_loss_by_step.svg) 和
[wall-clock 横轴](assets/tinystories_loss_by_wall_clock.svg)。

### 学习率扫描

每个 run 使用相同初始种子和架构，训练 1,500 steps；除学习率外其余条件保持一致。

| max LR | 状态 | steps | final train loss | final/best val loss | wall time | 结论 |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| \(1\times10^{-4}\) | completed | 1,500 | 2.626446 | 2.592490 | 126.724 s | 学习率过小，收敛慢 |
| \(3\times10^{-4}\) | completed | 1,500 | 2.114150 | 2.083701 | 126.698 s | 优于 \(10^{-4}\)，但仍欠拟合 |
| \(1\times10^{-3}\) | completed | 1,500 | 1.729356 | **1.702899** | 126.278 s | 正常完成 run 中最佳 |
| \(3\times10^{-2}\) | completed | 1,500 | 2.780065 | 2.748372 | 129.288 s | 越过最佳区，质量反转 |
| \(1\times10^{-1}\) | completed | 1,500 | 3.634242 | 3.609175 | 126.070 s | 数值有限，但严重退化 |
| \(3\times10^{-1}\) | completed | 1,500 | 5.030360 | 5.018348 | 125.541 s | 数值有限，接近失效 |
| \(1\) | completed | 1,500 | 4.706799 | 5.035303 | 125.375 s | 数值有限，接近失效 |
| \(30\) | **diverged** | **45** | **668.014343** | 9.259990† | **4.750 s** | **non_finite_gradient** |

† LR=30 在下一次验证前已发散，9.259990 是 step 0 初始验证值，不代表训练后质量。
该 run 在 warmup 中的 step 10/20/30/40 train loss 为
603.708/155.983/14.006/152.793；step 45（当时 LR=13.5）升到 668.014，并出现非有限梯度。
结果表明 \(10^{-3}\) 是本次 1,500-step 扫描的经验最佳点；\(3\times10^{-2}\) 起性能明显
反转，而 Pre-Norm 和梯度裁剪等使 0.03–1.0 仍保持有限数值，LR=30 则给出了明确的
数值发散实例。可视化见
[稳定区 step 曲线](assets/lr_sweep_stable_loss_by_step.svg)、
[稳定区 wall-clock 曲线](assets/lr_sweep_stable_loss_by_wall_clock.svg) 和
[全扫描曲线](assets/lr_sweep_full_loss_by_step.svg)。

### Batch size 与吞吐

每个 batch size 运行 50 steps，记录 tokens/s、峰值显存和是否 OOM。该实验比较的是硬件
利用率与显存边界，不把不同 batch 下 50-step 的 loss 直接当作收敛质量比较。

| batch | 状态 | tokens/s | peak memory |
| ---: | --- | ---: | ---: |
| 1 | completed | 8,630 | 0.47 GiB |
| 2 | completed | 24,529 | 0.59 GiB |
| 4 | completed | 49,648 | 0.82 GiB |
| 8 | completed | 99,348 | 1.28 GiB |
| 16 | completed | 194,201 | 2.20 GiB |
| 32 | completed | 305,877 | 4.04 GiB |
| 64 | completed | 372,613 | 7.73 GiB |
| 128 | completed | 408,791 | 15.09 GiB |
| 256 | completed | 429,834 | 29.83 GiB |
| 512 | completed | **448,279** | 59.30 GiB |
| 1,024 | completed | 447,817 | 118.25 GiB |
| 2,048 | **OOM** | 不适用 | 131.59 GiB（OOM 前峰值） |

实际扫描在首个 OOM 后按设计停止，因此未把 4,096 误写为“已测试”。batch 512 吞吐最高；
1,024 的峰值显存近乎翻倍而吞吐略降，说明约在 512 已进入饱和区。题目要求的 64 与
128 均完整运行，首个 OOM 为 2,048。

### 架构消融

四个必做消融均沿用 TinyStories baseline 的数据、seed、训练步数和优化器设置；
no-RMSNorm 的低 LR run 是额外稳定性对照。SiLU 使用
\(d_{ff}=2{,}048\)：其两投影参数量 \(2d\times2{,}048\) 与 baseline SwiGLU 的三投影
参数量 \(3d\times1{,}344\) 近似匹配，避免把 FFN 参数量差异误认为 activation 的效果。

| 模型 | max LR | 参数量 | final train | final/best val | PPL | wall time | \(\Delta\) val |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | \(10^{-3}\) | 22,696,448 | 1.365764 | 1.370364 | 3.9368 | 823.320 s | 0 |
| no RMSNorm | \(10^{-3}\) | 22,691,840 | 1.375150 | 1.382416 | 3.9845 | 772.429 s | +0.012052 |
| no RMSNorm（低 LR） | \(10^{-4}\) | 22,691,840 | 1.835701 | 1.818733 | 6.1640 | 792.161 s | +0.448369 |
| Post-Norm | \(10^{-3}\) | 22,696,448 | 1.356367 | **1.362665** | 3.9066 | 825.945 s | **-0.007699** |
| NoPE | \(10^{-3}\) | 22,696,448 | 1.430324 | 1.432178 | 4.1878 | 816.840 s | +0.061814 |
| matched SiLU | \(10^{-3}\) | 22,827,520 | 1.381219 | 1.387026 | 4.0029 | 822.962 s | +0.016661 |

matched SiLU 比 baseline 多 131,072 个参数（0.58%），仍可视为近似匹配。同 LR 下
no-RMSNorm 能稳定训练但略差；把它降到 \(10^{-4}\) 反而严重欠优化。NoPE 退化最大，
表明位置编码有实质贡献；参数近似匹配的 SiLU 略差于 SwiGLU。Post-Norm 在本次单 seed、
小模型实验中优 0.0077，差距较小，不将它外推为一般结论。曲线见
[step 横轴](assets/ablations_loss_by_step.svg) 和
[wall-clock 横轴](assets/ablations_loss_by_wall_clock.svg)。

### OpenWebText

OWT 使用与 TinyStories baseline 相同的 \(S,d,L,h,f\)、batch size 和 10,000 个 iterations；
tokenizer/vocabulary 改为 32K OWT 版本，并使用该数据的峰值 LR \(3\times10^{-4}\)。

| run | steps | processed tokens | parameters | final train loss | final/best val loss | PPL | wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OWT baseline | 10,000 | 327,680,000 | 45,224,448 | 4.290242 | 4.296564 | 73.4470 | 1,157.660 s |

OWT validation loss 从 10.385933 持续降到 4.296564，未见过拟合。开放域语料的词汇、
文体和主题更多样，且 32K 词表使模型参数量增至 45.2M；在相同 327.68M tokens 下明显
比 TinyStories 更难。两者 tokenizer 不同，因此 raw loss/PPL 不宜作严格横向质量比较。曲线见
[step 横轴](assets/owt_loss_by_step.svg) 和
[wall-clock 横轴](assets/owt_loss_by_wall_clock.svg)。

### 文本生成

四份生成均使用 best checkpoint 和 seed 42；TinyStories 的 prompt 为
“Once upon a time”，OWT 的 prompt 为 “The history of science”。

| 语料 | temperature | top-p | 新 token | 停在 EOS |
| --- | ---: | ---: | ---: | --- |
| TinyStories 默认 | 0.8 | 0.95 | 117/256 | 是 |
| TinyStories 低温 | 0.6 | 0.95 | 110/256 | 是 |
| TinyStories 较低 top-p | 0.8 | 0.80 | 163/256 | 是 |
| OWT 默认 | 0.8 | 0.95 | 256/256 | 否 |

TinyStories 默认样例：

> Once upon a time, there was a big, heavy rock. The rock was sad because it was so heavy. It needed to be fixed.
> A little bird lived in the rock. The bird could not fly very well. The bird wanted to help the rock. The bird had an idea. It would find a new battery for the rock.
> The bird looked for a new battery. It found one in the ground. The bird gave the new battery to the rock. The rock was happy. The bird and the rock became good friends. They played together in the sky every day.
> `<|endoftext|>`

默认样例有完整的“开端—问题—解决”结构并正确输出 EOS，语法总体流畅，但“鸟住在岩石里/
给岩石换电池/一起在天上玩”有语义跳跃。降温到 0.6 后，样例更模板化、局部更连贯，
但出现 `rock` 高频重复与“rock 想和 rock 交朋友”。保持温度 0.8 而把 top-p 降到 0.80 时，
本 seed 产生了更长的故事和“岩石原来是乌龟”的反转，但后续又把岩石与乌龟写成两个实体，
角色一致性仍不完美。单个 seed 只能说明本次样例，不把长度差异外推为采样参数的普遍规律。

OWT 样例：

> The history of science is that science is a revolutionary that would change the topic of the book. And it has yet to go, and so on. It is the latter in a paper I have here at the University of Edinburgh and there are many more and more people that have even read a book about the paper. So the two parts of the book are, but it is the bit and the new book is coming out. And so I’ll have to find out about these different things.
>
> But even if it’s not in the writing the book itself, it’s a good way to do it. It’s a wonderful blend of theories and philosophers to remember that there are no plans to explain the story.
>
> But let’s not forget: I think there’s a lot of evidence to answer that on the one hand, and we’re the one on the other. To me, the paper is, to think, a few of the books, where a lot of the books will show that you know the book is the one that the book is going to be about:
>
> I’m going to write this piece as a general magazine, the book about what I read from the book about the book, and I think that I think it’

OWT 样例具有博客/评论文体的句法外观，但 `book`/`paper`/`I think` 循环明显，到 256-token
上限仍未 EOS 并在句中截断，与开放域任务更高的 validation loss 一致。完整结构化记录见
[TinyStories 默认](logs/generation/generations/tinystories.json)、
[低温对照](logs/generation/generations/tinystories_t06_p095.json)、
[较低 top-p 对照](logs/generation/generations/tinystories_t08_p080.json) 和
[OWT](logs/generation/generations/owt.json)。

## 复现说明

以下命令均从公开的 assignment1-basics 仓库根目录执行。环境要求为 Python 3.12–3.13；
依赖由仓库的 uv.lock 固定。

### 环境与测试

~~~bash
mkdir -p .runtime/uv-cache
export UV_CACHE_DIR="$PWD/.runtime/uv-cache"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"
uv sync --frozen
uv run pytest
~~~

上述变量把 uv cache 和虚拟环境显式放在当前仓库；训练脚本也会把编译 cache 与临时文件定向到
.runtime/，无需依赖用户主目录。
本次最终运行结果为 47 passed、1 xpassed。

### 公开数据下载

~~~bash
mkdir -p data
wget -P data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget -P data https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget -P data https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
wget -P data https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gzip -d data/owt_train.txt.gz
gzip -d data/owt_valid.txt.gz
~~~

### 训练 tokenizer 与测量指标

~~~bash
uv run python scripts/train_tokenizer.py \
  --corpus data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --output-dir tokenizer_artifacts/tinystories \
  --special-token '<|endoftext|>' \
  --num-processes 48

uv run python scripts/train_tokenizer.py \
  --corpus data/owt_train.txt \
  --vocab-size 32000 \
  --output-dir tokenizer_artifacts/owt \
  --special-token '<|endoftext|>' \
  --num-processes 48

uv run python scripts/tokenizer_metrics.py \
  --corpus data/TinyStoriesV2-GPT4-valid.txt \
  --vocab tokenizer_artifacts/tinystories/vocab.json \
  --merges tokenizer_artifacts/tinystories/merges.txt \
  --output tokenizer_artifacts/tinystories/metrics_tinystories_valid.json

uv run python scripts/tokenizer_metrics.py \
  --corpus data/owt_valid.txt \
  --vocab tokenizer_artifacts/tinystories/vocab.json \
  --merges tokenizer_artifacts/tinystories/merges.txt \
  --output tokenizer_artifacts/tinystories/metrics_owt_valid.json

uv run python scripts/tokenizer_metrics.py \
  --corpus data/TinyStoriesV2-GPT4-valid.txt \
  --vocab tokenizer_artifacts/owt/vocab.json \
  --merges tokenizer_artifacts/owt/merges.txt \
  --output tokenizer_artifacts/owt/metrics_tinystories_valid.json

uv run python scripts/tokenizer_metrics.py \
  --corpus data/owt_valid.txt \
  --vocab tokenizer_artifacts/owt/vocab.json \
  --merges tokenizer_artifacts/owt/merges.txt \
  --output tokenizer_artifacts/owt/metrics_owt_valid.json
~~~

四项测量都使用完整验证集、4 MiB chunk 和单个编码进程，保证 compression 与 throughput
口径一致。

### 编码语料

~~~bash
uv run python scripts/encode_dataset.py \
  --corpus data/TinyStoriesV2-GPT4-train.txt \
  --vocab tokenizer_artifacts/tinystories/vocab.json \
  --merges tokenizer_artifacts/tinystories/merges.txt \
  --output data/tinystories_train.bin --dtype uint16 --num-processes 48

uv run python scripts/encode_dataset.py \
  --corpus data/TinyStoriesV2-GPT4-valid.txt \
  --vocab tokenizer_artifacts/tinystories/vocab.json \
  --merges tokenizer_artifacts/tinystories/merges.txt \
  --output data/tinystories_valid.bin --dtype uint16 --num-processes 48

uv run python scripts/encode_dataset.py \
  --corpus data/owt_train.txt \
  --vocab tokenizer_artifacts/owt/vocab.json \
  --merges tokenizer_artifacts/owt/merges.txt \
  --output data/owt_train.bin --dtype uint16 --num-processes 48

uv run python scripts/encode_dataset.py \
  --corpus data/owt_valid.txt \
  --vocab tokenizer_artifacts/owt/vocab.json \
  --merges tokenizer_artifacts/owt/merges.txt \
  --output data/owt_valid.bin --dtype uint16 --num-processes 48
~~~

### 训练、扫参与生成

~~~bash
# 单独运行 TinyStories baseline
uv run python scripts/train_lm.py \
  --config configs/tinystories_baseline.json --overwrite

# 自动使用所有可见 GPU，一张卡运行一个独立实验
uv run python scripts/run_experiment_suite.py \
  --suite-config configs/experiment_suite.json

# 从 best checkpoint 生成不少于 256 个 token 的候选样例
uv run python scripts/generate.py \
  --config configs/tinystories_baseline.json \
  --checkpoint runs/tinystories_baseline/best.pt \
  --vocab tokenizer_artifacts/tinystories/vocab.json \
  --merges tokenizer_artifacts/tinystories/merges.txt \
  --prompt 'Once upon a time' \
  --max-new-tokens 256 --temperature 0.8 --top-p 0.95 \
  --output runs/generations/tinystories.json

uv run python scripts/generate.py \
  --config configs/owt_baseline.json \
  --checkpoint runs/owt_baseline/best.pt \
  --vocab tokenizer_artifacts/owt/vocab.json \
  --merges tokenizer_artifacts/owt/merges.txt \
  --prompt 'The history of science' \
  --max-new-tokens 256 --temperature 0.8 --top-p 0.95 \
  --output runs/generations/owt.json
~~~

从 SummerQuest-2026 根目录同步公开提交：

~~~bash
python3 scripts/sync_a1_submission.py --name '左景萱'
~~~

- 配置文件：submission/configs/tinystories_baseline.json、
  submission/configs/owt_baseline.json、submission/configs/experiment_suite.json
- 真实实现：submission/cs336_basics/
- 测试 adapter：submission/tests/adapters.py
- 训练、编码与生成入口：submission/scripts/

## 实验日志

- 日志目录：[`logs/`](logs/)；索引为 [`logs/summary.json`](logs/summary.json)，自动验收结果为
  [`logs/validation.json`](logs/validation.json)。
- tokenizer 训练、compression/throughput、训练逐点 JSONL、run summary、batch benchmark 与
  生成记录分别对应上方各实验结果小节。
- 训练日志的核心字段为 step、wall_clock_sec、train_loss、lr 和周期性 val_loss；summary
  同时记录最终/最佳 validation loss、总时间、processed tokens 和完整模型/优化配置。
- checkpoint、编码后的数据和依赖环境不复制到公开提交。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/HIludi31voqCeYxtsHMcvJUynLd
- 权限：组织内获得链接的成员可阅读，已关闭组织外访问与邀请，未开启互联网公开。

补充文档应设置为组织内公开且不允许互联网公开，只保存不能进入 GitHub 但确有审核必要的
最小差量材料；密钥、cookie 和访问凭据在任何情况下都不写入文档或仓库。
