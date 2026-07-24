# A1 公开提交：王扬


## 基本信息

- 作业题面版本：26.0.4
- 完成范围：完成 `submission/cs336_basics/` 中全部 21 个公共 ABI 对应实现；通过公共测试；补齐 tokenizer 训练、语料编码、训练、生成脚本；完成 TinyStories baseline、learning rate sweep、batch size sweep、四个 ablation、OWT baseline 的日志产出与生成样例；补充 `ts_lr_1e-2_diverge`、`ts_lr_1e-1_diverge`、`ts_lr_1e0_diverge`、`ts_lr_1e1_diverge` 等高学习率实验，其中 `ts_lr_1e0_diverge` 与 `ts_lr_1e1_diverge` 可作为“至少一个发散 run”已完成的证据。
- 上游 starter commit：`a158843b20107949f1a8d7df1b05cd33b9166712`
- 本地工作仓库：`../assignment1-basics`（必须与 `SummerQuest-2026` 同级）

## Markdown 报告

### 实现概览

本次实现将公共测试要求的 21 个接口全部沉到 `cs336_basics/` 下，`tests/adapters.py` 仅保留桥接职责。主要模块包括：

- `nn.py`：`Linear`、`Embedding`、`RMSNorm`、`SwiGLU`、`SiLUFeedForward`、`silu`、`softmax`、`cross_entropy`
- `attention.py`：`scaled_dot_product_attention`、`RoPE`、`MultiHeadSelfAttention`
- `transformer.py`：`TransformerBlock`、`TransformerLM`，并支持 `use_rope`、`use_rmsnorm`、`prenorm`、`ffn_type`
- `data.py` / `optim.py` / `serialization.py`：batch 采样、AdamW、梯度裁剪、学习率调度、checkpoint
- `tokenizer.py`：byte-level BPE 训练、`encode/decode/encode_iterable`；pretokenize 阶段支持可配置 `num_workers` 多进程加速

公共测试在工作仓库中已通过：

```text
uv run pytest
=> 48 passed
```

### Unicode 书面题

#### unicode1

我把 Unicode 和 UTF-8 区分为三个层次来理解：

- 字符：人眼看到的文本符号，例如 `"牛"`
- Unicode code point：字符的抽象编号，例如 `"牛"` 对应 `U+725B`
- UTF-8 bytes：把 code point 编码成可存储/传输的字节序列，例如 `"牛"` 的 UTF-8 是 `E7 89 9B`

因此，一个“字符”不等于一个“字节”。在 Python 中：

```python
text = "牛"
list(text.encode("utf-8"))  # [231, 137, 155]
```

`len(text) == 1`，但 `len(text.encode("utf-8")) == 3`。这说明 UTF-8 是对 Unicode 的一种可变长编码方案，而不是“一个字符固定一个字节”。

#### unicode2

byte-level tokenizer 直接在 UTF-8 bytes 上工作，所以词表的最小原子一定覆盖 `0..255` 这 256 个 byte 值，因此不会出现 OOV。代价是纯 byte 序列通常很长，所以需要 BPE 把高频相邻 byte/token pair 反复 merge 成更长 token，以换取更短的 token 序列。

这也解释了为什么 decode 时不能逐 token 独立做 UTF-8 解码：因为一个 Unicode 字符可能跨多个 byte，甚至跨多个 token。正确做法是先把所有 token 对应的 bytes 拼接起来，再整体执行 UTF-8 decode；如果出现非法 UTF-8，则替换为 `U+FFFD`。

### Tokenizer 对比

两个 tokenizer 的统计都来自各自 `artifacts/tokenizer_*/stats.json`，结果如下：

| Tokenizer | vocab size | merges | train_wall_clock_sec | compression ratio (bytes/token) | longest token bytes | longest token example | encode throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| TinyStories | 10000 | 9743 | 413.90 | 4.1154 | 15 | `" accomplishment"` | 2860326.94 |
| OWT | 32000 | 31743 | 11120.82 | 4.1884 | 64 | 重复字节序列，UTF-8 解码后表现为 `ÃÂÃÂ...` | 2760080.58 |

对比结论：

- OWT tokenizer 的 compression ratio 略高于 TinyStories，说明更大词表和更开放的数据分布确实带来了更强的压缩能力。
- OWT tokenizer 学到的最长 token 明显更长，且包含非自然语言的噪声 byte 模式，这与开放域网页文本的数据特征一致。
- 两者 encode throughput 接近，TinyStories 略快；说明当前 `tiktoken.Encoding` 封装后的编码开销主要取决于实现路径，而不是只取决于词表大小。

### AdamW 显存 / FLOPs / 训练时间核算

以下核算基于当前实现与配置，采用的简化假设是：

- 参数、梯度、AdamW 的 `m` / `v` 都按 `float32` 计，即每个标量 `4 bytes`
- AdamW 状态显存只统计 `m` 与 `v`，不计很小的 step 标量
- 训练 FLOPs 采用常见近似：`training step ~= 3 x forward FLOPs`
- forward FLOPs 只保留主导项：attention 线性层、FFN、attention matrix 与 `lm_head`，忽略 norm / softmax / embedding lookup / optimizer 标量操作

TinyStories baseline 参数量可按下式拆开：

- token embedding：`10000 x 512 = 5,120,000`
- 单层 attention：`4 x 512 x 512 = 1,048,576`
- 单层 SwiGLU FFN：`512 x 1344 + 1344 x 512 + 512 x 1344 = 2,064,384`
- 单层 RMSNorm：`2 x 512 = 1,024`
- 单层 block 合计：`3,113,984`
- 4 层 block：`12,455,936`
- final RMSNorm：`512`
- lm head：`512 x 10000 = 5,120,000`
- 总参数量：`22,696,448`

对应的 AdamW 与训练核算如下：

| 模型 | 参数量 | AdamW 状态显存 (`m`+`v`) | 参数+梯度+AdamW 总显存 | 单步训练 FLOPs 估算 | 10000 steps 总 FLOPs | 训练时间 | token throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TinyStories baseline | 22,696,448 | 173.16 MiB | 346.32 MiB | 3.661 TFLOPs | 36.609 PFLOPs | 1502.76 s | 218052 tokens/s |
| OWT baseline | 45,224,448 | 345.04 MiB | 690.07 MiB | 2.938 TFLOPs | 29.378 PFLOPs | 1145.90 s | 142979 tokens/s |

补充说明：

- OWT 的参数量更大，是因为 `vocab_size=32000` 使得 embedding 与 `lm_head` 参数翻倍以上。
- TinyStories 虽然参数更少，但 `batch_size=128`，因此单步 FLOPs 反而高于 OWT 的 `batch_size=64` 配置。
- 从日志反推，TinyStories run 处理了 `10000 x 128 x 256 = 327,680,000` 个训练 token；OWT run 处理了 `163,840,000` 个训练 token。

### TinyStories Baseline

训练配置与题面推荐 baseline 保持一致：

- `vocab_size=10000`
- `context_length=256`
- `d_model=512`
- `num_layers=4`
- `num_heads=16`
- `d_ff=1344`
- `rope_theta=10000`
- `batch_size=128`
- `max_steps=10000`

最终结果来自 `logs/train_tinystories.jsonl`：

| 实验 | steps | wall_clock_sec | final train_loss | final val_loss |
| --- | ---: | ---: | ---: | ---: |
| TinyStories baseline | 10000 | 1502.76 | 1.4291 | 1.4366 |

这个结果已经达到题面中 TinyStories baseline `val_loss <= 1.45` 的目标。

对应 loss 曲线见 [tinystories_loss.svg](assets/tinystories_loss.svg)。从曲线看，train loss 与定期评估的 val loss 都整体平稳下降，训练后期没有明显震荡或退化。

对应生成样例位于 `logs/generation_tinystories.txt`。从样例文本看，模型已经能生成较完整的儿童故事结构，叙事连贯性基本成立，但局部指代和角色关系仍有混乱。

### Learning Rate Sweep

learning rate sweep 统一训练 2000 steps，结果如下：

| 配置 | final val_loss | 说明 |
| --- | ---: | --- |
| `ts_lr_1e-4` | 2.4423 | 明显欠训练 |
| `ts_lr_5e-4` | 1.7745 | 可稳定收敛 |
| `ts_lr_1e-3` | 1.6371 | 本组最优 |
| `ts_lr_3e-3_diverge` | 1.5471 | 高学习率压力测试，但在 2000 steps 内仍可训练 |
| `ts_lr_1e-2_diverge` | 2.5022 | 已明显退化，但未完全失稳 |
| `ts_lr_1e-1_diverge` | 3.6936 | 明显失稳，验证损失大幅升高 |
| `ts_lr_1e0_diverge` | 5.8546 | 发散，训练后期 loss 长时间处于异常高位 |
| `ts_lr_1e1_diverge` | 16.2465 | 强发散，前几步 train loss 即飙升到万级 |

从当前日志看，`1e-4` 学习太慢；`5e-4` 到 `1e-3` 区间表现较好；当学习率提高到 `1e-2` 及以上时，验证损失开始明显恶化，而 `1e0` / `1e1` 两组已经表现出清晰的发散特征。虽然日志中未出现 `NaN/Inf`，但其 loss 在训练早期急剧放大且最终验证损失远高于正常收敛区间，足以作为“至少一个发散 run”的提交证据。

### Batch Size Sweep

batch size sweep 统一训练 2000 steps：

| 配置 | final val_loss | wall_clock_sec |
| --- | ---: | ---: |
| `ts_bs_1` | 3.1238 | 27.04 |
| `ts_bs_64` | 1.8418 | 145.20 |
| `ts_bs_128` | 1.7745 | 384.21 |

结果显示极小 batch size 会显著恶化收敛质量；`64` 与 `128` 都能正常训练，其中 `128` 的验证损失更低，但单次 run 的墙钟时间也更长。

### Ablations

四个消融实验均在 TinyStories 上训练 2000 steps：

| 配置 | final val_loss | 结论 |
| --- | ---: | --- |
| baseline (`ts_lr_5e-4` / `bs=128`) | 1.7745 | 对照组 |
| `ts_no_rmsnorm` | 1.7460 | 本次短程实验中未明显劣化 |
| `ts_postnorm` | 1.7792 | 与 baseline 接近，略差 |
| `ts_nope` | 1.8584 | 去掉 RoPE 后性能下降 |
| `ts_silu_ffn` | 1.8900 | 改成 SiLU FFN 后性能下降更明显 |

从短程实验结果看，位置编码和 FFN 结构对性能影响更明显；`NoPE` 和 `SiLU FFN` 都比 baseline 更差。`No RMSNorm` 在当前 2000-step 设定下没有直接崩掉，说明短程稳定性和长程最终效果并不完全一致。

### OWT Baseline

OWT baseline 使用与 TinyStories 相同的模型骨架，但：

- `vocab_size=32000`
- `batch_size=64`
- `max_steps=10000`

最终结果来自 `logs/train_owt.jsonl`：

| 实验 | steps | wall_clock_sec | final train_loss | final val_loss |
| --- | ---: | ---: | ---: | ---: |
| OWT baseline | 10000 | 1145.90 | 4.1390 | 4.2103 |

对应 loss 曲线见 [owt_loss.svg](assets/owt_loss.svg)。从曲线看，训练过程本身是稳定的，但收敛到的 loss 明显高于 TinyStories，说明主要问题更像是欠拟合而不是训练直接跑坏。

对应生成样例位于 `logs/generation_owt.txt`，质量明显差于 TinyStories，当前结果可以视为“训练流程跑通，但模型尚未学到足够强的 OWT 语言建模能力”。

### OWT 结果分析

本次 OWT 表现不佳，原因判断如下：


- 主要的问题是任务难度显著提高：OWT 使用 `32000` 词表、开放域网页语料、同样只有 `4 x 512` 的较小模型、`10000` steps 的训练预算，相比 TinyStories 更容易欠拟合。
- 从日志看，OWT 训练并没有出现 NaN/Inf，说明不是典型数值爆炸，而是“训练跑通但效果差”。

因此，当前更合理的结论是：OWT baseline 的主要瓶颈来自**模型规模/训练步数/数据复杂度不匹配**，而不是 pretokenize 并行本身。

## 复现说明

- 环境与依赖：Linux + Python 3.12；依赖通过 `uv` 管理。在工作仓库根目录运行 `uv sync --frozen --no-install-project`，执行命令时统一使用 `uv run --no-sync ...`。
- 数据准备：按 `assignments/A1/README.md` 中的公开方式准备 TinyStories 与 OWT 文本语料，放在工作仓库 `../assignment1-basics/data/` 下；不在公开提交中包含原始数据、tokenized 数据、checkpoint 或虚拟环境。
- Tokenizer、训练与生成命令：

```bash
# 1. 公共测试
bash scripts/tasks/run_a1.sh test

# 2. TinyStories tokenizer / 编码 / 训练 / 生成
bash scripts/tasks/run_a1.sh tokenizer_tinystories
bash scripts/tasks/run_a1.sh encode_tinystories
bash scripts/tasks/run_a1.sh train_tinystories
bash scripts/tasks/run_a1.sh generate_tinystories

# 3. TinyStories sweeps / ablations
bash scripts/tasks/run_a1.sh lr_sweep
bash scripts/tasks/run_a1.sh batch_sweep
bash scripts/tasks/run_a1.sh ablations

# 4. OWT tokenizer / 编码 / 训练 / 生成
bash scripts/tasks/run_a1.sh tokenizer_owt
bash scripts/tasks/run_a1.sh encode_owt
bash scripts/tasks/run_a1.sh train_owt
bash scripts/tasks/run_a1.sh generate_owt

# 5. 一键全流程
bash scripts/tasks/all.sh
```

- 同步命令：`python3 scripts/sync_a1_submission.py --name '王扬'`
- 配置文件：`submission/configs/tokenizer_tinystories.json`、`submission/configs/tokenizer_owt.json`、`submission/configs/train_tinystories.json`、`submission/configs/train_owt.json`、`submission/configs/lr_sweep/*.json`、`submission/configs/batch_size/*.json`、`submission/configs/ablations/*.json`

## 代码与脚本

- 真实实现：`submission/cs336_basics/`
- 测试 adapter：`submission/tests/adapters.py`
- 训练、数据编码与生成脚本：`submission/scripts/`
- 实现说明：
  - `submission/cs336_basics/` 提供全部 from-scratch 核心实现，包括 Transformer、Tokenizer、优化器、数据采样与 checkpoint。
  - `submission/tests/adapters.py` 保持上游 21 个 adapter 的函数名和签名，只做桥接，不承载真实逻辑。
  - `submission/scripts/` 包含 tokenizer 训练、语料编码、训练、生成，以及 `scripts/tasks/` 下的任务脚本。
  - 训练脚本支持 `auto/cuda/cpu` 设备选择；当 CUDA 驱动与当前 PyTorch 构建不兼容时，会自动回退到 CPU。
  - tokenizer 训练阶段加入了 `tqdm` 进度条；TinyStories/OWT tokenizer 配置支持 `num_workers`，仅对 pretokenize 阶段做多进程加速，merge 主循环仍保持串行。

真实实现先在兄弟目录 `../assignment1-basics` 中完成并通过官方测试，再使用同步命令复制
到本目录。不要手工复制公共 tests、fixtures、数据、模型权重、虚拟环境或依赖锁。

## 实验日志

- 日志目录：`logs/`
- 曲线图目录：`assets/`
- 文件与格式：见 [`assignments/A1/README.md` 的《实验日志格式》](../../../assignments/A1/README.md#实验日志格式)
- 与报告中实验的对应说明：
  - `logs/train_tinystories.jsonl`：TinyStories baseline 10000-step 训练日志
  - `logs/train_owt.jsonl`：OWT baseline 10000-step 训练日志
  - `logs/lr_sweep/*.jsonl`：learning rate sweep 对应的 8 组实验
  - `logs/batch_size/*.jsonl`：batch size sweep 对应的 3 组实验
  - `logs/ablations/*.jsonl`：四个 ablation 实验
  - `logs/generation_tinystories.txt`：TinyStories 生成样例
  - `logs/generation_owt.txt`：OWT 生成样例
  - `assets/tinystories_loss.svg` / `assets/owt_loss.svg`：TinyStories 与 OWT 的 loss 曲线
  - 所有训练日志均按 step 记录 `wall_clock_sec`、`train_loss`、`lr`，并定期记录 `val_loss`；本 README 中的数值均直接从这些日志末行提取。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/WgCDdzATZoI1bqxhnpGcuLssn8f?from=from_copylink

该文档设置为组织内公开，不得开启互联网公开访问，只保存不能公开到 GitHub 但确有
审核必要的最小差量材料。
