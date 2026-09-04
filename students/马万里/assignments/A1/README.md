# A1 公开提交：马万里

## 基本信息

- 作业题面版本：26.0.4
- 完成范围：Byte-level BPE tokenizer（训练、编码、解码、流式）、Transformer LM 全部模块（Linear、Embedding、RMSNorm、SwiGLU、RoPE、Attention、MHA、Block、LM）、训练组件（cross-entropy、AdamW、cosine schedule、gradient clipping、data loader、checkpoint）、TinyStories baseline 训练、学习率扫描、batch size 扫描、四个架构消融（删除 RMSNorm、Post-Norm、NoPE、SiLU 替代 SwiGLU）、文本生成。
- 未完成项：OpenWebText（OWT）tokenizer 训练与模型训练
- 上游 starter commit：`a158843b20107949f1a8d7df1b05cd33b9166712`
- 本地工作仓库：`../assignment1-basics`（必须与 `SummerQuest-2026` 同级）

## Markdown 报告

### 1. 书面题

#### 1.1 Unicode 与 UTF-8

**unicode1**：
Unicode 是字符到抽象编号（码点）的映射，例如汉字“牛”的码点是 U+725B。UTF-8 是一种可变长度编码方式，将码点编码为 1～4 个字节。在 Python 中，一个字符可能对应多个 UTF-8 字节，例如汉字“牛”编码后占用 3 个字节，而字符本身的长度仍为 1。这种差异是理解 byte-level tokenizer 的基础。

**unicode2**：
byte-level tokenizer 以 UTF-8 字节（0～255）为最小单位，所有 Unicode 文本最终都表示为这 256 种字节的组合，因此不会出现未登录词（OOV）。代价是序列可能较长，需要 BPE 来压缩。

#### 1.2 AdamW 显存、FLOPs 与训练时间核算

**模型配置**：`d_model=512`, `num_layers=4`, `num_heads=16`, `d_ff=1344`, `vocab_size=10000`, `context_length=256`, `batch_size=128`, `total_steps=10000`。

**参数量**：

- Token Embedding：
  $10000 \times 512 = 5{,}120{,}000$

- LM Head（输出投影）：
  $512 \times 10000 = 5{,}120{,}000$

- 每个 Transformer Block：
  - Attention（Q、K、V、O 四个矩阵）：
    $4 \times (512 \times 512) = 1{,}048{,}576$
  - FFN（SwiGLU）：
    $W_1: 1344 \times 512 = 688{,}128$
    $W_2: 512 \times 1344 = 688{,}128$
    $W_3: 1344 \times 512 = 688{,}128$
    小计：$2{,}064{,}384$
  - RMSNorm ×2：
    $2 \times 512 = 1{,}024$
  - 每层总计：
    $1{,}048{,}576 + 2{,}064{,}384 + 1{,}024 = 3{,}113{,}984 \approx 3.11M$

- 4 层总参数量：
  $4 \times 3{,}113{,}984 = 12{,}455{,}936$

- 总参数量：
  $5{,}120{,}000 \ (\text{Embedding}) + 12{,}455{,}936 \ (\text{4个Block}) + 5{,}120{,}000 \ (\text{LM Head}) = 22{,}695{,}936 \approx 22.7M$

**显存估算（fp32）**：

- 模型权重：
  $22.7M \times 4\ \text{bytes} \approx 90.8\ \text{MB}$

- AdamW 优化器状态（一阶矩 `m` + 二阶矩 `v`）：
  $2 \times 22.7M \times 4\ \text{bytes} \approx 181.6\ \text{MB}$

- 梯度：
  $22.7M \times 4\ \text{bytes} \approx 90.8\ \text{MB}$

- 激活显存（主要开销）：
  - 注意力得分矩阵：形状 $(B, H, T, T)$，fp32 下每个 token 的注意力分数占用 $H \times T \times 4$ 字节，总大小为 $B \times H \times T \times T \times 4$ 字节。
    - batch size=128 时：$128 \times 16 \times 256 \times 256 \times 4 \approx 536\ \text{MB}$
  - Q/K/V 激活：每个约为 $B \times T \times d_{model} \times 4$ 字节，batch size=128 时约 67 MB，三个约 201 MB。
  - FFN 中间激活：约 $B \times T \times d_{ff} \times 4$ 字节，batch size=128 时约 176 MB。
  - 每层总激活约 1～2 GB，4 层总激活在 batch size=128 时约 **5～8 GB**

- **总显存占用**：
  batch size=128 时约 **6～9 GB**，在 24 GB 的 RTX 4090 上运行正常。

**FLOPs 估算**：

一个 step 处理的 token 数：
$128 \times 256 = 32{,}768$

一次前向 FLOPs（参数法）：
$2 \times 22.7M \times 32{,}768 \approx 1.49 \times 10^{12}\ \text{FLOPs} = 1.49\ \text{TFLOPs}$

反向约为前向的 2 倍，因此每个 step 总 FLOPs：
$1.49 \times 3 \approx 4.47\ \text{TFLOPs}$

10000 步总 FLOPs：
$4.47 \times 10^{12} \times 10{,}000 = 4.47 \times 10^{16}\ \text{FLOPs} = 44.7\ \text{PFLOPs}$

**训练时间**：
实测训练在单张 RTX 4090 上完成，10000 步耗时约 1973 秒（约 33 分钟）。实际有效算力约 $44.7 \times 10^{15} / 1973 \approx 22.7\ \text{TFLOPS}$，约为 RTX 4090 FP32 理论峰值（82.6 TFLOPS）的 27%。

### 2. Tokenizer 实验

**TinyStories 10K 词表**

根据 `logs/tokenizer_tinystories.jsonl` 中的记录：

| 指标 | 数值 |
|------|------|
| 训练耗时 | 854.28 秒（约 14.2 分钟） |
| 峰值内存 | 2396.45 MB（约 2.34 GB） |
| 最长 token | `" accomplishment"`（15 字节） |
| 验证集压缩率 | 4.1169 bytes/token |
| 验证集吞吐量 | 165,856.77 tokens/s |
| 训练集压缩率 | 4.1161 bytes/token |
| 训练集吞吐量 | 161,560.49 tokens/s |

**分析**：
- 训练时间远低于 30 分钟限制，内存占用约 2.34 GB，远低于 30 GB 限制，说明 BPE 实现高效。
- 最长 token 为 `" accomplishment"`（包含一个空格前缀），长度为 15 字节，是一个合理的常见子词。
- 压缩率约 4.12 bytes/token，意味着序列长度被压缩了约 4 倍，有利于模型训练。
- 编码吞吐量超过 16 万 tokens/s，能够高效处理大规模数据。

（注：OWT 实验未完成，因此不提供 OWT 对比。）

### 3. TinyStories 训练

- 最终验证 loss：1.3785（第 9500 步记录，总步数 10000，最终可能略低）
- 总训练时间：约 1973 秒（约 33 分钟）

训练达到并超过了目标（val_loss ≤ 1.45），说明模型架构和超参数选择合理。

### 4. 学习率扫描

对学习率 `{1e-4, 1e-3, 1e-2, 1e-1, 10}` 进行扫描，每个训练 3000 步，batch size 128。结果：

| 学习率 | 最终 val loss | 是否发散 |
|--------|--------------|---------|
| 1e-4   | 2.00         | 否 |
| 1e-3   | 1.56         | 否 |
| 1e-2   | 2.53         | 否 |
| 1e-1   | 3.62         | 否 |
| 10     | 5.70         | 是 |

**分析**：学习率 `1e-2`、`1e-1` 和 `10` 过高，训练收敛慢，尤其学习率为 `10` 时会导致训练发散；`1e-3` 表现最好，验证了基线学习率的合理性。较低学习率如 `1e-4` 收敛较慢。

### 5. Batch size 扫描

对 batch size `{32, 64, 128, 256}` 进行扫描，固定学习率 1e-3，训练 3000 步。结果：

| Batch size | 最终 val loss | 训练时间/s |
|------------|--------------|---------|
| 32         | 1.75         | 357.50 |
| 64         | 1.62         | 588.82 |
| 128        | 1.53         | 1338.82 |
| 256        | OOM（显存不足）| N/A |

**分析**：batch size 从 32 到 128 时，训练较为稳定；batch size 256 因注意力矩阵显存过大导致 CUDA OOM。根据数据可以分析出通常来说 batch size 越大，训练效果越好，但同时所需的训练时间也越多。

### 6. 四个架构消融

所有消融实验与 baseline 相同的训练步数（5000 步）和学习率 1e-3，仅改变模型的单个模块。结果：

| 消融 | 最终 val loss | 与 baseline 差异 | 分析 |
|------|--------------|----------------|------|
| Baseline (Pre-Norm + RoPE + SwiGLU) | 1.47 | - | 基准 |
| 删除 RMSNorm | 1.70 | 较差 | RMSNorm 对向量进行归一化操作，防止梯度爆炸/消失，同时加快收敛，模型性能得到提升。 |
| Post-Norm | 1.44 | 稍好 | Post-Norm 是原版 Transformer 中的架构，相较于 Pre-Norm 对参数正则化的效果更强，进而模型的鲁棒性也会更好。而 Pre-Norm 因为有一部分参数直接加在了后面，不需要对这部分参数进行正则化，可以防止模型的梯度爆炸或者梯度消失。在我们这个模型中，只堆叠了 4 个 Transformer Block，并不会出现梯度问题，因此 Post-Norm 的表现会更好。 |
| NoPE | 1.52 | 稍差 | NoPE 是无位置编码，不会给模型提供 token 的信息位置。RoPE 是旋转位置编码，会给 token 标注位置信息，模型可以学习到位置信息进行更好的训练拟合。但是这二者各有优劣，RoPE 在短上下文中表现得不错，但是在长上下文中的效果较差。而 NoPE 在短上下文表现较差，在长上下文中表现不错。本次训练的文本为 TinyStories，属于短上下文，因此 RoPE 的表现较好，如果是在 OWT 上进行训练，可能 NoPE 会表现得更好。但是由于各种原因暂时未做 OWT 相关实验。 |
| SiLU 替代 SwiGLU | 1.47 | 相近 | 理论上来说在相同的参数量的情况下 SwiGLU 的表现应该比 SiLU 好，推断由于 TinyStories 只有 10K 词表，并且模型本身的参数量足以完成任务，因此不同的激活模型的效果被模型本身的能力稀释了，不能体现两种激活参数的差异。 |

### 7. 文本生成样本

使用 `scripts/generate.py` 生成，参数：temperature=1.0, top_p=0.9, max_tokens=256。

```txt
Once upon a time, there was a little boy named Tim. Tim was a good boy who liked to play outside. One day, he found a big door in his house. He wanted to open it, but he did not know how.
Tim asked his mom, "Mom, can you help me open the door?" His mom said, "Yes, Tim. I will help you." They opened the door together. Inside, they found a room full of old clothes. Tim was excited to try them on.
Tim and his mom put on the clothes and opened the door. Inside, they found a room full of old clothes. They put on the clothes and felt very lucky. They walked back home, feeling happy and loved. Tim knew that the attic was a special place for him to play and have fun.
<|endoftext|>
```

### 8. 实现说明

- **BPE 训练**：采用唯一 pre-token 聚合策略，以频率加权方式更新 pair 统计；使用堆（`heapq`）和自定义比较器（`_ReversePair`）实现并列 tie-break；特殊 token 作为硬边界不参与合并。预分词采用 GPT-2 正则。
- **Transformer 模型**：所有组件 from scratch，未使用 `torch.nn.Linear`、`nn.Embedding` 等现成层。`TransformerLM` 支持通过 `block_type` 参数切换 Pre-Norm、Post-Norm、NoRMSNorm、NoPE，以及通过 `use_silu_ffn` 切换 SwiGLU/SiLU。
- **训练组件**：实现了交叉熵（数值稳定）、AdamW（解耦 weight decay）、余弦学习率调度（warmup 支持）、全局梯度裁剪、随机 batch 采样和 checkpoint 保存/加载。
- **文本生成**：支持 temperature 和 top-p 采样，遇 `<|endoftext|>` 停止。

## 复现说明

- **环境与依赖**：Python 3.12，PyTorch 2.5.1+cu121，以及 `einops>=0.8`、`einx>=0.4`、`jaxtyping>=0.3`、`numpy>=2.4`、`psutil>=7`、`pytest>=9.0`、`pytest-timeout`、`regex>=2026.3.32`、`tiktoken>=0.12.0`、`tqdm>=4.67`、`wandb>=0.25`、`ty>=0.0.26`、`ruff>=0.15.8`。
- **数据准备**：
  - 下载 TinyStories 数据集（使用镜像站）：
    `wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -O data/TinyStories/TinyStoriesV2-GPT4-train.txt`
  - 训练 tokenizer 并编码：
    `python scripts/prepare_tinystories.py --train_file TinyStoriesV2-GPT4-train.txt`
- **Tokenizer、训练与生成命令**：详见 `submission/scripts/` 下的脚本。
- **同步命令**：`python3 scripts/sync_a1_submission.py --name '马万里'`
- **配置文件**：无

## 代码与脚本

- 真实实现：`submission/cs336_basics/`
- 测试 adapter：`submission/tests/adapters.py`
- 训练、数据编码与生成脚本：`submission/scripts/`
- 实现说明：如上文“实现说明”一节所述。

## 实验日志

- 日志目录：`logs/`
- 与报告中实验的对应说明：
  - `tokenizer_tinystories.jsonl`：TinyStories tokenizer 训练与编码指标日志
  - `train_log_base.jsonl`：TinyStories baseline 训练日志
  - `lr_sweep/tr_*_log.jsonl`：学习率扫描各 run 日志
  - `batch_sweep/bs_*_log.jsonl`：batch size 扫描各 run 日志
  - `ablation/ablation_*_log.jsonl`：四个消融实验日志

## 飞书补充文档

https://fudan-nlp.feishu.cn/wiki/TLLYwcIk6iwoRxkHErzcx2ksnHh