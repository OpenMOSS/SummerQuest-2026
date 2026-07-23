# A1 学习记录 04：完整实验里程碑

- 日期：2026-07-17
- 状态：TinyStories、OpenWebText、跨语料 tokenizer 对比与全部消融实验均已完成

## 1. 本阶段结论

这一阶段把此前通过单元测试的 tokenizer、Transformer、训练循环和生成器接成了真实实验链路。完整 TinyStories tokenizer 在资源限制内训练完成，训练集和验证集均已编码；模型处理 327,680,000 个 token 后取得 `1.355541` validation loss，达到题目要求的 `1.45`；学习率扫、batch size 扫、架构消融和文本生成也已完成。

OpenWebText 32K tokenizer、完整 train/validation 编码、20,000-step LM 和生成同样完成。OpenWebText 最终 validation loss 为 `4.015182`，按 validation compression ratio 换算为 `0.919 nats/byte`；跨语料对比显示 tokenizer 的语料匹配程度与词表大小都会影响压缩率。

## 2. 实验设计中的主要原则

### 2.1 区分正确性、吞吐与最终质量

- 单元测试和单批次过拟合回答“实现链路是否正确”。
- GPU probe 回答“什么 batch size 能放进显存、吞吐是否继续上升”。
- 固定 token 预算的 sweep 回答“在相近计算与数据预算下，超参数如何影响收敛”。
- 完整 TinyStories baseline 回答“模型在题目给定总 token 预算下能否达到目标 validation loss”。

这几类实验不能相互替代。例如 batch size 64 能在显存中运行，并不意味着它在固定 token 预算下得到最低 loss；单批次可以过拟合，也不意味着完整语料会泛化。

### 2.2 Batch size 比较固定总 token 数

batch size sweep 固定总训练量为 `2,097,152` tokens，同时按 token 数固定 warmup、验证间隔和验证预算。这样不同 batch 不会因为 step 数相同而看到不同数量的数据。由于 `tokens = batch_size × context_length × steps`，batch 越大，对应 optimizer update 次数越少。

## 3. Tokenizer 实验与编码

### 3.1 BPE 训练

| 项目 | 结果 |
|---|---:|
| vocab size | 10,000 |
| special token | `<\|endoftext\|>` |
| elapsed | 355.968 s |
| peak RSS | 10,850.37 MiB |
| longest learned token | `b' accomplishment'`，15 bytes |

完整语料训练约 5.93 分钟、peak RSS 约 10.60 GiB，低于题目的 30 分钟和 30 GB 限制。最长 token 是带前导空格的常见英文词，符合 GPT-2 风格 pre-tokenization 会把可选前导空格并入英文词的预期。

在 5M sample 的 profile 中，heap 优化后 `count_pretokens` 已经取代 pair 选择成为主要热点；完整语料的内存和时间也主要由预分词及其计数结构驱动，而不是 GPU 计算。

### 3.2 数据编码

| split | tokens | elapsed | throughput | bytes/token | serialized size |
|---|---:|---:|---:|---:|---:|
| validation | 5,461,210 | 12.785 s | 1.679 MiB/s | 4.12044 | 10.416 MiB |
| training | 540,796,778 | 1,204.044 s | 1.765 MiB/s | 4.11939 | 1,031.488 MiB |

`bytes/token` 是这里的 compression ratio：训练集平均一个 token 表示约 4.12 个原始 byte。词表大小 10,000 小于 `uint16` 能表示的 65,536 个不同值，因此编码结果可以安全保存为 `uint16`，相较 `int32` 将磁盘占用减半。

以训练集实测的 `1.76451 MiB/s` 单进程吞吐粗略外推，编码 825 GB 文本约需 124 小时，即 5.16 天。这个估计忽略了语料结构、存储速度和并行化差异，只用于量级判断。

### 3.3 OpenWebText BPE 训练

| 项目 | 结果 |
|---|---:|
| vocab size | 32,000 |
| merges | 31,743 |
| elapsed | 4,905.077 s（约 1 小时 21 分 45 秒） |
| peak RSS | 9,979.38 MiB（约 9.75 GiB） |
| longest learned token | `b'\xc3\x83\xc3\x82'` 重复 16 次，64 bytes |

耗时和内存均低于题目的 12 小时、100 GB 限制。最长 token 解码后是 `ÃÂ` 的重复，属于网页中常见的字符编码错乱；它与 TinyStories tokenizer 学到的完整叙事词形成对照，反映 OpenWebText 更广泛也更噪杂的语料分布。

### 3.4 OpenWebText 编码

| split | tokens | elapsed | workers | throughput | bytes/token |
|---|---:|---:|---:|---:|---:|
| validation | 66,401,098 | 566.855 s | 1 | 0.488 MiB/s | 4.36738 |
| training | 2,727,120,452 | 1,536 s | 16 | 7.401 MiB/s | 4.37110 |

单进程编码完整训练集会逼近作业时间窗口，因此先在 validation split 上验证按换行边界分片、并行编码、按序拼接的输出与串行版本具有相同大小和 SHA-256，再用 16 个 worker 编码训练集。实测并行吞吐约为单进程的 15.2 倍，同时保持 token 序列不变。这次优化的关键不是改变 tokenizer，而是利用 `encode_iterable` 已经逐行定义的独立边界并保持分片顺序。

### 3.5 跨语料 tokenizer 对比

两个语料各取约 5 MiB，并用两个 tokenizer 交叉编码：

| corpus | TinyStories 10K | OpenWebText 32K |
|---|---:|---:|
| TinyStories | **4.11857 bytes/token** | 4.00383 bytes/token |
| OpenWebText | 3.22096 bytes/token | **4.42749 bytes/token** |

各自 tokenizer 在本域样本上压缩率更高。OpenWebText tokenizer 在 OWT 样本上少产生约 27.3% token；但它虽然词表更大，在 TinyStories 上仍略逊于 10K tokenizer，说明 domain-specific merge rules 可以抵消单纯扩大词表的优势。四组运行吞吐接近，主要差异来自 token 数而非 I/O。

## 4. GPU batch probe

环境为 NVIDIA RTX 4060 Ti 16GB，PyTorch `2.11.0+cu130`，context length 256。计时对象是完整训练 step，而非仅 forward。

| batch size | seconds/step | tokens/s | peak allocated |
|---:|---:|---:|---:|
| 1 | 0.01163 | 22,016 | 0.418 GiB |
| 16 | 0.12284 | 33,343 | 2.432 GiB |
| 32 | 0.25433 | 32,210 | 4.567 GiB |
| 64 | 0.51103 | 32,061 | 8.860 GiB |

batch 96 只有在启用 expandable segments 时才能运行，而且更慢；batch 128 OOM。吞吐从 batch 1 到 16 明显提升，但 16 之后基本平台化，因此继续增大 batch 主要消耗显存，没有带来吞吐收益。最终 baseline 使用 batch 64 是为了在可接受显存内以较少 step 覆盖题目要求的总 token 数，不代表 batch 64 在所有预算下都最优。

## 5. 学习率实验

前四个 run 使用相同的 batch 64、context length 256 和 `9,994,240` processed tokens；更高学习率在已经足以判断不稳定或明显较差时提前停止。

| max learning rate | completed steps | processed tokens | wall clock | final val loss | 观察 |
|---:|---:|---:|---:|---:|---|
| `3e-4` | 610 | 9,994,240 | 331.327 s | 2.678493 | 收敛慢 |
| `1e-3` | 610 | 9,994,240 | 331.858 s | 2.207574 | 明显改善 |
| `3e-3` | 610 | 9,994,240 | 331.810 s | **2.020021** | 此预算下最好 |
| `1e-2` | 610 | 9,994,240 | 331.802 s | 2.552282 | 越过较优区间 |
| `3e-2` | 200 | 3,276,800 | 108.409 s | 3.624071 | 明显较差，提前停止 |
| `1e-1` | 100 | 1,638,400 | 57.238 s | 4.512198 | warmup 中强烈不稳定，提前停止 |

`1e-1` 在 step 10、20、25 的 train loss 分别约为 `7.718`、`10.929`、`11.410`，step 25 validation loss 为 `9.858`；随着学习率随后衰减，它又部分恢复。因此更准确的结论是“出现过边缘稳定性之外的爆炸并且在当前预算内没有收敛”，而不是声称整个 run 最终产生 NaN。

短预算结果呈现典型的先改善后恶化：从 `3e-4` 增大到 `3e-3` 加快收敛，但 `1e-2` 开始变差，`3e-2` 和 `1e-1` 出现明显不稳定。完整 baseline 因此选择 `3e-3`。

## 6. Batch size 实验

所有成功 run 固定 `2,097,152` processed tokens，并使用随 batch size 按平方根缩放的 learning rate。

| batch size | steps | max learning rate | wall clock | final val loss |
|---:|---:|---:|---:|---:|
| 1 | 8,192 | `3.75e-4` | 94.989 s | 2.776371 |
| 16 | 512 | `1.5e-3` | 66.963 s | **2.595449** |
| 32 | 256 | `2.12132e-3` | 69.688 s | 2.693691 |
| 64 | 128 | `3e-3` | 71.995 s | 3.265839 |
| 128 | — | — | — | OOM |

batch 16 在这组固定 token 预算实验中同时取得最快 wall clock 和最低 validation loss。batch 1 的小矩阵利用率较差，运行更慢；batch 32 和 64 虽然单步处理更多 token，但总吞吐没有继续改善，而且固定 token 数意味着 optimizer update 更少，导致短预算下的优化效果变差。这个结论依赖当前 learning-rate scaling 和 token 预算，不能推广为“batch 16 永远最好”。

## 7. 完整 TinyStories baseline

| 项目 | 结果 |
|---|---:|
| model | vocab 10,000；context 256；`d_model=512`；`d_ff=1344`；4 layers；16 heads |
| batch size | 64 |
| max learning rate | `3e-3` |
| training steps | 20,000 |
| processed tokens | 327,680,000 |
| wall clock | 10,501.408 s（约 2 h 55 min） |
| final train loss | 1.352580 |
| final validation loss | **1.355541** |
| target | validation loss ≤ 1.45，已达到 |

短预算 sweep 的 validation loss 仍在 2 左右，但把训练量扩展到题目要求的 327.68M tokens 后，loss 降到 1.3555。这个差异说明短 sweep 适合筛选明显不合适的超参数，却不能直接代替完整训练结论。

## 8. 文本生成

生成配置为 seed 42、temperature 0.8、top-p 0.9，prompt 为 `Once upon a time, there was a little girl named Lily.`。模型在达到 256 个新 token 前生成了 `<|endoftext|>`，因此按题目规则提前停止。

输出已经具备完整英文句法、人物一致性和简单故事结构，说明模型学到了 TinyStories 的主要分布；不足之处是情节非常模板化，“shy” 等描述重复，因果发展也较弱。质量主要受训练语料本身的简单风格、17M 级模型容量与训练预算，以及 temperature/top-p 的随机性与截断方式影响。

完整文本保存在 `experiments/generation_tinystories.txt`。

## 9. 架构消融

六个 run 使用共同的 batch 64、context length 256、2,000 steps 和 32,768,000-token 预算。除低学习率 No-RMSNorm 外，max learning rate 均为 `3e-3`。

| run | 变化 | wall clock | final train loss | final val loss | best val loss |
|---|---|---:|---:|---:|---:|
| control | Pre-Norm + RoPE + SwiGLU | 1,079.911 s | 1.763753 | 1.762212 | 1.758050 |
| No-RMSNorm | 删除 RMSNorm | 1,005.896 s | NaN | NaN | 3.208766 |
| No-RMSNorm，低 LR | 删除 RMSNorm，LR `1e-3` | 1,007.047 s | 1.793884 | 1.791714 | 1.786389 |
| Post-Norm | 改为 Post-Norm | 1,080.102 s | 1.722937 | **1.722657** | **1.718559** |
| NoPE | 移除 RoPE | 1,014.047 s | 2.035836 | 2.032531 | 2.031106 |
| SiLU | `d_ff=2048` 的 SiLU FFN | 1,075.549 s | 1.744851 | 1.741140 | 1.735889 |

原学习率下的 No-RMSNorm 在 step 180 首次产生非有限 train loss；此前最后一次 validation 是 step 100 的 `3.208766`。把 learning rate 降到 `1e-3` 后训练保持有限，说明 normalization 不只是改善最终 loss，也扩大了稳定学习率范围。

Post-Norm 在这个短预算内取得最低 validation loss，但模型只有 4 层，结论不能直接外推到更深的 Transformer。NoPE 明显较差，表明 256-token 上下文中位置结构仍然提供重要信息。参数量近似匹配的 SiLU 略优于 control 的 final loss，但略差于 Post-Norm；这组结果没有显示 SwiGLU 在当前短预算下具有优势。

这六个消融 run 内部使用一致的 AdamW `betas=(0.9, 0.999)` 和 weight decay `0.01`，所以它们之间可以做因果比较。此前 baseline、learning-rate sweep 和 batch-size sweep 使用 `betas=(0.9, 0.95)` 和 weight decay `0.1`，因此不能把历史 baseline 与本套件 control 的 loss 差异解释为结构效果。

## 10. OpenWebText baseline 与生成

OpenWebText 与 TinyStories 使用相同的 context length 256、`d_model=512`、`d_ff=1344`、4 layers、16 heads、batch 64、20,000 steps 和 `3e-3` max learning rate，词表按题目要求改为 32K。

| 项目 | 结果 |
|---|---:|
| processed tokens | 327,680,000 |
| wall clock | 4,205.326 s |
| final train loss | 4.029969 |
| final validation loss | 4.015182 |
| best validation loss | 3.996987，step 19,000 |
| validation nats/byte | 0.919357 |
| validation bits/byte | 1.326352 |

训练曲线持续下降并在末段接近 4.0。TinyStories 与 OpenWebText 的 tokenizer 和数据分布不同，不能直接用 per-token loss 排名；换算到 byte 后，OpenWebText 仍明显更难。这符合网页语料覆盖题材更广、文体更复杂且包含更多噪声的预期。两次训练还使用了不同 GPU，因此 wall-clock 只用于记录复现成本，不用于比较数据集计算难度。

以 prompt `The`、temperature 0.8、top-p 0.9 生成 256 个新 token。输出具有局部新闻句法和引语格式，但反复围绕 `the president` 打转，人物与政策关系互相矛盾。相比 TinyStories，OpenWebText 模型面对的语言分布更宽，而当前模型容量和 token 预算有限；采样随机性进一步放大了篇章一致性问题。

## 11. 证据边界与完成状态

18 个训练 run 的逐点 `metrics.jsonl`、18 份可移植配置和五张 SVG loss curve 均已回收到公开提交。OpenWebText 编码统计、四组 tokenizer 对比和两个完整生成样本也分别保存在机器可读 JSON 与文本文件中。图由 `scripts/plot_experiment_logs.py` 从原始 JSONL 生成；机器可读汇总见 `experiments/summary.json`。

至此，题面要求的 tokenizer、TinyStories baseline、learning-rate sweep、batch-size sweep、四类架构消融、OpenWebText baseline 和两组文本生成全部完成。
