# A1 脱敏实验日志

> 本日志仅记录公开数据、相对路径、超参数和实验结果。数据集、tokenizer 产物、模型权重和运行环境的内部信息均未提交。

> 本文件是人类可读的实验索引。结构化汇总见 `summary.json`；训练类 run 的逐点 `step`、`wall_clock_sec`、`train_loss`、`lr`、`val_loss` 和 `tokens` 见同目录 JSONL 及 `lr_sweep/`、`batch_size/`。

## 1. 数据与 Tokenizer

### TinyStories

- Train 文本：2,227,753,162 bytes，SHA-256 `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443`
- Validation 文本：22,502,601 bytes，SHA-256 `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d`
- BPE：vocab 10,000，9,743 merges，`<|endoftext|>` ID 256
- BPE 耗时：716.529 s；峰值 RSS：10,855.2 MiB
- 资源限制：时间与内存均通过

| Split | Token 数 | dtype | bytes/token | 编码耗时 | 峰值 RSS |
| --- | ---: | --- | ---: | ---: | ---: |
| Train | 541,229,347 | `uint16` | 4.1161 | 3,654.143 s | 2,101.4 MiB |
| Validation | 5,465,883 | `uint16` | 4.1169 | 33.458 s | 56.8 MiB |

### OpenWebText

- Train 文本：11,920,511,059 bytes，SHA-256 `bbeb7f291a981ecfd5cf44b84d0f654b9e96c53dff99f2556b7d2cccaf8c1918`
- Validation 文本：289,998,753 bytes，SHA-256 `2406f278e71829d273b315e9b403285baea7022b26a96d2728dd8b776ea40660`
- BPE：vocab 32,000，31,743 merges，`<|endoftext|>` ID 256
- BPE 耗时：3,416.680 s；峰值 RSS：13,025.9 MiB
- 资源限制：时间与内存均通过

| Split | Token 数 | dtype | bytes/token | 编码耗时 | 峰值 RSS |
| --- | ---: | --- | ---: | ---: | ---: |
| Train | 2,727,120,452 | `uint16` | 4.3711 | 17,403.259 s | 10,456.9 MiB |
| Validation | 66,401,098 | `uint16` | 4.3674 | 439.563 s | 305.0 MiB |

### Tokenizer 抽样对比

随机种子为 42，从每个训练集使用 reservoir sampling 抽样 10 篇文档。

| 文档样本 | 样本 bytes | TinyStories tokenizer | OWT tokenizer |
| --- | ---: | ---: | ---: |
| TinyStories | 8,648 | 2,114 tokens / 4.0908 bytes/token | 2,189 tokens / 3.9507 bytes/token |
| OWT | 88,949 | 26,601 tokens / 3.3438 bytes/token | 19,611 tokens / 4.5357 bytes/token |

## 2. TinyStories 共同配置

- Vocab size：10,000
- Context length：256
- `d_model / d_ff`：512 / 1,344
- Layers / heads：4 / 16
- RoPE theta：10,000
- Training steps：10,000
- Warmup steps：200
- 初始化：±3σ truncated normal
- 默认 batch size：128
- 默认 processed tokens：327,680,000

## 3. Learning-rate sweep

除 maximum learning rate 外，数据、模型、seed、batch size、训练步数和其他 optimizer 参数保持一致。

| Maximum LR | Final train | Final validation | Best validation | Best step | Wall time | 有限数检查 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0003 | 1.489005 | 1.514594 | 1.510579 | 9,500 | 1,015.467 s | 通过 |
| 0.0005 | 1.416070 | 1.443639 | 1.438100 | 9,500 | 1,014.958 s | 通过 |
| 0.0008 | 1.365790 | 1.394515 | 1.389458 | 9,500 | 1,015.518 s | 通过 |
| 0.0010 | 1.340495 | 1.375914 | 1.370863 | 9,500 | 1,014.962 s | 通过 |
| 0.0030 | 1.292501 | 1.325279 | 1.321314 | 9,500 | 1,013.410 s | 通过 |
| 0.0100 | 1.299076 | 1.326189 | 1.325737 | 9,500 | 1,013.858 s | 通过 |
| 0.0300 | 1.431808 | 1.458449 | 1.458449 | 10,000 | 1,016.686 s | 通过；高 LR 阶段不收敛 |
| 0.1000 | 1.825361 | 1.859377 | 1.859377 | 10,000 | 1,021.705 s | 通过；发散/失稳 run，非 NaN 型 |

最佳 run 是 maximum LR 0.003，validation loss 1.321314。LR 0.03 在退火前长时间不收敛；LR 0.1 是指定的发散/失稳 run，虽然没有出现 NaN，但在退火后仍明显较差。因此实测稳定边界在 0.01–0.03 之间。

## 4. Batch-size sweep

使用 maximum LR 0.003 和固定 10,000 steps。由于固定的是 step 而不是 token 数，不同 batch 处理的总 token 数不同。

| Batch | Processed tokens | Final validation | Best validation | Best step | Wall time | 状态 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2,560,000 | 2.879569 | 2.867573 | 9,900 | 180.784 s | 完成 |
| 64 | 163,840,000 | 1.410141 | 1.403573 | 9,400 | 559.679 s | 完成 |
| 128 | 327,680,000 | 1.325279 | 1.321314 | 9,500 | 1,013.410 s | 复用 LR 0.003 run |
| 256 | 655,360,000 | 1.261825 | 1.259440 | 9,500 | 1,917.586 s | 完成 |
| 512 | 1,310,720,000 | 1.231734 | 1.228391 | 9,100 | 3,720.406 s | 完成 |
| 1,024 | — | — | — | — | — | CUDA OOM |

## 5. 架构消融

| 实验 | Maximum LR | Final validation | Best validation | Best step | Wall time | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pre-Norm + RoPE + SwiGLU | 0.003 | 1.325279 | 1.321314 | 9,500 | 1,013.410 s | 稳定 |
| No RMSNorm | 0.003 | NaN | 3.079791 | 100 | 914.585 s | 发散，967 条非有限 loss |
| No RMSNorm | 0.0003 | 1.531426 | 1.526934 | 9,500 | 916.381 s | 稳定 |
| Post-Norm | 0.003 | 1.362058 | 1.360077 | 9,500 | 1,014.215 s | 稳定 |
| NoPE | 0.003 | 1.382821 | 1.379379 | 9,500 | 925.615 s | 稳定 |
| SiLU FFN (`d_ff=2048`) | 0.003 | 1.329824 | 1.325535 | 9,500 | 1,000.880 s | 稳定 |

## 6. OpenWebText LM

- Vocab size：32,000
- 模型架构、batch size 和 training steps：与 TinyStories 主实验相同
- Maximum LR：0.003
- Processed tokens：327,680,000
- Final train loss：3.924047
- Final / best validation loss：3.912838（step 10,000）
- Wall time：1,300.142 s

OWT 的分布、词汇和 tokenizer 均与 TinyStories 不同，因此 per-token loss 只适合在同一数据/tokenizer 设定内比较。

## 7. 文本生成设置

| 模型 | Prompt | Temperature | Top-p | Max new tokens | 终止原因 |
| --- | --- | ---: | ---: | ---: | --- |
| TinyStories LR 0.003 | `Once upon a time` | 0.8 | 0.95 | 256 | 生成 `<|endoftext|>` |
| OpenWebText | `The` | 0.8 | 0.95 | 256 | 达到 token 上限 |

完整生成文本与分析见上级 `README.md`。

## 8. 曲线对应

- `../assets/lr_sweep_vs_step.png`：各 LR 的 validation loss / step
- `../assets/lr_sweep_vs_time.png`：各 LR 的 validation loss / wall-clock time
- `../assets/batch_size_vs_step.png`：各 batch size 的 validation loss / step
- `../assets/batch_size_vs_tokens.png`：各 batch size 的 validation loss / processed tokens
- `../assets/architecture_ablations.png`：架构消融
- `../assets/dataset_comparison.png`：TinyStories 与 OWT
- `../assets/tokenizer_compression.png`：Tokenizer 压缩率

## 9. 官方测试

- 命令：`.venv/bin/python -m pytest`
- 结果：`47 passed, 1 xpassed in 24.63s`
- 退出码：0
- 说明：`xpassed` 不是失败，而是上游标记为预期失败的测试实际通过。
