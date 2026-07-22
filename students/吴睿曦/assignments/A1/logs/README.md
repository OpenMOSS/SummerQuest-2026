# A1 实验日志

- `tokenizer/`：两个 tokenizer 的训练汇总，以及 1,000-document 双域压缩与吞吐测量。
- `tinystories_baseline/`：3e-4 baseline 的逐点 JSONL 与汇总。
- `lr_sweep/`：1e-4、6e-4、1e-3 与 1e-1 发散 run。
- `batch_size/`：batch 1/64/128/768 的真实数据短跑，以及脱敏容量 probe。
- `ablations/`：No RMSNorm、Post-Norm、NoPE 与等参数 SiLU。
- `owt_baseline/`：OWT 10K-step 训练的逐点日志与汇总。
- `generation/`：TinyStories 与 OWT 的固定 seed 可复现生成记录。

训练 JSONL 逐点包含 step、processed tokens、wall-clock、train/validation loss 和学习率；
`summary.json` 记录最终 loss、总时长、总 tokens、吞吐、参数量、关键配置和变体。

Post-Norm 的前约 3K steps 曾与 OWT 编码短暂重叠，发现 CPU 争用后立即暂停编码；因此该 run
总墙钟时间比其余同形状 run 多约 36 秒，loss 对比不受影响，但不使用该总时长判断架构速度。
