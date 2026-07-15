# A1 实验日志索引

每个正式训练目录包含：

- `metrics.jsonl`：逐点评估记录，包含 `step`、`processed_tokens`、`wall_clock_sec`、`train_loss`、`lr`，并按配置间隔记录 `val_loss`；
- `summary.json`：最终 validation loss、总训练时间、模型结构、batch size、world size、总步数和 processed tokens；
- `generation.txt`：仅 TinyStories 和 OWT 主实验提供，对应报告中的生成样本。

目录对应关系：

- `ts_baseline/`：TinyStories 主训练；
- `lr_sweep/`：固定学习率 sweep，包含稳定、停滞和发散 run；
- `batch_sweep/`：不同 batch size 的固定 token 预算实验；
- `ablation_no_rmsnorm_lr1e3/`、`ablation_no_rmsnorm_lr3e4/`：删除 RMSNorm；
- `ablation_postnorm/`：post-norm；
- `ablation_nope/`：NoPE；
- `ablation_silu/`：无门控 SiLU FFN；
- `owt_baseline/`：OpenWebText 主训练。

模型权重、TensorBoard event、数据集和 tokenized arrays 均未提交。`summary.json` 中原本用于本地恢复训练的 checkpoint 路径已替换为公开说明，避免暴露内部文件系统信息。
