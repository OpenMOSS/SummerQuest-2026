from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无头模式
import matplotlib.pyplot as plt


def load_jsonl(path: str | Path) -> list[dict]:
    """加载 JSONL 文件，返回 dict 列表。"""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_series(records: list[dict], key: str) -> tuple[list, list]:
    """从日志记录中提取 (step, value) 序列，跳过缺失 key 的行。"""
    steps = []
    values = []
    for r in records:
        if key in r and r[key] is not None:
            steps.append(r["step"])
            values.append(r[key])
    return steps, values


def plot_single(records: list[dict], out_path: str, title: str = "Training Loss"):
    """绘制单个训练 run 的 loss 曲线。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # train loss
    steps, loss = extract_series(records, "train_loss")
    ax.plot(steps, loss, alpha=0.6, label="train_loss", color="steelblue", linewidth=0.8)

    # val loss
    steps_v, val_loss = extract_series(records, "val_loss")
    ax.plot(steps_v, val_loss, label="val_loss", color="coral", linewidth=1.5, marker="o", markersize=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注最终 val loss
    if val_loss:
        final_vl = val_loss[-1]
        ax.axhline(y=final_vl, color="coral", linestyle="--", alpha=0.5)
        ax.text(steps_v[-1], final_vl, f"  final={final_vl:.4f}", color="coral", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}", flush=True)
    plt.close()


def plot_lr_sweep(log_dir: str, out_path: str):
    """绘制学习率扫对比图。"""
    log_dir = Path(log_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    # 找到所有子目录 lr_*
    lr_dirs = sorted(log_dir.glob("lr_*"))
    if not lr_dirs:
        print(f"[plot] no lr_* directories found in {log_dir}", flush=True)
        return

    colors = plt.cm.tab10.colors

    for i, lr_dir in enumerate(lr_dirs):
        jsonl_path = lr_dir / "train_log.jsonl"
        if not jsonl_path.exists():
            continue
        records = load_jsonl(jsonl_path)
        steps, val_loss = extract_series(records, "val_loss")
        if not val_loss:
            steps, val_loss = extract_series(records, "train_loss")
        if not val_loss:
            continue

        # 从目录名提取 lr
        lr_label = lr_dir.name.replace("lr_", "").replace("p", ".").replace("neg_", "-")
        color = colors[i % len(colors)]
        ax.plot(steps, val_loss, label=f"lr={lr_label}", color=color, linewidth=1.5, marker="o", markersize=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Learning Rate Sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}", flush=True)
    plt.close()


def plot_ablation(
    baseline_log: str,
    ablations: list[tuple[str, str]],
    out_path: str,
):
    """绘制消融实验对比图。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # baseline
    records = load_jsonl(baseline_log)
    steps, val_loss = extract_series(records, "val_loss")
    if not val_loss:
        steps, val_loss = extract_series(records, "train_loss")
    ax.plot(steps, val_loss, label="baseline (Pre-Norm + RoPE + RMSNorm + SwiGLU)",
            color="black", linewidth=2.0, linestyle="-")

    # 各消融
    colors = plt.cm.Set1.colors
    linestyles = ["--", "-.", ":", (0, (3, 1, 1, 1))]

    for i, (name, log_path) in enumerate(ablations):
        records = load_jsonl(log_path)
        steps, val_loss = extract_series(records, "val_loss")
        if not val_loss:
            steps, val_loss = extract_series(records, "train_loss")
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        ax.plot(steps, val_loss, label=name, color=color, linewidth=1.5, linestyle=ls)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Ablation Study: Architecture Variants vs Baseline")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}", flush=True)
    plt.close()


def plot_batch_size(log_dir: str, out_path: str):
    """绘制 batch size 对比图。"""
    log_dir = Path(log_dir)
    fig, ax = plt.subplots(figsize=(10, 6))

    bs_dirs = sorted(log_dir.glob("bs_*"))
    if not bs_dirs:
        print(f"[plot] no bs_* directories found in {log_dir}", flush=True)
        return

    colors = plt.cm.tab10.colors

    for i, bs_dir in enumerate(bs_dirs):
        jsonl_path = bs_dir / "train_log.jsonl"
        if not jsonl_path.exists():
            continue
        records = load_jsonl(jsonl_path)
        steps, val_loss = extract_series(records, "val_loss")
        if not val_loss:
            steps, val_loss = extract_series(records, "train_loss")
        if not val_loss:
            continue

        bs_label = bs_dir.name.replace("bs_", "")
        color = colors[i % len(colors)]
        ax.plot(steps, val_loss, label=f"batch_size={bs_label}", color=color, linewidth=1.5, marker="o", markersize=2)

    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Batch Size Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"[plot] saved to {out_path}", flush=True)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="实验图表生成")

    subparsers = parser.add_subparsers(dest="mode", help="图表类型")

    # single: 单个训练曲线
    p_single = subparsers.add_parser("single", help="单个训练 loss 曲线")
    p_single.add_argument("--log", type=str, required=True, help="JSONL 日志路径")
    p_single.add_argument("--out", type=str, required=True, help="输出图片路径")
    p_single.add_argument("--title", type=str, default="Training Loss")

    # lr_sweep: 学习率扫对比
    p_lr = subparsers.add_parser("lr_sweep", help="学习率扫对比图")
    p_lr.add_argument("--log-dir", type=str, required=True, help="lr_sweep 目录")
    p_lr.add_argument("--out", type=str, required=True)

    # ablation: 消融对比
    p_abl = subparsers.add_parser("ablation", help="消融实验对比图")
    p_abl.add_argument("--baseline", type=str, required=True, help="baseline JSONL")
    p_abl.add_argument("--ablations", type=str, nargs="*", required=True,
                       help="消融列表: name1 log1 name2 log2 ...")
    p_abl.add_argument("--out", type=str, required=True)

    # batch_size: batch size 对比
    p_bs = subparsers.add_parser("batch_size", help="batch size 对比图")
    p_bs.add_argument("--log-dir", type=str, required=True)
    p_bs.add_argument("--out", type=str, required=True)

    args = parser.parse_args()

    if args.mode == "single":
        records = load_jsonl(args.log)
        plot_single(records, args.out, args.title)

    elif args.mode == "lr_sweep":
        plot_lr_sweep(args.log_dir, args.out)

    elif args.mode == "ablation":
        # 解析 ablations: name1 log1 name2 log2 ...
        if len(args.ablations) % 2 != 0:
            print("Error: --ablations must be pairs of name log_path", file=sys.stderr)
            sys.exit(1)
        ablations = []
        for i in range(0, len(args.ablations), 2):
            ablations.append((args.ablations[i], args.ablations[i + 1]))
        plot_ablation(args.baseline, ablations, args.out)

    elif args.mode == "batch_size":
        plot_batch_size(args.log_dir, args.out)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
