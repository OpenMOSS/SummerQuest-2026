import json
import matplotlib.pyplot as plt
matplotlib.use("Agg") 
from pathlib import Path

def load_log(path):
    """读 jsonl,分离 train 和 val 记录"""
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "train_loss" in r:
                train_steps.append(r["step"])
                train_losses.append(r["train_loss"])
            if "val_loss" in r:
                val_steps.append(r["step"])
                val_losses.append(r["val_loss"])
    return train_steps, train_losses, val_steps, val_losses

# ============ 图1:TinyStories baseline 训练曲线 ============
def plot_baseline():
    ts, tl, vs, vl = load_log("runs/ts_baseline/train_log.jsonl")
    plt.figure(figsize=(8, 5))
    plt.plot(ts, tl, alpha=0.4, label="train loss")
    plt.plot(vs, vl, "o-", label="val loss", color="red")
    plt.axhline(1.45, ls="--", color="gray", label="target (1.45)")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("TinyStories Baseline Training Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("runs/plot_baseline.png", dpi=120, bbox_inches="tight")
    print("saved plot_baseline.png")

# ============ 图2:LR sweep 叠加(展示发散)============
def plot_lr_sweep():
    plt.figure(figsize=(8, 5))
    lrs = {
        "1e-4": "runs/lr_1e-4",
        "3e-4 (baseline)": "runs/ts_baseline",
        "1e-3": "runs/lr_1e-3",
        "3e-3": "runs/lr_3e-3",
        "1e-2": "runs/lr_1e-2",
        "1e-1": "runs/lr_1e-1",
    }
    for label, d in lrs.items():
        p = Path(d) / "train_log.jsonl"
        if not p.exists():
            continue
        ts, tl, _, _ = load_log(p)
        plt.plot(ts, tl, label=f"lr={label}", alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("train loss")
    plt.title("Learning Rate Sweep")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 10)              # 限制 y 轴,发散的会冲顶
    plt.savefig("runs/plot_lr_sweep.png", dpi=120, bbox_inches="tight")
    print("saved plot_lr_sweep.png")

# ============ 图3:消融对比 ============
def plot_ablations():
    plt.figure(figsize=(8, 5))
    runs = {
        "baseline": "runs/ts_baseline",
        "no RMSNorm": "runs/ts_nonorm",
        "post-norm": "runs/ts_postnorm",
        "NoPE": "runs/ts_norope",
        "SiLU FFN": "runs/ts_silu",
    }
    for label, d in runs.items():
        p = Path(d) / "train_log.jsonl"
        if not p.exists():
            continue
        vs, vl = load_log(p)[2:]        # 只要 val
        plt.plot(vs, vl, "o-", label=label, alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("val loss")
    plt.title("Ablations vs Baseline")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("runs/plot_ablations.png", dpi=120, bbox_inches="tight")
    print("saved plot_ablations.png")

if __name__ == "__main__":
    plot_baseline()
    plot_lr_sweep()
    plot_ablations()