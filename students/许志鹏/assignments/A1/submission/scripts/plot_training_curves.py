from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from a train_log.jsonl file.")
    parser.add_argument("run_dir", help="Run directory containing train_log.jsonl.")
    parser.add_argument("--out", default=None, help="Output PNG path. Defaults to <run_dir>/training_curves.png.")
    return parser.parse_args()


def load_records(log_path: Path) -> list[dict]:
    records = []
    if not log_path.exists():
        return records
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def split_series(records: list[dict], split: str, key: str) -> tuple[list[int], list[float]]:
    xs = []
    ys = []
    for record in records:
        value = record.get(key)
        if record.get("split") == split and value is not None and math.isfinite(float(value)):
            xs.append(int(record["step"]))
            ys.append(float(value))
    return xs, ys


def plot_curves(run_dir: str | Path, out_path: str | Path | None = None) -> Path | None:
    run_dir = Path(run_dir)
    log_path = run_dir / "train_log.jsonl"
    records = load_records(log_path)
    if not records:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_path) if out_path is not None else run_dir / "training_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    train_steps, train_loss = split_series(records, "train", "loss")
    val_steps, val_loss = split_series(records, "val", "loss")
    train_ppl_steps, train_ppl = split_series(records, "train", "ppl")
    val_ppl_steps, val_ppl = split_series(records, "val", "ppl")
    lr_steps = [int(record["step"]) for record in records if record.get("split") == "train" and record.get("lr") is not None]
    lrs = [float(record["lr"]) for record in records if record.get("split") == "train" and record.get("lr") is not None]
    tps_steps = [
        int(record["step"])
        for record in records
        if record.get("split") == "train" and record.get("tokens_per_sec") is not None
    ]
    tps = [
        float(record["tokens_per_sec"])
        for record in records
        if record.get("split") == "train" and record.get("tokens_per_sec") is not None
    ]
    mfu_steps = [
        int(record["step"])
        for record in records
        if record.get("split") == "train" and record.get("mfu_percent") is not None
    ]
    mfu = [
        float(record["mfu_percent"])
        for record in records
        if record.get("split") == "train" and record.get("mfu_percent") is not None
    ]

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    loss_ax, ppl_ax, lr_ax, tps_ax, mfu_ax, blank_ax = axes.flatten()

    if train_loss:
        loss_ax.plot(train_steps, train_loss, label="train", linewidth=1.2)
    if val_loss:
        loss_ax.plot(val_steps, val_loss, label="val", marker="o", linewidth=1.5)
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("step")
    loss_ax.set_ylabel("cross entropy")
    loss_ax.grid(alpha=0.25)
    loss_ax.legend()

    if train_ppl:
        ppl_ax.plot(train_ppl_steps, train_ppl, label="train", linewidth=1.2)
    if val_ppl:
        ppl_ax.plot(val_ppl_steps, val_ppl, label="val", marker="o", linewidth=1.5)
    ppl_ax.set_title("Perplexity")
    ppl_ax.set_xlabel("step")
    ppl_ax.set_ylabel("ppl")
    ppl_ax.grid(alpha=0.25)
    ppl_ax.legend()

    if lrs:
        lr_ax.plot(lr_steps, lrs, color="tab:green", linewidth=1.2)
    lr_ax.set_title("Learning Rate")
    lr_ax.set_xlabel("step")
    lr_ax.set_ylabel("lr")
    lr_ax.grid(alpha=0.25)

    if tps:
        tps_ax.plot(tps_steps, tps, color="tab:purple", linewidth=1.2)
    tps_ax.set_title("Throughput")
    tps_ax.set_xlabel("step")
    tps_ax.set_ylabel("tokens/sec")
    tps_ax.grid(alpha=0.25)

    if mfu:
        mfu_ax.plot(mfu_steps, mfu, color="tab:red", linewidth=1.2)
    mfu_ax.set_title("MFU")
    mfu_ax.set_xlabel("step")
    mfu_ax.set_ylabel("percent")
    mfu_ax.grid(alpha=0.25)

    blank_ax.axis("off")

    fig.suptitle(run_dir.name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    try:
        out = plot_curves(args.run_dir, args.out)
    except ImportError as exc:
        print(f"matplotlib is required for plotting: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if out is None:
        print("No training records found.")
    else:
        print(out)


if __name__ == "__main__":
    main()
