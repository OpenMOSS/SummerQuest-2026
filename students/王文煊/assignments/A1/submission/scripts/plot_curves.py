"""Plot loss curves from JSONL training logs into compressed PNGs for the report.

Reads one or more JSONL logs and renders train/val curves against either step or
wall-clock seconds. Kept dependency-light (matplotlib only).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows, xkey, ykey):
    xs, ys = [], []
    for r in rows:
        if ykey in r and xkey in r:
            xs.append(r[xkey])
            ys.append(r[ykey])
    return xs, ys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", required=True, help="label=path pairs")
    p.add_argument("--xkey", default="step", choices=["step", "wall_clock_sec"])
    p.add_argument("--metric", default="val_loss", choices=["val_loss", "train_loss", "both"])
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--ymax", type=float, default=None)
    args = p.parse_args()

    plt.figure(figsize=(7, 4.2), dpi=110)
    for item in args.logs:
        label, path = item.rsplit("=", 1)
        rows = read_jsonl(path)
        if args.metric in ("val_loss", "both"):
            xs, ys = series(rows, args.xkey, "val_loss")
            if xs:
                plt.plot(xs, ys, marker="o", ms=3, label=f"{label} val")
        if args.metric in ("train_loss", "both"):
            xs, ys = series(rows, args.xkey, "train_loss")
            if xs:
                plt.plot(xs, ys, alpha=0.6, lw=1, label=f"{label} train")

    plt.xlabel("wall-clock seconds" if args.xkey == "wall_clock_sec" else "step")
    plt.ylabel("cross-entropy loss (per token)")
    if args.ymax:
        plt.ylim(top=args.ymax)
    if args.title:
        plt.title(args.title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
