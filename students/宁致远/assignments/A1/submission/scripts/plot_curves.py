"""Plot train/val loss vs step and vs wall-clock from a train.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path: str):
    steps, wall, train, val_steps, val, val_wall = [], [], [], [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "train_loss" in r:
                steps.append(r["step"]); wall.append(r["wall"]); train.append(r["train_loss"])
            if "val_loss" in r:
                val_steps.append(r["step"]); val.append(r["val_loss"]); val_wall.append(r["wall"])
    return steps, wall, train, val_steps, val_wall, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="pairs of label=path/to/train.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for spec in args.runs:
        # Split on the LAST '=' so labels can contain '=' (e.g. lr=1e-4=path)
        label, path = spec.rsplit("=", 1)
        steps, wall, train, vs, vw, v = load(path)
        axs[0].plot(steps, train, label=f"{label} train", alpha=0.6)
        if v:
            axs[0].plot(vs, v, label=f"{label} val", linestyle="--")
            axs[1].plot(vw, v, label=f"{label} val", linestyle="--")
        axs[1].plot(wall, train, label=f"{label} train", alpha=0.6)
    axs[0].set_xlabel("step"); axs[0].set_ylabel("loss"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].set_xlabel("wall (s)"); axs[1].set_ylabel("loss"); axs[1].legend(); axs[1].grid(alpha=.3)
    if args.title: fig.suptitle(args.title)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
