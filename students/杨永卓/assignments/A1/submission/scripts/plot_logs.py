#!/usr/bin/env python3
"""Render public loss curves from one or more JSONL training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_log(path: str):
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True)
    parser.add_argument("--label", action="append")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    labels = args.label or [Path(path).parent.name for path in args.log]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for path, label in zip(args.log, labels):
        records = read_log(path)
        train = [record for record in records if "train_loss" in record]
        val = [record for record in records if "val_loss" in record]
        axes[0].plot([record["step"] for record in train], [record["train_loss"] for record in train], label=label)
        axes[0].scatter([record["step"] for record in val], [record["val_loss"] for record in val], s=14)
        axes[1].plot(
            [record["wall_clock_sec"] for record in train],
            [record["train_loss"] for record in train],
            label=label,
        )
    axes[0].set(xlabel="Step", ylabel="Cross-entropy loss", title="Loss by optimization step")
    axes[1].set(xlabel="Wall-clock seconds", ylabel="Cross-entropy loss", title="Loss by wall-clock time")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
