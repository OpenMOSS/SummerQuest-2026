#!/usr/bin/env python3
"""Reproduce the assignment's ten-step decayed-SGD learning-rate example."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from _common import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--learning-rate",
        action="append",
        type=float,
        dest="learning_rates",
        help="initial learning rate; repeat to compare several values",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="optional JSON destination")
    return parser.parse_args()


def run(initial_learning_rate: float, iterations: int, seed: int) -> dict[str, object]:
    if initial_learning_rate < 0:
        raise ValueError("learning rates must be non-negative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    losses: list[float] = []
    for step in range(iterations):
        weights.grad = None
        loss = (weights**2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        with torch.no_grad():
            weights.add_(weights.grad, alpha=-initial_learning_rate / math.sqrt(step + 1))

    return {
        "initial_learning_rate": initial_learning_rate,
        "iterations": iterations,
        "loss_before_each_update": losses,
        "initial_loss": losses[0],
        "tenth_forward_loss": losses[9] if iterations >= 10 else None,
        "final_forward_loss": losses[-1],
    }


def main() -> None:
    args = parse_args()
    learning_rates = args.learning_rates or [1.0, 10.0, 100.0, 1000.0]
    payload = {
        "format": "cs336-toy-sgd-v1",
        "seed": args.seed,
        "runs": [run(learning_rate, args.iterations, args.seed) for learning_rate in learning_rates],
    }
    if args.output is not None:
        atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
