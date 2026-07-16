#!/usr/bin/env python3
"""Reproduce the assignment's decaying-SGD learning-rate toy experiment."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable
from pathlib import Path

import torch


class SGD(torch.optim.Optimizer):
    """The exact ``lr / sqrt(t + 1)`` SGD variant from the handout."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                iteration = state.get("t", 0)
                parameter.add_(parameter.grad, alpha=-group["lr"] / math.sqrt(iteration + 1))
                state["t"] = iteration + 1
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learning-rate", type=float, action="append", default=[10.0, 100.0, 1000.0])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(learning_rate: float, steps: int, seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    optimizer = SGD([weights], lr=learning_rate)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    trend = "diverged" if not math.isfinite(losses[-1]) or losses[-1] > losses[0] else "decreased"
    return {"learning_rate": learning_rate, "steps": steps, "losses": losses, "trend": trend}


def main() -> None:
    args = parse_args()
    results = [run(learning_rate, args.steps, args.seed) for learning_rate in args.learning_rate]
    rendered = json.dumps(results, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
