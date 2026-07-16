from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


class SGD(torch.optim.Optimizer):
    """The inverse-square-root-decay SGD optimizer from the A1 handout."""

    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                t = state.get("t", 0)
                parameter.add_(parameter.grad, alpha=-lr / math.sqrt(t + 1))
                state["t"] = t + 1
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the A1 toy SGD learning-rate experiment.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(args.seed)
    initial_weights = 5 * torch.randn((10, 10), generator=generator)
    results: dict[str, object] = {"seed": args.seed, "steps": args.steps, "runs": {}}
    for learning_rate in (1e1, 1e2, 1e3):
        weights = torch.nn.Parameter(initial_weights.clone())
        optimizer = SGD([weights], lr=learning_rate)
        losses = []
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss = (weights**2).mean()
            losses.append(float(loss.item()))
            loss.backward()
            optimizer.step()
        results["runs"][f"{learning_rate:.0e}"] = {
            "learning_rate": learning_rate,
            "losses": losses,
            "behavior": "decreasing" if losses[-1] < losses[0] else "diverging",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
