from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Iterable

import torch


class SGD(torch.optim.Optimizer):
    """The decaying-learning-rate SGD optimizer from Assignment 1 section 4.2."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None if closure is None else closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue

                    state = self.state[parameter]
                    iteration = state.get("t", 0)
                    effective_lr = lr / math.sqrt(iteration + 1)
                    parameter.add_(parameter.grad, alpha=-effective_lr)
                    state["t"] = iteration + 1

        return loss


def run_experiment(
    initial_weights: torch.Tensor,
    learning_rate: float,
    steps: int,
) -> dict[str, object]:
    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=learning_rate)
    losses_before_update: list[float] = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = weights.square().mean()
        losses_before_update.append(loss.item())
        loss.backward()
        optimizer.step()

    final_loss = weights.square().mean().item()
    return {
        "learning_rate": learning_rate,
        "steps": steps,
        "losses_before_update": losses_before_update,
        "loss_after_final_update": final_loss,
        "all_losses_finite": all(math.isfinite(loss) for loss in losses_before_update)
        and math.isfinite(final_loss),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the section 4.2 SGD learning-rate experiment."
    )
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    initial_weights = 5 * torch.randn((10, 10))
    results = {
        "experiment": "sgd_learning_rate_tuning",
        "seed": args.seed,
        "objective": "mean(weights ** 2)",
        "initial_loss": initial_weights.square().mean().item(),
        "runs": [
            run_experiment(initial_weights, learning_rate, args.steps)
            for learning_rate in (1e1, 1e2, 1e3)
        ],
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
