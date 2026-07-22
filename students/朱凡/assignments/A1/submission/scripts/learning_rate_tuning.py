"""CS336 作业 1 · 4.2 节 `learning_rate_tuning` 实验.

用自己实现的 SGD 在一个简单回归任务上跑 10 步,学习率分别为 1e1、1e2、1e3,
观察 loss 是下降、变慢还是发散。SGD 更新规则:`theta -= lr / sqrt(t+1) * grad`。
"""

from __future__ import annotations

import torch

from cs336_basics.model import Linear
from cs336_basics.optimizer import SGD


def run_single(lr: float, steps: int = 10, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    model = Linear(3, 2)
    opt = SGD(model.parameters(), lr=lr)

    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        x = torch.rand(3)
        y_hat = model(x)
        y = torch.tensor([x[0] + x[1], -x[2]])
        loss = ((y - y_hat) ** 2).sum()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


def main() -> None:
    print("=== learning_rate_tuning (SGD, 10 steps) ===")
    print(f"{'step':>4} | {'lr=1e1':>12} | {'lr=1e2':>12} | {'lr=1e3':>12}")
    runs = {lr: run_single(lr) for lr in (1e1, 1e2, 1e3)}
    for step in range(10):
        print(f"{step + 1:>4} | {runs[1e1][step]:>12.4f} | {runs[1e2][step]:>12.4f} | {runs[1e3][step]:>12.4f}")

    print()
    for lr in (1e1, 1e2, 1e3):
        first, last = runs[lr][0], runs[lr][-1]
        if not torch.isfinite(torch.tensor(last)):
            verdict = "发散 (NaN/Inf)"
        elif last < first:
            verdict = "下降"
        elif last > first * 10:
            verdict = "发散 (爆炸)"
        else:
            verdict = "变慢/停滞"
        print(f"lr={lr:.0e}: loss {first:.4f} -> {last:.4f}  =>  {verdict}")


if __name__ == "__main__":
    main()
