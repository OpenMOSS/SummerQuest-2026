"""复现作业 PDF 4.2.1 的 SGD toy example，并比较不同学习率。"""

from __future__ import annotations

import math
from pathlib import Path

import torch


LEARNING_RATES = (1e1, 1e2, 1e3)
NUM_ITERATIONS = 10
SEED = 336
PLOT_PATH = Path("assets/learning_rate_tuning.svg")


class SGD(torch.optim.Optimizer):
    """PDF 4.2.1 中使用的、按 1/sqrt(t+1) 衰减步长的 SGD。"""

    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, {"lr": lr})

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                iteration = state.get("t", 0)
                parameter.data -= group["lr"] / math.sqrt(iteration + 1) * parameter.grad.data
                state["t"] = iteration + 1
        return loss


def run_experiment(initial_weights: torch.Tensor, learning_rate: float) -> list[float]:
    """从相同初始参数出发，记录 10 次 SGD 更新前的 loss。"""
    weights = torch.nn.Parameter(initial_weights.clone())
    optimizer = SGD([weights], lr=learning_rate)
    losses: list[float] = []

    for _ in range(NUM_ITERATIONS):
        optimizer.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return losses


def write_plot(all_losses: dict[float, list[float]], output_path: Path) -> None:
    """使用标准库生成对数纵轴 SVG 折线图，避免额外绘图库依赖。"""
    width, height = 900, 540
    left, right, top, bottom = 95, 35, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    log_losses = [math.log10(loss) for losses in all_losses.values() for loss in losses]
    min_log_loss = math.floor(min(log_losses))
    max_log_loss = math.ceil(max(log_losses))

    def x_coordinate(step: int) -> float:
        return left + step / (NUM_ITERATIONS - 1) * plot_width

    def y_coordinate(loss: float) -> float:
        return top + (max_log_loss - math.log10(loss)) / (max_log_loss - min_log_loss) * plot_height

    colors = {1e1: "#2563eb", 1e2: "#16a34a", 1e3: "#dc2626"}
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: sans-serif; fill: #1f2937; } .tick { font-size: 13px; }</style>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" font-weight="bold">SGD loss under different learning rates</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
    ]
    for exponent in range(min_log_loss, max_log_loss + 1):
        y = y_coordinate(10**exponent)
        svg.extend(
            [
                f'<line x1="{left}" y1="{y}" x2="{left + plot_width}" y2="{y}" stroke="#d1d5db" stroke-dasharray="4 4"/>',
                f'<text class="tick" x="{left - 10}" y="{y + 5}" text-anchor="end">1e{exponent}</text>',
            ]
        )
    for step in range(NUM_ITERATIONS):
        x = x_coordinate(step)
        svg.extend(
            [
                f'<line x1="{x}" y1="{top + plot_height}" x2="{x}" y2="{top + plot_height + 5}" stroke="#374151"/>',
                f'<text class="tick" x="{x}" y="{top + plot_height + 25}" text-anchor="middle">{step}</text>',
            ]
        )
    for line_index, (learning_rate, losses) in enumerate(all_losses.items()):
        color = colors[learning_rate]
        points = " ".join(f"{x_coordinate(step):.2f},{y_coordinate(loss):.2f}" for step, loss in enumerate(losses))
        legend_x = left + 15 + line_index * 150
        svg.extend(
            [
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>',
                *[
                    f'<circle cx="{x_coordinate(step):.2f}" cy="{y_coordinate(loss):.2f}" r="3.5" fill="{color}"/>'
                    for step, loss in enumerate(losses)
                ],
                f'<line x1="{legend_x}" y1="{top - 18}" x2="{legend_x + 24}" y2="{top - 18}" stroke="{color}" stroke-width="3"/>',
                f'<text class="tick" x="{legend_x + 30}" y="{top - 13}">lr={learning_rate:.0e}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="15">Training iteration</text>',
            f'<text x="22" y="{height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {height / 2})">Loss (log scale)</text>',
            '</svg>',
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    """运行三种学习率，并打印每一步的 loss。"""
    torch.manual_seed(SEED)
    initial_weights = 5 * torch.randn((10, 10))

    print("step  | lr=1e1       | lr=1e2       | lr=1e3")
    print("-" * 50)
    all_losses = {learning_rate: run_experiment(initial_weights, learning_rate) for learning_rate in LEARNING_RATES}
    for step in range(NUM_ITERATIONS):
        print(
            f"{step:>4}  | "
            + " | ".join(f"{all_losses[learning_rate][step]:>12.6g}" for learning_rate in LEARNING_RATES)
        )
    write_plot(all_losses, PLOT_PATH)
    print(f"\n折线图已保存到：{PLOT_PATH}")


if __name__ == "__main__":
    main()
