"""读取学习率 sweep 的 JSONL 日志，生成作业可引用的 SVG 曲线和汇总表。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == "submission":
    ROOT = ROOT.parent
DEFAULT_LOG_ROOT = ROOT / "logs" / "lr_sweep"
DEFAULT_OUTPUT = ROOT / "assets" / "tinystories_lr_sweep.svg"


def load_runs(log_root: Path) -> list[dict[str, object]]:
    """从每个 Hydra run 中提取 peak LR、训练曲线和最终验证损失。"""
    runs = []
    for metrics_path in sorted(log_root.glob("**/metrics.jsonl")):
        config_path = metrics_path.parent / "config.yaml"
        if not config_path.exists():
            continue
        config = OmegaConf.load(config_path)
        peak_lr = float(config.optimizer.max_lr)
        metrics = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        points = [(int(row["step"]), float(row["val_loss"])) for row in metrics if "val_loss" in row]
        summary_path = metrics_path.parent / "summary.json"
        final_loss = None
        if summary_path.exists():
            final_loss = json.loads(summary_path.read_text(encoding="utf-8")).get("final_val_loss")
        runs.append(
            {
                "label": str(config.run_name).split("/")[-1],
                "peak_lr": peak_lr,
                "points": points,
                "final_loss": None if final_loss is None else float(final_loss),
            }
        )
    return sorted(runs, key=lambda run: float(run["peak_lr"]))


def colors(number: int) -> Iterable[str]:
    palette = ("#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#be123c", "#4d7c0f", "#7c3aed")
    for index in range(number):
        yield palette[index % len(palette)]


def write_svg(runs: list[dict[str, object]], output: Path) -> None:
    """以标准库写 SVG，避免为了作业作图额外引入 matplotlib。"""
    plotted = [run for run in runs if run["points"]]
    if not plotted:
        raise ValueError("没有找到包含 val_loss 的 metrics.jsonl；请先完成至少一个 sweep run。")
    width, height = 960, 570
    left, right, top, bottom = 88, 40, 74, 76
    plot_width, plot_height = width - left - right, height - top - bottom
    maximum_step = max(step for run in plotted for step, _ in run["points"])  # type: ignore[misc]
    losses = [loss for run in plotted for _, loss in run["points"]]  # type: ignore[misc]
    low, high = min(losses), max(losses)
    margin = max((high - low) * 0.08, 0.02)
    low, high = low - margin, high + margin

    def x(step: int) -> float:
        return left + step / maximum_step * plot_width

    def y(loss: float) -> float:
        return top + (high - loss) / (high - low) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#1f2937}.tick{font-size:13px}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">TinyStories learning-rate sweep</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
    ]
    for fraction in range(6):
        value = low + (high - low) * fraction / 5
        coordinate = y(value)
        svg += [
            f'<line x1="{left}" y1="{coordinate:.2f}" x2="{left + plot_width}" y2="{coordinate:.2f}" stroke="#e5e7eb"/>',
            f'<text class="tick" x="{left - 10}" y="{coordinate + 4:.2f}" text-anchor="end">{value:.3f}</text>',
        ]
    for fraction in range(6):
        step = round(maximum_step * fraction / 5)
        coordinate = x(step)
        svg += [
            f'<line x1="{coordinate:.2f}" y1="{top + plot_height}" x2="{coordinate:.2f}" y2="{top + plot_height + 5}" stroke="#374151"/>',
            f'<text class="tick" x="{coordinate:.2f}" y="{top + plot_height + 26}" text-anchor="middle">{step}</text>',
        ]
    target_y = y(1.45)
    if top <= target_y <= top + plot_height:
        svg += [
            f'<line x1="{left}" y1="{target_y:.2f}" x2="{left + plot_width}" y2="{target_y:.2f}" stroke="#111827" stroke-dasharray="6 4"/>',
            f'<text class="tick" x="{left + plot_width - 4}" y="{target_y - 6:.2f}" text-anchor="end">target: 1.45</text>',
        ]
    for color, run in zip(colors(len(plotted)), plotted, strict=True):
        points = run["points"]
        point_text = " ".join(f"{x(step):.2f},{y(loss):.2f}" for step, loss in points)  # type: ignore[misc]
        svg.append(f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="3"/>')
        label = f"peak lr={float(run['peak_lr']):.1e}"
        legend_y = top + 20 * list(plotted).index(run)
        svg += [
            f'<line x1="{left + 12}" y1="{legend_y - 5}" x2="{left + 36}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>',
            f'<text class="tick" x="{left + 42}" y="{legend_y}">{label}</text>',
        ]
    svg += [
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="15">optimizer updates</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {height / 2})">validation loss (per token)</text>',
        "</svg>",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runs = load_runs(args.log_root)
    write_svg(runs, args.output)
    print("peak_lr       final_val_loss  status")
    for run in runs:
        final_loss = run["final_loss"]
        status = "complete" if final_loss is not None else "incomplete/diverged"
        formatted_loss = "-" if final_loss is None else f"{float(final_loss):.6f}"
        print(f"{float(run['peak_lr']):<13.1e} {formatted_loss:<15} {status}")
    print(f"\n曲线已写入 {args.output}")


if __name__ == "__main__":
    main()
