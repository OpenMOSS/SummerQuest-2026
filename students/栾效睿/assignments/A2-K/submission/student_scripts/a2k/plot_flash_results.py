from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "flash_benchmark.csv"
ASSETS = ROOT / "assets"
DISPLAY_NAMES = {
    "explicit_pytorch_eager": "eager PyTorch",
    "compiled_pytorch": "compiled PyTorch",
    "triton_flashattention": "Triton FlashAttention",
}
ANNOTATION_OFFSETS = {
    "explicit_pytorch_eager": (10, 12),
    "compiled_pytorch": (10, -24),
    "triton_flashattention": (10, 18),
}


def rows() -> list[dict[str, str]]:
    with INPUT.open(newline="", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file) if row["status"] == "success"]


def annotate_max(ax, grouped: dict[str, list[tuple[int, float]]], unit: str, decimals: int) -> None:
    for implementation, values in grouped.items():
        sequence_length, maximum = max(values, key=lambda item: item[1])
        value = f"{maximum:.{decimals}f}"
        ax.annotate(
            f"{DISPLAY_NAMES[implementation]} max\n{value} {unit} @ S={sequence_length}",
            xy=(sequence_length, maximum),
            xytext=ANNOTATION_OFFSETS[implementation],
            textcoords="offset points",
            fontsize=8,
            color="black",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "0.35", "shrinkA": 0, "shrinkB": 3},
        )


def plot_latency(data: list[dict[str, str]]) -> None:
    grouped = defaultdict(list)
    for row in data:
        if row["phase"] == "forward" and row["head_dim"] == "64":
            grouped[row["implementation"]].append((int(row["sequence_length"]), float(row["latency_ms_p50"])))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for implementation, values in grouped.items():
        values.sort()
        ax.plot([x for x, _ in values], [y for _, y in values], marker="o", label=DISPLAY_NAMES[implementation])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("forward p50 latency (ms, log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    annotate_max(ax, grouped, "ms", 3)
    fig.tight_layout()
    fig.savefig(ASSETS / "flash_latency.png", dpi=160)
    plt.close(fig)


def plot_memory(data: list[dict[str, str]]) -> None:
    grouped = defaultdict(list)
    for row in data:
        if row["phase"] == "forward" and row["head_dim"] == "64":
            grouped[row["implementation"]].append((int(row["sequence_length"]), float(row["peak_allocated_mib"])))
    fig, ax = plt.subplots(figsize=(10, 7.5))
    for implementation, values in grouped.items():
        values.sort()
        ax.plot([x for x, _ in values], [y for _, y in values], marker="o", label=DISPLAY_NAMES[implementation])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("forward peak allocated (MiB, log scale)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    annotate_max(ax, grouped, "MiB", 1)
    fig.tight_layout()
    fig.savefig(ASSETS / "flash_memory.png", dpi=160)
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    data = rows()
    if not data:
        raise RuntimeError(f"No successful rows found in {INPUT}")
    plot_latency(data)
    plot_memory(data)
    print(f"wrote {ASSETS / 'flash_latency.png'} and {ASSETS / 'flash_memory.png'}")


if __name__ == "__main__":
    main()
