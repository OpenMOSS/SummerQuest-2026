from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT.parent / "SummerQuest-2026" / "students" / "张俊鹏" / "assignments" / "A2-K" / "results"
ASSETS = RESULTS.parent / "assets"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def flash_speedup_plot() -> None:
    rows = [
        row for row in read_csv("flash_benchmark.csv")
        if row["phase"] == "forward_backward" and row["matrix"] == "core"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for axis, head_dim in zip(axes, ("64", "128")):
        subset = [row for row in rows if row["head_dim"] == head_dim]
        for implementation, style in (("compiled", "o-"), ("triton", "s-")):
            series = sorted((row for row in subset if row["implementation"] == implementation), key=lambda row: int(row["sequence_length"]))
            axis.plot(
                [int(row["sequence_length"]) for row in series],
                [float(row["speedup_vs_eager"]) for row in series],
                style,
                label=implementation,
            )
        axis.axhline(1.0, color="black", linewidth=0.8)
        axis.set_xscale("log", base=2)
        axis.set_xticks([512, 2048, 8192])
        axis.set_xticklabels(["512", "2048", "8192"])
        axis.set_title(f"head dim = {head_dim}")
        axis.set_xlabel("sequence length")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("speedup vs eager (p50, forward+backward)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(ASSETS / "attention-speedup.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def checkpoint_tradeoff_plot() -> None:
    rows = [row for row in read_csv("checkpointing.csv") if row["context_length"] == "1024" and row["status"] == "ok"]
    rows.sort(key=lambda row: int(row["checkpoint_block_size"]))
    labels = ["none" if row["checkpoint_block_size"] == "0" else row["checkpoint_block_size"] for row in rows]
    memory = [float(row["peak_allocated_mib"]) / 1024 for row in rows]
    time_ms = [float(row["step_time_ms_p50"]) for row in rows]
    fig, axis_left = plt.subplots(figsize=(6.4, 3.6))
    line_memory = axis_left.plot(labels, memory, "o-", label="peak allocated memory")
    axis_left.set_xlabel("checkpoint block size (layers)")
    axis_left.set_ylabel("peak allocated memory (GiB)")
    axis_left.grid(axis="y", alpha=0.25)
    axis_right = axis_left.twinx()
    line_time = axis_right.plot(labels, time_ms, "s-", color="tab:orange", label="training-step p50")
    axis_right.set_ylabel("training-step p50 (ms)")
    axis_left.legend(line_memory + line_time, [line.get_label() for line in line_memory + line_time], frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(ASSETS / "checkpointing-tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    flash_speedup_plot()
    checkpoint_tradeoff_plot()
    for path in sorted(ASSETS.glob("*.png")):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
