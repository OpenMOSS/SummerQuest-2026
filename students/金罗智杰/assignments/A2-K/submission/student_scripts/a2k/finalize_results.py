"""Build lightweight A2-K metadata, memory evidence, and report figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from student_scripts.a2k.common import ALLOCATOR_LIMIT_MIB, HARD_LIMIT_MIB, write_json


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def successful(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row["status"] == "success"]


def maximum(rows: list[dict[str, str]], field: str) -> float:
    values = [float(row[field]) for row in successful(rows) if row.get(field)]
    return max(values, default=0.0)


def build_memory_evidence(result_dir: Path, tables: dict[str, list[dict[str, str]]]) -> None:
    environment = read_json(result_dir / "flash_benchmark.metadata.json")["environment"]
    experiment_peaks = {
        name: {
            "peak_allocated_mib": maximum(rows, "peak_allocated_mib"),
            "peak_reserved_mib": maximum(rows, "peak_reserved_mib"),
        }
        for name, rows in tables.items()
    }
    highest_allocated = max(value["peak_allocated_mib"] for value in experiment_peaks.values())
    highest_reserved = max(value["peak_reserved_mib"] for value in experiment_peaks.values())
    payload = {
        "allocator": {
            "allocator_fraction": environment["allocator_fraction"],
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
            "applied_before_first_cuda_tensor_allocation": True,
        },
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": highest_allocated,
        "pytorch_peak_reserved_mib": highest_reserved,
        "within_24gib": highest_reserved <= HARD_LIMIT_MIB,
        "per_experiment_maximum": experiment_peaks,
    }
    write_json(result_dir / "memory_evidence.json", payload)


def build_run_metadata(result_dir: Path) -> None:
    metadata_files = (
        "checkpointing.metadata.json",
        "attention_baseline.metadata.json",
        "compile_comparison.metadata.json",
        "flash_benchmark.metadata.json",
    )
    metadata = [read_json(result_dir / filename) for filename in metadata_files]
    environments = [item["environment"] for item in metadata]
    stable_fields = (
        "gpu_name",
        "memory_total_mib",
        "driver_version",
        "cuda_runtime",
        "torch_version",
        "triton_version",
        "power_limit_w",
        "pstate",
        "allocator_limit_mib",
        "tf32_matmul",
        "tf32_cudnn",
    )
    environment = {field: environments[0][field] for field in stable_fields}
    environment["minimum_starting_free_memory_mib"] = min(item["memory_free_mib"] for item in environments)
    environment["allocator_fraction_samples"] = sorted(
        {round(item["allocator_fraction"], 12) for item in environments}
    )
    payload = {
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "seed": 42,
        "environment": environment,
        "measurement": {
            "checkpointing": {"warmup_steps": 3, "measurement_steps": 5},
            "attention": {"timer": "triton.testing.do_bench", "warmup_ms": 100, "rep_ms": 300},
            "attention_quantiles": [0.2, 0.5, 0.8],
            "synchronization": "CUDA synchronization at cold-start and step timing boundaries",
            "compile_config": {"torch_functorch_donated_buffer": False},
            "tf32": {
                "performance": "enabled",
                "fp32_correctness": "disabled with matmul.fp32_precision=ieee",
                "bf16_correctness": "not applicable",
            },
        },
        "commands": [
            "python student_scripts/a2k/run_official_attention_tests.py",
            "python student_scripts/a2k/run_correctness.py",
            "python student_scripts/a2k/benchmark_checkpointing.py",
            "python student_scripts/a2k/benchmark_attention_baseline.py",
            "python student_scripts/a2k/benchmark_compile.py",
            "python student_scripts/a2k/benchmark_flash.py",
        ],
    }
    write_json(result_dir / "run_metadata.json", payload)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_checkpoint_tradeoff(result_dir: Path, rows: list[dict[str, str]]) -> None:
    selected = [row for row in rows if row["context_length"] == "1024" and row["status"] == "success"]
    labels = [
        "none" if row["checkpoint_block_size"] == "none" else f"block {row['checkpoint_block_size']}"
        for row in selected
    ]
    latency = [float(row["step_time_ms_p50"]) for row in selected]
    memory = [float(row["peak_allocated_mib"]) / 1024 for row in selected]

    figure, left = plt.subplots(figsize=(8.2, 4.8))
    positions = list(range(len(labels)))
    left.bar([position - 0.18 for position in positions], memory, width=0.36, color="#4472C4", label="Peak allocated (GiB)")
    left.set_ylabel("Peak allocated memory (GiB)")
    left.set_xticks(positions, labels)
    left.set_xlabel("Checkpoint configuration (context 1024)")
    right = left.twinx()
    right.bar([position + 0.18 for position in positions], latency, width=0.36, color="#ED7D31", label="Step p50 (ms)")
    right.set_ylabel("Training-step p50 (ms)")
    handles_left, labels_left = left.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")
    left.set_title("Activation checkpointing: memory–compute trade-off")
    save_figure(result_dir / "assets" / "checkpoint_tradeoff.png")


def plot_flash_forward_latency(result_dir: Path, rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    colors = {"eager_pytorch": "#4472C4", "compiled_pytorch": "#70AD47", "triton": "#ED7D31"}
    labels = {"eager_pytorch": "Eager PyTorch", "compiled_pytorch": "Compiled PyTorch", "triton": "Student Triton"}
    for axis, head_dim in zip(axes, (64, 128), strict=True):
        for implementation in ("eager_pytorch", "compiled_pytorch", "triton"):
            selected = [
                row
                for row in rows
                if row["phase"] == "forward"
                and row["head_dim"] == str(head_dim)
                and row["implementation"] == implementation
                and row["status"] == "success"
            ]
            selected.sort(key=lambda row: int(row["sequence_length"]))
            if selected:
                axis.plot(
                    [int(row["sequence_length"]) for row in selected],
                    [float(row["latency_p50_ms"]) for row in selected],
                    marker="o",
                    color=colors[implementation],
                    label=labels[implementation],
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Sequence length")
        axis.set_title(f"head_dim = {head_dim}")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Forward p50 latency (ms)")
    axes[1].legend()
    figure.suptitle("Causal BF16 attention forward latency")
    save_figure(result_dir / "assets" / "flash_forward_latency.png")


def plot_flash_memory(result_dir: Path, rows: list[dict[str, str]]) -> None:
    selected = [
        row
        for row in rows
        if row["phase"] == "forward_backward"
        and row["head_dim"] == "128"
        and row["implementation"] in {"eager_pytorch", "triton"}
        and row["status"] == "success"
    ]
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    for implementation, label, color in (
        ("eager_pytorch", "Eager PyTorch", "#4472C4"),
        ("triton", "Student Triton", "#ED7D31"),
    ):
        part = sorted(
            (row for row in selected if row["implementation"] == implementation),
            key=lambda row: int(row["sequence_length"]),
        )
        axis.plot(
            [int(row["sequence_length"]) for row in part],
            [float(row["peak_allocated_mib"]) for row in part],
            marker="o",
            label=label,
            color=color,
        )
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Peak allocated memory (MiB)")
    axis.set_title("Forward-backward memory, BF16 causal attention (head_dim=128)")
    axis.grid(alpha=0.25)
    axis.legend()
    save_figure(result_dir / "assets" / "flash_memory.png")


def main() -> None:
    result_dir = Path("local_results/a2k")
    tables = {
        "checkpointing": read_csv(result_dir / "checkpointing.csv"),
        "attention_baseline": read_csv(result_dir / "attention_baseline.csv"),
        "compile_comparison": read_csv(result_dir / "compile_comparison.csv"),
        "flash_benchmark": read_csv(result_dir / "flash_benchmark.csv"),
    }
    build_memory_evidence(result_dir, tables)
    build_run_metadata(result_dir)
    plot_checkpoint_tradeoff(result_dir, tables["checkpointing"])
    plot_flash_forward_latency(result_dir, tables["flash_benchmark"])
    plot_flash_memory(result_dir, tables["flash_benchmark"])
    print("saved memory evidence, run metadata, and 3 figures")


if __name__ == "__main__":
    main()
