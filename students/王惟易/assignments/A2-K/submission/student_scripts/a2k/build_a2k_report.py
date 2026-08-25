from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public A2-K metadata and report figures.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_successful_rows(name: str, rows: list[dict[str, str]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"{name}: expected {expected_count} rows, found {len(rows)}")
    failures = [row["config_id"] for row in rows if row["status"] != "ok"]
    if failures:
        raise ValueError(f"{name}: unsuccessful rows: {failures}")


def safe_environment(correctness: dict[str, Any]) -> dict[str, Any]:
    environment = correctness["environment"]
    keys = (
        "gpu",
        "gpu_memory_total_mib",
        "gpu_memory_free_mib_at_start",
        "driver_version",
        "power_limit_w",
        "pstate",
        "python",
        "pytorch",
        "triton",
        "cuda_runtime",
        "cudnn_version",
        "compute_capability",
        "visible_cuda_devices",
        "device_reported_total_mib",
        "tf32_matmul_allowed",
        "float32_matmul_precision",
    )
    return {key: environment[key] for key in keys}


def build_run_metadata(
    result_dir: Path,
    checkpoint_rows: list[dict[str, str]],
    baseline_rows: list[dict[str, str]],
    compile_rows: list[dict[str, str]],
    flash_rows: list[dict[str, str]],
    correctness: dict[str, Any],
) -> None:
    allocator = correctness["allocator"]
    metadata = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_control": {
            "starter_commit": STARTER_COMMIT,
            "working_tree_head_at_measurement": STARTER_COMMIT,
        },
        "environment": safe_environment(correctness),
        "hardware_scope": {
            "single_visible_gpu": True,
            "platform_reported_memory_mib": correctness["environment"]["gpu_memory_total_mib"],
            "minimum_free_memory_mib": 22 * 1024,
            "larger_instance_used_with_fixed_allocator_cap": True,
        },
        "allocator": allocator,
        "numerics": {
            "performance_dtype": "bf16",
            "performance_is_causal": True,
            "performance_seed": 0,
            "correctness_seeds": correctness["config"]["seeds"],
            "tf32_matmul_allowed": correctness["environment"]["tf32_matmul_allowed"],
            "float32_matmul_precision": correctness["environment"]["float32_matmul_precision"],
            "triton_fp32_dot_input_precision": "ieee",
        },
        "measurement_protocols": {
            "checkpointing": {
                "warmup_steps": 3,
                "measurement_steps": 5,
                "timer": "synchronized perf_counter",
                "process_scope": "one configuration per Python process",
            },
            "attention": {
                "timer": "triton.testing.do_bench",
                "warmup_ms": 100,
                "rep_ms": 300,
                "quantiles": [0.2, 0.5, 0.8],
                "process_scope": "one configuration per Python process",
            },
            "compile": {
                "fullgraph": True,
                "cold_start_definition": "first invocation in a new Python process",
                "disk_cache_cleared_between_processes": False,
                "steady_state_timer": "triton.testing.do_bench",
            },
            "memory": {
                "peak_scope": "one steady-state phase invocation, or the full measured checkpoint step",
                "statistics": ["torch.cuda.max_memory_allocated", "torch.cuda.max_memory_reserved"],
            },
        },
        "result_counts": {
            "checkpointing_rows": len(checkpoint_rows),
            "attention_baseline_rows": len(baseline_rows),
            "compile_comparison_rows": len(compile_rows),
            "flash_benchmark_rows": len(flash_rows),
            "correctness_cases": correctness["summary"]["total_cases"],
            "correctness_passed": correctness["summary"]["passed_cases"],
            "official_tests_passed": 6,
            "official_tests_failed": 0,
            "official_tests_skipped": 0,
        },
        "commands": [
            "CUDA_VISIBLE_DEVICES=0 uv run pytest tests/test_attention.py -v --no-header --tb=short",
            "CUDA_VISIBLE_DEVICES=0 uv run python student_scripts/a2k/run_checkpoint_matrix.py --output-dir /tmp/a2k-checkpointing",
            "CUDA_VISIBLE_DEVICES=0 uv run python student_scripts/a2k/run_attention_baseline_matrix.py --output-dir /tmp/a2k-attention-baseline",
            "CUDA_VISIBLE_DEVICES=0 uv run python student_scripts/a2k/run_compile_comparison.py --output-dir /tmp/a2k-compile-comparison",
            "CUDA_VISIBLE_DEVICES=0 uv run python student_scripts/a2k/flash_correctness.py --output /tmp/a2k-correctness.json",
            "CUDA_VISIBLE_DEVICES=0 uv run python student_scripts/a2k/run_flash_benchmark_matrix.py --output-dir /tmp/a2k-flash-benchmark",
        ],
        "public_results": [
            "checkpointing.csv",
            "attention_baseline.csv",
            "compile_comparison.csv",
            "correctness.json",
            "flash_benchmark.csv",
            "unit_tests.txt",
        ],
    }
    write_json(result_dir / "run_metadata.json", metadata)


def peak_record(rows: list[dict[str, str]], field: str) -> dict[str, Any]:
    row = max(rows, key=lambda item: float(item[field]))
    return {
        "value_mib": float(row[field]),
        "config_id": row["config_id"],
    }


def build_memory_evidence(
    result_dir: Path,
    checkpoint_rows: list[dict[str, str]],
    compile_rows: list[dict[str, str]],
    flash_rows: list[dict[str, str]],
    correctness: dict[str, Any],
) -> None:
    matrices = {
        "checkpointing.csv": checkpoint_rows,
        "compile_comparison.csv": compile_rows,
        "flash_benchmark.csv": flash_rows,
    }
    all_rows = [row for rows in matrices.values() for row in rows]
    max_allocated = peak_record(all_rows, "peak_allocated_mib")
    max_reserved = peak_record(all_rows, "peak_reserved_mib")
    matrix_maxima = {
        name: {
            "rows": len(rows),
            "peak_allocated": peak_record(rows, "peak_allocated_mib"),
            "peak_reserved": peak_record(rows, "peak_reserved_mib"),
        }
        for name, rows in matrices.items()
    }
    evidence = {
        "schema_version": 1,
        "allocator": correctness["allocator"],
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": max_allocated["value_mib"],
        "pytorch_peak_reserved_mib": max_reserved["value_mib"],
        "within_24gib": max_reserved["value_mib"] <= HARD_LIMIT_MIB,
        "within_allocator_limit": max_reserved["value_mib"] <= ALLOCATOR_LIMIT_MIB,
        "peak_allocated_source": max_allocated["config_id"],
        "peak_reserved_source": max_reserved["config_id"],
        "performance_process_rows": len(all_rows),
        "all_rows_successful": all(row["status"] == "ok" for row in all_rows),
        "all_rows_report_within_24gib": all(row["within_24gib"] == "true" for row in all_rows),
        "attention_baseline_note": "The 18 baseline rows are exact views of eager rows already counted in flash_benchmark.csv.",
        "matrix_maxima": matrix_maxima,
    }
    write_json(result_dir / "memory_evidence.json", evidence)


def configure_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "font.size": 10,
            "legend.frameon": False,
            "savefig.bbox": "tight",
        }
    )


def save_svg(figure: Any, path: Path) -> None:
    figure.savefig(path, format="svg")
    svg = path.read_text(encoding="utf-8")
    path.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


def plot_flash_performance(asset_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    colors = {"eager": "#4C78A8", "compiled": "#F58518", "triton": "#54A24B"}
    markers = {64: "o", 128: "s"}
    figure, (latency_axis, speedup_axis) = plt.subplots(1, 2, figsize=(11, 4.2))

    for implementation in ("eager", "compiled", "triton"):
        for head_dim in (64, 128):
            series = sorted(
                (row for row in rows if row["implementation"] == implementation and row["phase"] == "forward" and int(row["head_dim"]) == head_dim),
                key=lambda row: int(row["sequence_length"]),
            )
            if not series:
                continue
            label = f"{implementation}, D={head_dim}"
            latency_axis.plot(
                [int(row["sequence_length"]) for row in series],
                [float(row["p50_ms"]) for row in series],
                color=colors[implementation],
                marker=markers[head_dim],
                linestyle="-" if head_dim == 64 else "--",
                label=label,
            )

    for implementation in ("compiled", "triton"):
        for head_dim in (64, 128):
            series = sorted(
                (row for row in rows if row["implementation"] == implementation and row["phase"] == "forward_backward" and int(row["head_dim"]) == head_dim),
                key=lambda row: int(row["sequence_length"]),
            )
            if not series:
                continue
            speedup_axis.plot(
                [int(row["sequence_length"]) for row in series],
                [float(row["speedup_vs_eager"]) for row in series],
                color=colors[implementation],
                marker=markers[head_dim],
                linestyle="-" if head_dim == 64 else "--",
                label=f"{implementation}, D={head_dim}",
            )

    latency_axis.set_xscale("log", base=2)
    latency_axis.set_yscale("log")
    latency_axis.set_xticks([512, 2048, 8192, 16384], labels=["512", "2K", "8K", "16K"])
    latency_axis.set_xlabel("Sequence length")
    latency_axis.set_ylabel("Forward p50 latency (ms)")
    latency_axis.set_title("Forward latency")
    latency_axis.grid(alpha=0.25)
    latency_axis.legend(fontsize=8, ncol=2)

    speedup_axis.axhline(1.0, color="#777777", linewidth=1, linestyle=":", label="eager baseline")
    speedup_axis.set_xscale("log", base=2)
    speedup_axis.set_xticks([512, 2048, 8192, 16384], labels=["512", "2K", "8K", "16K"])
    speedup_axis.set_xlabel("Sequence length")
    speedup_axis.set_ylabel("Forward-backward speedup vs eager")
    speedup_axis.set_title("End-to-end crossover")
    speedup_axis.grid(alpha=0.25)
    speedup_axis.legend(fontsize=8, ncol=2)

    figure.suptitle("A2-K FlashAttention performance (BF16, causal, batch 1)")
    figure.tight_layout()
    save_svg(figure, asset_dir / "flash-performance.svg")
    plt.close(figure)


def plot_checkpoint_tradeoff(asset_dir: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    colors = {1024: "#4C78A8", 2048: "#E45756"}
    label_offsets = {
        (1024, "none"): (6, 5),
        (1024, "1"): (-10, 10),
        (1024, "2"): (-18, -13),
        (1024, "4"): (2, 9),
        (1024, "8"): (8, -6),
        (2048, "none"): (6, 5),
        (2048, "1"): (6, 5),
    }
    figure, (tradeoff_axis, budget_axis) = plt.subplots(1, 2, figsize=(11, 4.2))

    for context_length in (1024, 2048):
        series = [row for row in rows if int(row["context_length"]) == context_length]
        tradeoff_axis.scatter(
            [float(row["peak_allocated_mib"]) / 1024 for row in series],
            [float(row["step_time_ms_p50"]) for row in series],
            color=colors[context_length],
            s=55,
            label=f"T={context_length}",
        )
        for row in series:
            label = "none" if row["checkpoint_block_size"] == "none" else f"K={row['checkpoint_block_size']}"
            text_offset = label_offsets[(context_length, row["checkpoint_block_size"])]
            tradeoff_axis.annotate(
                label,
                (float(row["peak_allocated_mib"]) / 1024, float(row["step_time_ms_p50"])),
                xytext=text_offset,
                textcoords="offset points",
                fontsize=8,
            )

    ordered = sorted(rows, key=lambda row: (int(row["context_length"]), row["checkpoint_block_size"] != "none", row["checkpoint_block_size"]))
    labels = [f"T={row['context_length']}\n{'none' if row['checkpoint_block_size'] == 'none' else 'K=' + row['checkpoint_block_size']}" for row in ordered]
    percentages = [100 * float(row["peak_reserved_mib"]) / ALLOCATOR_LIMIT_MIB for row in ordered]
    budget_axis.bar(
        range(len(ordered)),
        percentages,
        color=[colors[int(row["context_length"])] for row in ordered],
    )
    budget_axis.axhline(100, color="#777777", linestyle=":", linewidth=1.2, label="23 GiB allocator cap")
    budget_axis.set_xticks(range(len(ordered)), labels=labels, rotation=35, ha="right")
    budget_axis.set_ylabel("Peak reserved / allocator cap (%)")
    budget_axis.set_ylim(0, 105)
    budget_axis.set_title("Reserved-memory budget")
    budget_axis.grid(axis="y", alpha=0.25)
    budget_axis.legend(fontsize=8)

    tradeoff_axis.set_xlabel("Peak allocated memory (GiB)")
    tradeoff_axis.set_ylabel("Training-step p50 (ms)")
    tradeoff_axis.set_title("Checkpoint time-memory tradeoff")
    tradeoff_axis.grid(alpha=0.25)
    tradeoff_axis.legend(fontsize=8)

    figure.suptitle("A2-K activation checkpointing (Stanford medium, BF16)")
    figure.tight_layout()
    save_svg(figure, asset_dir / "checkpoint-tradeoff.svg")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.asset_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows = read_csv(args.result_dir / "checkpointing.csv")
    baseline_rows = read_csv(args.result_dir / "attention_baseline.csv")
    compile_rows = read_csv(args.result_dir / "compile_comparison.csv")
    flash_rows = read_csv(args.result_dir / "flash_benchmark.csv")
    correctness = read_json(args.result_dir / "correctness.json")

    require_successful_rows("checkpointing", checkpoint_rows, 7)
    require_successful_rows("attention baseline", baseline_rows, 18)
    require_successful_rows("compile comparison", compile_rows, 24)
    require_successful_rows("flash benchmark", flash_rows, 66)
    if correctness["summary"] != {
        "total_cases": 38,
        "passed_cases": 38,
        "failed_cases": 0,
        "status": "ok",
    }:
        raise ValueError(f"unexpected correctness summary: {correctness['summary']}")

    build_run_metadata(args.result_dir, checkpoint_rows, baseline_rows, compile_rows, flash_rows, correctness)
    build_memory_evidence(args.result_dir, checkpoint_rows, compile_rows, flash_rows, correctness)
    configure_plot_style()
    plot_flash_performance(args.asset_dir, flash_rows)
    plot_checkpoint_tradeoff(args.asset_dir, checkpoint_rows)

    print(f"Wrote metadata to {args.result_dir}")
    print(f"Wrote figures to {args.asset_dir}")


if __name__ == "__main__":
    main()
