from __future__ import annotations

import argparse
import csv
import json
import pickle
import sqlite3
import statistics
from pathlib import Path
from typing import Any


BENCHMARK_RUNS = (
    "timings_small_forward_fp32",
    "timings_small_forward_backward_fp32",
    "timings_small_train_step_fp32",
    "timings_small_train_step_fp32_warmup0",
)
PROFILE_MODELS = ("small", "large")
PROFILE_CONTEXTS = (256, 512, 1024)
PROFILE_RANGES = (
    "profile/measure",
    "measurement/step_1",
    "forward",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
)
PHASE_COLUMNS = (
    "zero_grad_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
    "total_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export compact, public A2-P evidence from local profiler results."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def compact_environment(environment: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "python_version",
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_name",
        "gpu_compute_capability",
        "gpu_total_memory_bytes",
    )
    return {key: environment[key] for key in allowed if key in environment}


def sample_summary(values: list[float]) -> dict[str, float | int]:
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean_ms": mean,
        "sample_std_ms": standard_deviation,
        "cv": standard_deviation / mean if mean else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
    }


def export_benchmark(results_dir: Path, output_dir: Path) -> None:
    output_rows: list[dict[str, Any]] = []
    for run_name in BENCHMARK_RUNS:
        timing_rows = read_csv(results_dir / f"{run_name}.csv")
        metadata = load_json(results_dir / f"{run_name}.metadata.json")
        config = metadata["config"]
        common = {
            "run_name": run_name,
            "model_size": config["model_size"],
            "mode": config["mode"],
            "dtype": config["dtype"],
            "batch_size": config["batch_size"],
            "context_length": config["context_length"],
            "warmup": config["warmup"],
            "steps": config["steps"],
        }
        for phase_column in PHASE_COLUMNS:
            values = [
                float(row[phase_column])
                for row in timing_rows
                if row.get(phase_column) not in (None, "")
            ]
            if not values:
                continue
            phase = phase_column.removesuffix("_ms")
            for row, value in zip(
                (row for row in timing_rows if row.get(phase_column) not in (None, "")),
                values,
                strict=True,
            ):
                output_rows.append(
                    {
                        **common,
                        "record_type": "measurement",
                        "measurement": row["measurement"],
                        "phase": phase,
                        "value_ms": f"{value:.9f}",
                        "count": "",
                        "mean_ms": "",
                        "sample_std_ms": "",
                        "cv": "",
                        "min_ms": "",
                        "max_ms": "",
                        "peak_allocated_mib": "",
                        "peak_reserved_mib": "",
                        "command": "",
                    }
                )
            summary = sample_summary(values)
            output_rows.append(
                {
                    **common,
                    "record_type": "summary",
                    "measurement": "",
                    "phase": phase,
                    "value_ms": "",
                    **{key: f"{value:.9f}" if isinstance(value, float) else value for key, value in summary.items()},
                    "peak_allocated_mib": (
                        f"{metadata['statistics']['peak_memory_allocated_bytes'] / 1024**2:.3f}"
                        if phase == "total"
                        else ""
                    ),
                    "peak_reserved_mib": (
                        f"{metadata['statistics']['peak_memory_reserved_bytes'] / 1024**2:.3f}"
                        if phase == "total"
                        else ""
                    ),
                    "command": metadata["command"] if phase == "total" else "",
                }
            )

    fieldnames = [
        "run_name",
        "record_type",
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "warmup",
        "steps",
        "measurement",
        "phase",
        "value_ms",
        "count",
        "mean_ms",
        "sample_std_ms",
        "cv",
        "min_ms",
        "max_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "command",
    ]
    write_csv(output_dir / "benchmark.csv", fieldnames, output_rows)


def find_nvtx_rows(path: Path) -> list[dict[str, str]]:
    selected = []
    for row in read_csv(path):
        range_name = row["Range"].lstrip(":")
        if range_name in PROFILE_RANGES:
            selected.append(row)
    return selected


def shorten_name(name: str, limit: int = 180) -> str:
    return name if len(name) <= limit else f"{name[: limit - 1]}…"


def profile_command(run_name: str, model: str, context: int) -> str:
    return (
        "nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx "
        "--pytorch=autograd-shapes-nvtx --capture-range=nvtx "
        "--capture-range-end=stop --nvtx-capture='profile/measure@*' "
        "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 "
        f"--output=results/nsys/{run_name} -- python profiling/benchmark.py "
        f"--model-size {model} --batch-size 4 --context-length {context} "
        "--mode train_step --warmup 5 --steps 1 --dtype fp32 "
        "--nvtx-attention "
        f"--output results/nsys/{run_name}.timings.csv"
    )


def export_profile(results_dir: Path, output_dir: Path) -> None:
    nsys_dir = results_dir / "nsys"
    summary_rows: list[dict[str, Any]] = []
    runs = []
    for model in PROFILE_MODELS:
        for context in PROFILE_CONTEXTS:
            run_name = f"{model}_ctx{context}_train_step_fp32"
            metadata = load_json(nsys_dir / f"{run_name}.timings.metadata.json")
            common = {
                "run_name": run_name,
                "model_size": model,
                "context_length": context,
                "mode": "train_step",
                "dtype": "fp32",
                "tool": "Nsight Systems",
            }

            for row in find_nvtx_rows(nsys_dir / f"{run_name}_nvtx_sum.csv"):
                range_name = row["Range"].lstrip(":")
                summary_rows.append(
                    {
                        **common,
                        "record_type": "nvtx_range",
                        "phase_range": range_name,
                        "name": range_name,
                        "calls": row["Instances"],
                        "range_total_ms": f"{int(row['Total Time (ns)']) / 1e6:.6f}",
                        "cpu_total_ms": "",
                        "cuda_total_ms": "",
                        "time_percent": row["Time (%)"],
                    }
                )

            for row in read_csv(nsys_dir / f"{run_name}_cuda_api_sum.csv")[:10]:
                summary_rows.append(
                    {
                        **common,
                        "record_type": "cuda_api",
                        "phase_range": "profile/measure",
                        "name": row["Name"],
                        "calls": row["Num Calls"],
                        "range_total_ms": "",
                        "cpu_total_ms": f"{int(row['Total Time (ns)']) / 1e6:.6f}",
                        "cuda_total_ms": "",
                        "time_percent": row["Time (%)"],
                    }
                )

            for row in read_csv(nsys_dir / f"{run_name}_cuda_gpu_kern_sum.csv")[:10]:
                summary_rows.append(
                    {
                        **common,
                        "record_type": "cuda_kernel",
                        "phase_range": "profile/measure",
                        "name": shorten_name(row["Name"]),
                        "calls": row["Instances"],
                        "range_total_ms": "",
                        "cpu_total_ms": "",
                        "cuda_total_ms": f"{int(row['Total Time (ns)']) / 1e6:.6f}",
                        "time_percent": row["Time (%)"],
                    }
                )

            runs.append(
                {
                    "run_name": run_name,
                    "model_size": model,
                    "context_length": context,
                    "batch_size": 4,
                    "mode": "train_step",
                    "dtype": "fp32",
                    "tool": "Nsight Systems",
                    "warmup_steps": 5,
                    "captured_measurement_steps": 1,
                    "trace_filename_local_only": f"{run_name}.nsys-rep",
                    "command": profile_command(run_name, model, context),
                    "environment": compact_environment(metadata["environment"]),
                }
            )

    fieldnames = [
        "run_name",
        "model_size",
        "context_length",
        "mode",
        "dtype",
        "tool",
        "record_type",
        "phase_range",
        "name",
        "calls",
        "range_total_ms",
        "cpu_total_ms",
        "cuda_total_ms",
        "time_percent",
    ]
    profile_dir = output_dir / "profile"
    write_csv(profile_dir / "trace_summary.csv", fieldnames, summary_rows)
    nsys_version = (nsys_dir / "nsys_version.txt").read_text(encoding="utf-8").strip()
    metadata_output = {
        "schema_version": 1,
        "primary_tool": "Nsight Systems",
        "tool_version": nsys_version,
        "capture_scope": "one post-warm-up measurement step per run",
        "submitted_trace_files": False,
        "local_raw_artifact_types": [".nsys-rep", "SQLite"],
        "runs": runs,
    }
    (profile_dir / "run_metadata.json").write_text(
        json.dumps(metadata_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def timing_record(results_dir: Path, run_name: str) -> dict[str, Any]:
    rows = read_csv(results_dir / f"{run_name}.csv")
    metadata = load_json(results_dir / f"{run_name}.metadata.json")
    values = [float(row["total_ms"]) for row in rows]
    summary = sample_summary(values)
    return {
        "run_name": run_name,
        "config": {
            key: metadata["config"][key]
            for key in (
                "model_size",
                "batch_size",
                "context_length",
                "mode",
                "warmup",
                "steps",
                "dtype",
                "seed",
            )
        },
        "total_ms": summary,
        "peak_allocated_mib": metadata["statistics"]["peak_memory_allocated_bytes"] / 1024**2,
        "peak_reserved_mib": metadata["statistics"]["peak_memory_reserved_bytes"] / 1024**2,
        "command": metadata["command"],
    }


def export_mixed_precision(results_dir: Path, output_dir: Path) -> None:
    source = load_json(results_dir / "mixed_precision.json")
    comparisons = []
    for model in ("small", "medium", "large", "xl", "10b"):
        for mode in ("forward", "forward_backward"):
            fp32_name = f"timings_{model}_{mode}_fp32"
            bf16_name = f"timings_{model}_{mode}_bf16"
            if not (results_dir / f"{fp32_name}.csv").exists():
                continue
            if not (results_dir / f"{bf16_name}.csv").exists():
                continue
            fp32 = timing_record(results_dir, fp32_name)
            bf16 = timing_record(results_dir, bf16_name)
            comparisons.append(
                {
                    "model_size": model,
                    "mode": mode,
                    "fp32": fp32,
                    "bf16": bf16,
                    "bf16_speedup": fp32["total_ms"]["mean_ms"] / bf16["total_ms"]["mean_ms"],
                }
            )

    output = {
        "schema_version": 1,
        "environment": compact_environment(source["environment"]),
        "accumulation": source["accumulation"],
        "toy_model_bf16_autocast": source["toy_model"],
        "benchmark_comparisons": comparisons,
    }
    (output_dir / "mixed_precision.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_memory(results_dir: Path, output_dir: Path) -> None:
    memory_dir = results_dir / "memory"
    selected_names = (
        "xl_ctx128_forward_fp32",
        "xl_ctx128_forward_bf16",
        "xl_ctx128_train_step_fp32",
        "xl_ctx128_train_step_bf16",
        "xl_ctx2048_forward_fp32",
        "xl_ctx2048_forward_bf16",
        "xl_ctx2048_train_step_fp32",
        "xl_ctx2048_train_step_bf16",
        "xl_ctx2048_train_step_b1_fp32",
        "xl_ctx1024_train_step_b1_fp32",
    )
    peak_rows = []
    run_metadata = []
    environment: dict[str, Any] = {}
    for run_name in selected_names:
        data = load_json(memory_dir / f"{run_name}.metadata.json")
        config = data["config"]
        memory = data["memory_after_measurement"]
        failure = data.get("failure", {})
        peak_rows.append(
            {
                "run_name": run_name,
                "model_size": config["model_size"],
                "batch_size": config["batch_size"],
                "context_length": config["context_length"],
                "mode": config["mode"],
                "dtype": config["dtype"],
                "warmup": config["warmup"],
                "status": data["status"],
                "peak_allocated_mib": f"{memory['peak_allocated_bytes'] / 1024**2:.3f}",
                "peak_reserved_mib": f"{memory['peak_reserved_bytes'] / 1024**2:.3f}",
                "failure_type": failure.get("type", ""),
                "failure_stage": failure.get("stage", ""),
                "snapshot_saved_locally": data["snapshot"]["saved"],
            }
        )
        run_metadata.append(
            {
                "run_name": run_name,
                "command": data["command"],
                "local_snapshot_filename": data["snapshot"]["filename"],
                "snapshot_submitted": False,
            }
        )
        if not environment:
            environment = compact_environment(data["environment"])

    fieldnames = [
        "run_name",
        "model_size",
        "batch_size",
        "context_length",
        "mode",
        "dtype",
        "warmup",
        "status",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "failure_type",
        "failure_stage",
        "snapshot_saved_locally",
    ]
    public_memory_dir = output_dir / "memory"
    write_csv(public_memory_dir / "peaks.csv", fieldnames, peak_rows)
    output_metadata = {
        "schema_version": 1,
        "environment": environment,
        "measurement_boundary": (
            "Warm-up completed before CUDA memory history was enabled; one independent "
            "forward or train step was then captured."
        ),
        "submitted_snapshots": False,
        "supplemental_nsys_memory_trace": {
            "configuration": {
                "model_size": "xl",
                "batch_size": 4,
                "context_length": 128,
                "mode": "train_step",
                "dtype": "fp32",
                "warmup": 0,
            },
            "purpose": (
                "A separate cold first-step trace exposes optimizer-state creation and "
                "CUDA allocation requests; it is not used as a steady-state latency result."
            ),
            "local_trace_filename": "memory_xl_ctx128_train_step_fp32.nsys-rep",
            "trace_submitted": False,
        },
        "runs": run_metadata,
    }
    (public_memory_dir / "run_metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    export_memory_block_summary(results_dir, public_memory_dir)
    export_memory_allocation_summary(results_dir, public_memory_dir)


def export_memory_block_summary(results_dir: Path, output_dir: Path) -> None:
    database_path = (
        results_dir / "nsys" / "memory_xl_ctx128_train_step_fp32.sqlite"
    )
    connection = sqlite3.connect(database_path)
    rows: list[dict[str, Any]] = []
    try:
        for phase in ("forward", "backward", "optimizer"):
            bounds = connection.execute(
                """
                SELECT start, end
                FROM NVTX_EVENTS
                WHERE text = ?
                ORDER BY (end - start) DESC
                LIMIT 1
                """,
                (phase,),
            ).fetchone()
            if bounds is None:
                continue
            count, total_bytes, largest_bytes = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(bytes), 0)
                FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                WHERE start >= ? AND start <= ?
                """,
                bounds,
            ).fetchone()
            rows.append(
                {
                    "record_type": "phase",
                    "phase": phase,
                    "block_index": "",
                    "allocation_calls": count,
                    "requested_mib": f"{total_bytes / 1024**2:.3f}",
                    "largest_request_mib": f"{largest_bytes / 1024**2:.3f}",
                }
            )

        for index in range(32):
            range_name = f"transformer_block/{index:02d}/forward"
            bounds = connection.execute(
                """
                SELECT start, end
                FROM NVTX_EVENTS
                WHERE text = ?
                ORDER BY (end - start) DESC
                LIMIT 1
                """,
                (range_name,),
            ).fetchone()
            if bounds is None:
                continue
            count, total_bytes, largest_bytes = connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(bytes), 0), COALESCE(MAX(bytes), 0)
                FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                WHERE start >= ? AND start <= ?
                """,
                bounds,
            ).fetchone()
            rows.append(
                {
                    "record_type": "transformer_block_forward",
                    "phase": "forward",
                    "block_index": index,
                    "allocation_calls": count,
                    "requested_mib": f"{total_bytes / 1024**2:.3f}",
                    "largest_request_mib": f"{largest_bytes / 1024**2:.3f}",
                }
            )
    finally:
        connection.close()

    write_csv(
        output_dir / "block_summary.csv",
        [
            "record_type",
            "phase",
            "block_index",
            "allocation_calls",
            "requested_mib",
            "largest_request_mib",
        ],
        rows,
    )


def export_memory_allocation_summary(
    results_dir: Path, output_dir: Path
) -> None:
    memory_dir = results_dir / "memory"
    selected_names = (
        "xl_ctx128_forward_fp32",
        "xl_ctx2048_forward_fp32",
        "xl_ctx128_train_step_fp32",
        "xl_ctx2048_train_step_b1_fp32",
    )
    operator_markers = {
        "structured_bmm": "bmm",
        "structured_div": "div",
        "structured_sigmoid": "sigmoid",
        "structured_mul": "mul",
        "empty_strided": "empty_strided",
    }
    rows = []
    for run_name in selected_names:
        snapshot_path = memory_dir / f"{run_name}.pickle"
        with snapshot_path.open("rb") as file:
            snapshot = pickle.load(file)
        allocation_events = [
            event
            for event in snapshot["device_traces"][0]
            if event.get("action") == "alloc"
        ]
        largest_size = max(event["size"] for event in allocation_events)
        largest_events = [
            event for event in allocation_events if event["size"] == largest_size
        ]
        operators = set()
        for event in largest_events:
            stack_names = " ".join(
                frame.get("name", "") for frame in event.get("frames", [])
            )
            for marker, label in operator_markers.items():
                if marker in stack_names:
                    operators.add(label)
        rows.append(
            {
                "run_name": run_name,
                "largest_allocation_mib": f"{largest_size / 1024**2:.3f}",
                "largest_allocation_count": len(largest_events),
                "stack_operator_categories": ";".join(sorted(operators)),
            }
        )

    write_csv(
        output_dir / "allocation_summary.csv",
        [
            "run_name",
            "largest_allocation_mib",
            "largest_allocation_count",
            "stack_operator_categories",
        ],
        rows,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_benchmark(args.results_dir, args.output_dir)
    export_profile(args.results_dir, args.output_dir)
    export_mixed_precision(args.results_dir, args.output_dir)
    export_memory(args.results_dir, args.output_dir)
    print(f"exported compact submission evidence to: {args.output_dir}")


if __name__ == "__main__":
    main()
