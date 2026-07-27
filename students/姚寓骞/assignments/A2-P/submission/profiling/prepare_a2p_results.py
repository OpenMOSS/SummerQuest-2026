"""Build the six lightweight A2-P submission result files from local raw runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TRACE_MODELS = ("small", "large")
TRACE_CONTEXTS = (256, 512, 1024)


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"required result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_benchmark(raw: Path, output: Path) -> None:
    with (raw / "benchmark.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {
        ("forward", "5", "fp32"),
        ("forward_backward", "5", "fp32"),
        ("train_step", "5", "fp32"),
        ("train_step", "0", "fp32"),
        ("train_step", "5", "bf16"),
    }
    selected = [
        row for row in rows if row["model_size"] == "small" and row["batch_size"] == "4" and row["context_length"] == "512" and (row["mode"], row["warmup"], row["dtype"]) in wanted
    ]
    present = {(row["mode"], row["warmup"], row["dtype"]) for row in selected}
    missing = wanted - present
    if missing:
        raise ValueError(f"benchmark.csv is missing required configurations: {sorted(missing)}")
    # Retain the additional language-model FP32/BF16 trend runs requested by
    # the fixed PDF, while avoiding duplicate baseline rows.
    selected_ids = {(row["model_size"], row["mode"], row["warmup"], row["dtype"]) for row in selected}
    for row in rows:
        identity = (row["model_size"], row["mode"], row["warmup"], row["dtype"])
        if (
            row["model_size"] in {"small", "medium", "large", "xl"}
            and row["batch_size"] == "4"
            and row["context_length"] == "512"
            and row["mode"] == "forward_backward"
            and row["warmup"] == "5"
            and row["dtype"] in {"fp32", "bf16"}
            and identity not in selected_ids
        ):
            selected.append(row)
            selected_ids.add(identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected[0].keys())
        writer.writeheader()
        writer.writerows(selected)


def first_row(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        try:
            return next(csv.DictReader(handle))
        except StopIteration as error:
            raise ValueError(f"empty CSV: {path}") from error


def nvtx_phase_times(path: Path) -> dict[str, float]:
    required = {
        "profile/measure",
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    }
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            # Nsight CSV renders the default (unnamed) NVTX domain as a
            # leading colon, e.g. ``:profile/measure``.
            name = row.get("Range", row.get("Name", "")).removeprefix(":")
            if name in required:
                result[name] = int(row["Total Time (ns)"]) / 1e6
    missing = required - result.keys()
    if missing:
        raise ValueError(f"NVTX summary {path} is missing ranges: {sorted(missing)}")
    return result


def build_profile(raw: Path, output: Path) -> None:
    summary_rows = []
    metadata = []
    for model in TRACE_MODELS:
        for context in TRACE_CONTEXTS:
            run_name = f"{model}_c{context}_train_step_fp32"
            run = read_json(raw / "profile" / f"{run_name}.json")
            kernel = first_row(raw / "profile" / f"{run_name}_stats_cuda_gpu_kern_sum.csv")
            api = first_row(raw / "profile" / f"{run_name}_stats_cuda_api_sum.csv")
            phases = nvtx_phase_times(raw / "profile" / f"{run_name}_stats_nvtx_sum.csv")
            summary_rows.append(
                {
                    "run_name": run_name,
                    "model_size": model,
                    "context_length": context,
                    "mode": "train_step",
                    "dtype": "fp32",
                    "phase_range": "profile/measure;forward;backward;optimizer;attention/scores;attention/softmax;attention/value",
                    "nvtx_cpu_times_ms": json.dumps(phases, sort_keys=True),
                    "step_time_ms": run["summary"]["mean_ms"],
                    "main_kernel": kernel["Name"],
                    "kernel_calls": kernel.get("Instances", kernel.get("Num Calls", "")),
                    "kernel_cuda_time_ms": int(kernel["Total Time (ns)"]) / 1e6,
                    "kernel_cuda_time_percent": kernel["Time (%)"],
                    "main_cuda_api": api["Name"],
                    "api_calls": api.get("Num Calls", api.get("Instances", "")),
                    "api_cpu_time_ms": int(api["Total Time (ns)"]) / 1e6,
                    "api_cpu_time_percent": api["Time (%)"],
                }
            )
            metadata.append(
                {
                    "run_name": run_name,
                    "config": run["config"],
                    "model": run["model"],
                    "environment": run["environment"],
                    "command": run["command"],
                    "local_trace_file": f"{run_name}.nsys-rep",
                    "submitted_trace": False,
                    "tool": "Nsight Systems",
                }
            )
    summary_path = output / "profile" / "trace_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    write_json(output / "profile" / "run_metadata.json", {"runs": metadata})


def build_mixed_precision(raw: Path, output: Path) -> None:
    write_json(
        output / "mixed_precision.json",
        {
            "accumulation": read_json(raw / "mixed_precision" / "accumulation.json"),
            "toy_model": read_json(raw / "mixed_precision" / "toy_model.json"),
        },
    )


def build_memory(raw: Path, output: Path) -> None:
    runs = [read_json(path) for path in sorted((raw / "memory").glob("official_*.json"))]
    if not runs:
        raise FileNotFoundError("no official_*.json memory runs were found")
    fields = [
        "model_size",
        "batch_size",
        "context_length",
        "mode",
        "dtype",
        "status",
        "phase",
        "active_peak_bytes",
        "allocated_peak_bytes",
        "reserved_peak_bytes",
        "error_type",
    ]
    peaks = output / "memory" / "peaks.csv"
    peaks.parent.mkdir(parents=True, exist_ok=True)
    with peaks.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: run.get(field, "") for field in fields})
    write_json(
        output / "memory" / "run_metadata.json",
        {
            "runs": [{key: value for key, value in run.items() if key not in {"error", "snapshot"}} for run in runs],
            "large_binary_policy": "Snapshots remain local and are not submitted.",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("results/a2p_raw"))
    parser.add_argument("--output", type=Path, default=Path("results/a2p_submission"))
    args = parser.parse_args()
    build_benchmark(args.raw_root, args.output / "benchmark.csv")
    build_profile(args.raw_root, args.output)
    build_mixed_precision(args.raw_root, args.output)
    build_memory(args.raw_root, args.output)
    print(f"Wrote lightweight A2-P results to {args.output}")


if __name__ == "__main__":
    main()
