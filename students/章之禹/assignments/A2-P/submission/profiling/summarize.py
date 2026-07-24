"""Build lightweight A2-P result tables and report-friendly SVG plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any


PROFILE_STAGE_NAMES = {
    "profile/warmup",
    "profile/measure",
    "zero_grad",
    "forward",
    "loss",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
}


def public_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    hidden = {"output", "trace_output", "table_output", "snapshot"}
    return {key: value for key, value in config.items() if key not in hidden}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def public_error(payload: dict[str, Any]) -> str | None:
    """Return a small, stable failure label without exposing runtime text."""

    if payload.get("status") == "complete":
        return None
    error_type = payload.get("error_type")
    phase = payload.get("phase")
    if error_type and phase:
        return f"{error_type} during {phase}"
    if error_type:
        return str(error_type)
    return "failed"


def public_filename(value: Any) -> str | None:
    """Keep machine-local paths out of public result tables."""

    if value is None:
        return None
    return Path(str(value)).name


def snapshot_summary(snapshot: Any) -> dict[str, Any]:
    """Extract only stable, public allocator facts from a local snapshot."""

    if not snapshot:
        return {}
    try:
        with Path(str(snapshot)).open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        return {}
    traces = payload.get("device_traces", [])
    if not traces:
        return {}
    events = traces[0]
    allocations = [event for event in events if event.get("action") == "alloc"]
    if not allocations:
        return {}
    largest = max(allocations, key=lambda event: int(event.get("size", 0)))
    operator_names: list[str] = []
    for frame in largest.get("frames", []):
        name = str(frame.get("name", ""))
        if name and name not in operator_names:
            operator_names.append(name)
    # Collapse the verbose C++ stack to a small operation category.  The full
    # stack stays in the private pickle; this public label is enough to explain
    # the allocation without leaking local filenames or build details.
    lowered = " ".join(operator_names).lower()
    if "softmax" in lowered or "sum_dim" in lowered or "reduction" in lowered:
        allocation_op = "softmax/reduction"
    elif "matmul" in lowered or "gemm" in lowered or "mm(" in lowered:
        allocation_op = "matmul"
    elif "empty_strided" in lowered or "to_copy" in lowered:
        allocation_op = "tensor materialization/cast"
    else:
        allocation_op = "other allocator operation"
    final_active_mib = sum(
        int(block.get("size", 0))
        for segment in payload.get("segments", [])
        for block in segment.get("blocks", [])
        if str(block.get("state", "")).startswith("active_")
    ) / 2**20
    return {
        "largest_allocation_mib": round(int(largest.get("size", 0)) / 2**20, 3),
        "largest_allocation_op": allocation_op,
        "snapshot_final_active_mib": round(final_active_mib, 3),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def benchmark_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        config = payload.get("config", {})
        stats = payload.get("stats", {})
        rows.append(
            {
                "source": path.name,
                "status": payload.get("status"),
                "model_size": config.get("model_size"),
                "batch_size": config.get("batch_size"),
                "context_length": config.get("context_length"),
                "mode": config.get("mode"),
                "dtype": config.get("dtype"),
                "autocast": config.get("autocast"),
                "warmup_steps": config.get("warmup_steps"),
                "measurement_steps": config.get("measurement_steps"),
                "mean_ms": stats.get("mean_ms"),
                "std_ms": stats.get("std_ms"),
                "cv": stats.get("cv"),
                "peak_allocated_mib": payload.get("peak_allocated_mib"),
                "peak_reserved_mib": payload.get("peak_reserved_mib"),
                "raw_timings_ms": json.dumps(payload.get("timings_ms", []), separators=(",", ":")),
                "error": public_error(payload),
            }
        )
    return rows


def profile_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        config = payload.get("config", {})
        common = {
            "source": path.name,
            "status": payload.get("status"),
            "model_size": config.get("model_size"),
            "batch_size": config.get("batch_size"),
            "context_length": config.get("context_length"),
            "dtype": config.get("dtype"),
            "tool": payload.get("tool"),
            "event_count": payload.get("event_count"),
            "peak_allocated_mib": payload.get("peak_allocated_mib"),
            "peak_reserved_mib": payload.get("peak_reserved_mib"),
            "trace_output": public_filename(payload.get("trace_output")),
            "table_output": public_filename(payload.get("table_output")),
        }
        operator_summary = payload.get("operator_summary", [])
        if not isinstance(operator_summary, list) or not operator_summary:
            rows.append({**common, "stage": "run", "name": "summary"})
            continue
        for operator in operator_summary:
            if not isinstance(operator, dict):
                continue
            name = str(operator.get("name", ""))
            record_type = operator.get("record_type")
            if not isinstance(record_type, str):
                record_type = "operator"
                if name in PROFILE_STAGE_NAMES:
                    record_type = (
                        "cpu_range"
                        if float(operator.get("cpu_total_us", 0.0)) > 0.0
                        else "gpu_annotation"
                    )
            rows.append(
                {
                    **common,
                    "stage": operator.get("stage", "operator"),
                    "name": name,
                    "record_type": record_type,
                    "calls": operator.get("calls"),
                    "cpu_total_us": operator.get("cpu_total_us"),
                    "cpu_self_us": operator.get("cpu_self_us"),
                    "cuda_total_us": operator.get("cuda_total_us"),
                    "cuda_self_us": operator.get("cuda_self_us"),
                    "cpu_memory_bytes": operator.get("cpu_memory_bytes"),
                    "cuda_memory_bytes": operator.get("cuda_memory_bytes"),
                }
            )
        rows.extend(profile_kernel_overlap_rows(path, payload, common))
    return rows


def profile_kernel_overlap_rows(
    profile_path: Path, payload: dict[str, Any], common: dict[str, Any]
) -> list[dict[str, Any]]:
    """Summarize CUDA kernel time inside CPU stage ranges from a local trace."""

    trace_output = payload.get("trace_output")
    if not trace_output:
        return []
    trace_path = Path(str(trace_output))
    if not trace_path.exists() and not trace_path.is_absolute():
        trace_path = profile_path.parent / trace_path
    try:
        trace = read_json(trace_path)
    except (OSError, json.JSONDecodeError):
        return []
    events = trace.get("traceEvents", [])
    if not isinstance(events, list):
        return []

    kernels = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("cat") == "kernel"
        and event.get("ph") == "X"
    ]
    if not kernels:
        return []

    def overlap(left: float, right: float, start: float, end: float) -> float:
        return max(0.0, min(right, end) - max(left, start))

    totals: dict[str, dict[str, float]] = {}
    for event in events:
        if (
            not isinstance(event, dict)
            or event.get("cat") != "user_annotation"
            or event.get("ph") != "X"
            or event.get("name") not in PROFILE_STAGE_NAMES
        ):
            continue
        start = float(event.get("ts", 0.0))
        duration = float(event.get("dur", 0.0))
        end = start + duration
        overlaps = [
            overlap(
                start,
                end,
                float(kernel.get("ts", 0.0)),
                float(kernel.get("ts", 0.0)) + float(kernel.get("dur", 0.0)),
            )
            for kernel in kernels
        ]
        kernel_count = sum(value > 0.0 for value in overlaps)
        if kernel_count == 0:
            continue
        stage = str(event["name"])
        current = totals.setdefault(
            stage,
            {"cpu_total_us": 0.0, "calls": 0.0, "cuda_total_us": 0.0},
        )
        current["cpu_total_us"] += duration
        current["calls"] += kernel_count
        current["cuda_total_us"] += sum(overlaps)

    rows: list[dict[str, Any]] = []
    for stage, values in totals.items():
        rows.append(
            {
                **common,
                "stage": stage,
                "name": "CUDA kernels overlapping CPU range",
                "record_type": "kernel_overlap",
                "calls": int(values["calls"]),
                "cpu_total_us": values["cpu_total_us"],
                "cpu_self_us": None,
                "cuda_total_us": values["cuda_total_us"],
                "cuda_self_us": None,
                "cpu_memory_bytes": None,
                "cuda_memory_bytes": None,
            }
        )
    return rows


def public_mixed_benchmark(benchmark: dict[str, Any]) -> dict[str, Any]:
    """Repair legacy in-process benchmark metadata from its measured config."""

    clean = dict(benchmark)
    config = dict(clean.get("config", {}))
    autocast = bool(config.get("autocast"))
    if not autocast and config.get("compute_dtype") == "torch.float32":
        config["dtype"] = "fp32"
    clean["config"] = config
    model_size = config.get("model_size")
    dtype = config.get("dtype")
    output_file = clean.get("output_file")
    if not isinstance(output_file, str) or not output_file:
        output_file = f"mixed_precision_{model_size}_{dtype}.json"
    clean["output_file"] = output_file
    if model_size and dtype:
        command = [
            "python profiling/benchmark.py",
            "--model-size",
            str(model_size),
            "--vocab-size",
            str(config.get("vocab_size")),
            "--batch-size",
            str(config.get("batch_size")),
            "--context-length",
            str(config.get("context_length")),
            "--mode",
            str(config.get("mode")),
            "--warmup",
            str(config.get("warmup_steps")),
            "--steps",
            str(config.get("measurement_steps")),
            "--dtype",
            str(dtype),
            "--device",
            "cuda",
            "--seed",
            str(config.get("seed")),
            "--output",
            output_file,
        ]
        if autocast:
            command.append("--autocast")
        clean["command"] = " ".join(command)
    clean["execution"] = "in_process"
    clean["error"] = public_error(clean)
    return clean


def memory_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = read_json(path)
        config = payload.get("config", {})
        snapshot_name = public_filename(payload.get("snapshot"))
        snapshot_info = snapshot_summary(payload.get("snapshot"))
        rows.append(
            {
                "source": path.name,
                "status": payload.get("status"),
                "model_size": config.get("model_size"),
                "batch_size": config.get("batch_size"),
                "context_length": config.get("context_length"),
                "mode": config.get("mode"),
                "dtype": config.get("dtype"),
                "autocast": config.get("autocast"),
                "peak_allocated_mib": payload.get("peak_allocated_mib"),
                "peak_reserved_mib": payload.get("peak_reserved_mib"),
                "snapshot": snapshot_name,
                "failure_phase": payload.get("phase"),
                "error_type": payload.get("error_type"),
                **snapshot_info,
                # Raw allocator error strings can contain local memory state,
                # internal runtime wording, or paths.  The public CSV keeps
                # only the structured failure classification required by the
                # assignment.
                "error": public_error(payload),
            }
        )
    return rows


def _plot(rows: list[dict[str, Any]], output: Path, x_key: str, y_key: str, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    points: list[tuple[float, float, str]] = []
    for row in rows:
        raw_x = row.get(x_key)
        raw_y = row.get(y_key)
        if not isinstance(raw_x, (int, float)) or not isinstance(raw_y, (int, float)):
            continue
        x_value = float(raw_x)
        y_value = float(raw_y)
        if math.isfinite(x_value) and math.isfinite(y_value):
            label = str(row.get("mode", row.get("model_size", "run")))
            points.append((x_value, y_value, label))
    if not points:
        return
    labels = sorted({str(point[2]) for point in points})
    fig, ax = plt.subplots(figsize=(7, 4))
    for label in labels:
        selected = [(float(x), float(y)) for x, y, current in points if str(current) == label]
        selected.sort()
        ax.plot([item[0] for item in selected], [item[1] for item in selected], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.grid(True, alpha=0.25)
    if len(labels) > 1:
        ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    plt.close(fig)


def _plot_profile_stages(rows: list[dict[str, Any]], output: Path) -> None:
    """Render a compact stage-time comparison for one representative trace.

    The raw Chrome traces remain local.  This chart is deliberately derived
    only from the public operator CSV and therefore contains no host paths or
    process metadata.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    preferred = {
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    }
    totals: dict[str, float] = {}
    candidates = sorted({str(row.get("source", "")) for row in rows})
    representative = (
        "medium_ctx1024.json"
        if "medium_ctx1024.json" in candidates
        else (candidates[-1] if candidates else None)
    )
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage not in preferred:
            continue
        value = row.get("cuda_total_us")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            value = row.get("cpu_total_us")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        source = str(row.get("source", ""))
        # Use one trace only to avoid adding incomparable runs together.
        if source != representative:
            continue
        totals[stage] = max(totals.get(stage, 0.0), float(value) / 1000.0)
    if not totals:
        return
    order = [
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    ]
    labels = [label for label in order if label in totals]
    fig, axis = plt.subplots(figsize=(8, 4.2), dpi=140)
    axis.bar(labels, [totals[label] for label in labels], color="#0f766e")
    axis.set_ylabel("time (ms)")
    axis.set_title(f"Representative train_step stages ({representative})")
    axis.tick_params(axis="x", rotation=28)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--memory-dir", type=Path)
    parser.add_argument("--mixed-precision", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_paths = sorted(args.benchmark_dir.glob("*.json")) if args.benchmark_dir else []
    profile_paths = sorted(args.profile_dir.glob("*.json")) if args.profile_dir else []
    memory_paths = sorted(args.memory_dir.glob("*.json")) if args.memory_dir else []
    b_rows = benchmark_rows(benchmark_paths)
    p_rows = profile_rows(profile_paths)
    m_rows = memory_rows(memory_paths)
    write_csv(output_dir / "benchmark.csv", b_rows)
    write_csv(output_dir / "profile" / "trace_summary.csv", p_rows)
    write_csv(output_dir / "memory" / "peaks.csv", m_rows)
    profile_metadata: dict[str, Any] = {
        "schema_version": 1,
        "profile_files": [path.name for path in profile_paths],
        "profile_runs": [],
    }
    for path in profile_paths:
        payload = read_json(path)
        profile_metadata["profile_runs"].append(
            {
                "source": path.name,
                "config": public_config(payload.get("config")),
                "hardware": payload.get("hardware"),
                "tool": payload.get("tool"),
                "command": payload.get("command"),
                "trace_file": public_filename(payload.get("trace_output")),
                "operator_count": payload.get("event_count"),
            }
        )

    memory_metadata: dict[str, Any] = {
        "schema_version": 1,
        "memory_files": [path.name for path in memory_paths],
        "memory_runs": [],
    }
    for path in memory_paths:
        payload = read_json(path)
        memory_metadata["memory_runs"].append(
            {
                "source": path.name,
                "config": payload.get("config"),
                "hardware": payload.get("hardware"),
                "command": payload.get("command"),
                "snapshot_file": public_filename(payload.get("snapshot")),
                **snapshot_summary(payload.get("snapshot")),
                "status": payload.get("status"),
                "phase": payload.get("phase"),
                "error_type": payload.get("error_type"),
                "error": public_error(payload),
            }
        )
    if args.mixed_precision:
        mixed_payloads = [read_json(path) for path in args.mixed_precision]
        merged: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "hardware": mixed_payloads[0].get("hardware", {}),
            "accumulation": mixed_payloads[0].get("accumulation", {}),
            "toy_model": mixed_payloads[0].get("toy_model", {}),
            "benchmarks": [],
            "runs": [],
        }
        for payload in mixed_payloads:
            for benchmark in payload.get("benchmarks", []):
                if not isinstance(benchmark, dict):
                    continue
                merged["benchmarks"].append(public_mixed_benchmark(benchmark))
            merged["runs"].append(
                {
                    "hardware": payload.get("hardware"),
                    "command": payload.get("command"),
                    "status": payload.get("status"),
                }
            )
            if payload.get("status") != "complete":
                merged["status"] = "partial"
            if any(
                isinstance(item, dict) and item.get("status") != "complete"
                for item in payload.get("benchmarks", [])
            ):
                merged["status"] = "partial"
        (output_dir / "mixed_precision.json").write_text(
            json.dumps(merged, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    (output_dir / "profile" / "run_metadata.json").write_text(
        json.dumps(profile_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "memory" / "run_metadata.json").write_text(
        json.dumps(memory_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assets = args.assets_dir or (output_dir / "assets")
    _plot(b_rows, assets / "benchmark_modes.svg", "mean_ms", "std_ms", "A2-P benchmark variability")
    _plot(m_rows, assets / "memory_peaks.svg", "context_length", "peak_allocated_mib", "A2-P peak allocated memory")
    # Profile CSVs are intentionally lightweight; use event count/peak memory as
    # a reproducible summary chart when the raw trace remains local.
    _plot(p_rows, assets / "profile_events.svg", "event_count", "peak_allocated_mib", "A2-P profiler coverage")
    _plot_profile_stages(p_rows, assets / "profile_stages.svg")
    print(json.dumps({"benchmark_rows": len(b_rows), "profile_rows": len(p_rows), "memory_rows": len(m_rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
