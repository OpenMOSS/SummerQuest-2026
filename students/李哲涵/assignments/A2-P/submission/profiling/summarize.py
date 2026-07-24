from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
from pathlib import Path
from typing import Any


def _json_files(root: Path, directory: str) -> list[Path]:
    return sorted((root / directory).rglob("*.json"))


def _snapshot_max_allocation(snapshot_path: Path) -> float | None:
    if not snapshot_path.is_file():
        return None
    try:
        with snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError):
        return None
    sizes = [
        int(block.get("size", 0))
        for segment in snapshot.get("segments", [])
        for block in segment.get("blocks", [])
    ]
    sizes.extend(
        int(history.get("real_size", history.get("size", 0)))
        for segment in snapshot.get("segments", [])
        for block in segment.get("blocks", [])
        for history in block.get("history", [])
    )
    return max(sizes, default=0) / (1024**2) or None


def summarize_benchmark(raw_root: Path, output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in _json_files(raw_root, "benchmark"):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        timing = data.get("timing", {})
        rows.append(
            {
                "file": path.name,
                "status": data.get("status"),
                "model_size": config.get("model_size"),
                "mode": config.get("mode"),
                "dtype": config.get("dtype"),
                "batch_size": config.get("batch_size"),
                "context_length": config.get("context_length"),
                "warmup": config.get("warmup"),
                "steps": config.get("steps"),
                "mean_ms": timing.get("mean_ms"),
                "sample_std_ms": timing.get("sample_std_ms"),
                "cv": timing.get("cv"),
                "min_ms": timing.get("min_ms"),
                "max_ms": timing.get("max_ms"),
                "raw_timings_ms": json.dumps(timing.get("raw_timings_ms", [])),
                "peak_allocated_mib": (data.get("memory") or {}).get("peak_allocated_mib"),
                "peak_reserved_mib": (data.get("memory") or {}).get("peak_reserved_mib"),
                "error": data.get("error"),
            }
        )
    destination = output_root / "benchmark.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "file",
        "status",
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "warmup",
        "steps",
        "mean_ms",
        "sample_std_ms",
        "cv",
        "min_ms",
        "max_ms",
        "raw_timings_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "error",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_profile(raw_root: Path, output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in sorted((raw_root / "profile" / "summaries").glob("*.json")):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        metadata.append(
            {
                "status": data.get("status"),
                "config": config,
                "tool": data.get("tool"),
                "command": data.get("command"),
                "environment": data.get("environment"),
                "trace_file": (data.get("trace_path") or "").split("/")[-1],
                "error": data.get("error"),
            }
        )
        summary = data.get("summary", {})
        for kind in ("operators", "kernels", "ranges"):
            for row in summary.get(kind, []):
                rows.append(
                    {
                        "run": path.stem,
                        "model_size": config.get("model_size"),
                        "context_length": config.get("context_length"),
                        "dtype": config.get("dtype"),
                        **row,
                    }
                )
    destination = output_root / "profile" / "trace_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "model_size",
        "context_length",
        "dtype",
        "kind",
        "name",
        "calls",
        "cpu_total_us",
        "cpu_self_us",
        "cuda_total_us",
        "cuda_self_us",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "profile" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def summarize_memory(raw_root: Path, output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in sorted((raw_root / "memory" / "runs").glob("*.json")):
        data = json.loads(path.read_text())
        config = data.get("config", {})
        memory = data.get("memory", {})
        snapshot_name = Path(data.get("snapshot_path", "")).name
        largest_allocation = data.get("peak_allocation_from_snapshot_mib")
        if largest_allocation is None:
            largest_allocation = _snapshot_max_allocation(
                raw_root / "memory" / "snapshots" / snapshot_name
            )
        rows.append(
            {
                "run": path.stem,
                "status": data.get("status"),
                "model_size": config.get("model_size"),
                "mode": config.get("mode"),
                "dtype": config.get("dtype"),
                "batch_size": config.get("batch_size"),
                "context_length": config.get("context_length"),
                "warmup": config.get("warmup"),
                "warmup_mode": config.get("warmup_mode"),
                "peak_allocated_mib": memory.get("peak_allocated_mib"),
                "peak_reserved_mib": memory.get("peak_reserved_mib"),
                "peak_active_mib": memory.get("peak_active_mib"),
                "largest_snapshot_allocation_mib": largest_allocation,
                "failed_stage": data.get("failed_stage"),
                "memory_points": json.dumps(data.get("memory_points", {})),
                "error": data.get("error"),
            }
        )
        metadata.append(
            {
                "run": path.stem,
                "status": data.get("status"),
                "config": config,
                "environment": data.get("environment"),
                "snapshot_file": snapshot_name,
                "failed_stage": data.get("failed_stage"),
                "error": data.get("error"),
            }
        )
    destination = output_root / "memory" / "peaks.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else [
        "run",
        "status",
        "model_size",
        "mode",
        "dtype",
        "batch_size",
        "context_length",
        "warmup",
        "warmup_mode",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "peak_active_mib",
        "largest_snapshot_allocation_mib",
        "failed_stage",
        "memory_points",
        "error",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "memory" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def summarize(raw_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summarize_benchmark(raw_root, output_root)
    summarize_profile(raw_root, output_root)
    summarize_memory(raw_root, output_root)
    mixed = raw_root / "mixed_precision.json"
    if mixed.is_file():
        shutil.copy2(mixed, output_root / "mixed_precision.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create A2-P lightweight result files")
    parser.add_argument("--raw-root", default="results/a2p")
    parser.add_argument("--output-root", default="results/a2p_public")
    args = parser.parse_args()
    root = Path(args.raw_root)
    output = Path(args.output_root)
    summarize(root, output)
    print(f"Wrote public summaries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
