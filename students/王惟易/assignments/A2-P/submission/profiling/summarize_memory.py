import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = (
    "source_json",
    "timestamp_utc",
    "status",
    "failure_stage",
    "error_type",
    "model_size",
    "d_model",
    "d_ff",
    "num_layers",
    "num_heads",
    "vocab_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "warmup_steps",
    "measurement_steps",
    "seed",
    "learning_rate",
    "device",
    "gpu",
    "gpu_memory_mib",
    "compute_capability",
    "driver_version",
    "cuda_runtime",
    "cudnn_version",
    "pytorch",
    "python",
    "measurement_started",
    "loss",
    "peak_scope",
    "current_active_mib",
    "peak_active_mib",
    "current_allocated_mib",
    "peak_allocated_mib",
    "current_reserved_mib",
    "peak_reserved_mib",
    "snapshot_written",
    "snapshot_kind",
    "snapshot_output",
    "command",
)

MODEL_ORDER = {
    "small": 0,
    "medium": 1,
    "large": 2,
    "xl": 3,
    "10b": 4,
}

MODE_ORDER = {
    "forward": 0,
    "train_step": 1,
}

DTYPE_ORDER = {
    "fp32": 0,
    "bf16": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine memory-profile metadata into a lightweight CSV.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    environment = payload["environment"]
    measurement = payload["measurement"]
    memory = payload["memory"]
    snapshot = payload["snapshot"]

    return {
        "source_json": path.name,
        "timestamp_utc": payload["timestamp_utc"],
        "status": payload["status"],
        "failure_stage": payload["failure_stage"],
        "error_type": payload["error_type"],
        "model_size": config["model_size"],
        "d_model": config["d_model"],
        "d_ff": config["d_ff"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "vocab_size": config["vocab_size"],
        "batch_size": config["batch_size"],
        "context_length": config["context_length"],
        "mode": config["mode"],
        "dtype": config["dtype"],
        "warmup_steps": config["warmup_steps"],
        "measurement_steps": config["measurement_steps"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "device": environment["device"],
        "gpu": environment["gpu"],
        "gpu_memory_mib": environment["gpu_memory_mib"],
        "compute_capability": environment["compute_capability"],
        "driver_version": environment["driver_version"],
        "cuda_runtime": environment["cuda_runtime"],
        "cudnn_version": environment["cudnn_version"],
        "pytorch": environment["pytorch"],
        "python": environment["python"],
        "measurement_started": measurement["started"],
        "loss": measurement["loss"],
        "peak_scope": measurement["peak_scope"],
        "current_active_mib": memory["current_active_mib"],
        "peak_active_mib": memory["peak_active_mib"],
        "current_allocated_mib": memory["current_allocated_mib"],
        "peak_allocated_mib": memory["peak_allocated_mib"],
        "current_reserved_mib": memory["current_reserved_mib"],
        "peak_reserved_mib": memory["peak_reserved_mib"],
        "snapshot_written": snapshot["written"],
        "snapshot_kind": snapshot["kind"],
        "snapshot_output": snapshot["output"],
        "command": payload["command"],
    }


def row_order(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        MODEL_ORDER[str(row["model_size"])],
        int(row["batch_size"]),
        int(row["context_length"]),
        MODE_ORDER[str(row["mode"])],
        DTYPE_ORDER[str(row["dtype"])],
    )


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.metadata.json"))
    if not input_paths:
        raise FileNotFoundError(
            f"no memory metadata JSON files found in {args.input_dir}",
        )

    rows = sorted((load_row(path) for path in input_paths), key=row_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} memory rows to {args.output}")


if __name__ == "__main__":
    main()
