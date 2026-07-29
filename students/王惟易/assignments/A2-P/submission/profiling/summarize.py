import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDNAMES = (
    "source_json",
    "timestamp_utc",
    "model_size",
    "d_model",
    "d_ff",
    "num_layers",
    "num_heads",
    "vocab_size",
    "batch_size",
    "context_length",
    "mode",
    "warmup_steps",
    "measurement_steps",
    "dtype",
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
    "samples_ms",
    "mean_ms",
    "stdev_ms",
    "cv",
    "losses",
    "first_loss",
    "last_loss",
    "peak_allocated_mib",
    "peak_reserved_mib",
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
    "forward_backward": 1,
    "train_step": 2,
}

DTYPE_ORDER = {
    "fp32": 0,
    "bf16": 1,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine benchmark JSON artifacts into a lightweight CSV.",
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload["config"]
    environment = payload["environment"]
    timing = payload["timing"]
    losses = payload["numerics"]["losses"]
    memory = payload["memory"]

    return {
        "source_json": path.name,
        "timestamp_utc": payload["timestamp_utc"],
        "model_size": config["model_size"],
        "d_model": config["d_model"],
        "d_ff": config["d_ff"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "vocab_size": config["vocab_size"],
        "batch_size": config["batch_size"],
        "context_length": config["context_length"],
        "mode": config["mode"],
        "warmup_steps": config["warmup_steps"],
        "measurement_steps": config["measurement_steps"],
        "dtype": config["dtype"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "device": environment["device"],
        "gpu": environment.get("gpu", ""),
        "gpu_memory_mib": environment.get("gpu_memory_mib", ""),
        "compute_capability": environment.get("compute_capability", ""),
        "driver_version": environment.get("driver_version", ""),
        "cuda_runtime": environment.get("cuda_runtime", ""),
        "cudnn_version": environment.get("cudnn_version", ""),
        "pytorch": environment["pytorch"],
        "python": environment["python"],
        "samples_ms": json.dumps(timing["samples_ms"], separators=(",", ":")),
        "mean_ms": timing["mean_ms"],
        "stdev_ms": timing["stdev_ms"],
        "cv": timing["cv"],
        "losses": json.dumps(losses, separators=(",", ":")),
        "first_loss": losses[0] if losses else "",
        "last_loss": losses[-1] if losses else "",
        "peak_allocated_mib": memory["peak_allocated_mib"],
        "peak_reserved_mib": memory["peak_reserved_mib"],
        "command": payload["command"],
    }


def row_order(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        MODEL_ORDER[str(row["model_size"])],
        MODE_ORDER[str(row["mode"])],
        DTYPE_ORDER[str(row["dtype"])],
        -int(row["warmup_steps"]),
    )


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("*.json"))
    if not input_paths:
        raise FileNotFoundError(f"no JSON files found in {args.input_dir}")

    rows = sorted((load_row(path) for path in input_paths), key=row_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
