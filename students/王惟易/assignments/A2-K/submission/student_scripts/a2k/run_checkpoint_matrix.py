from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


STANDARD_BLOCK_SIZES: tuple[int | None, ...] = (None, 1, 2, 4, 8)
CSV_FIELDS = (
    "config_id",
    "model_size",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "allocator_limit_mib",
    "allocator_fraction",
    "within_24gib",
    "status",
    "failure_stage",
    "error_type",
    "source_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed A2-K checkpointing matrix serially.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-free-mib", type=float, default=22 * 1024)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def block_label(block_size: int | None) -> str:
    return "none" if block_size is None else str(block_size)


def result_path(output_dir: Path, context_length: int, block_size: int | None) -> Path:
    return output_dir / f"medium_b1_t{context_length}_bf16_checkpoint_{block_label(block_size)}.json"


def load_completed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") not in {"ok", "oom"}:
        return None
    return result


def run_case(
    args: argparse.Namespace,
    context_length: int,
    block_size: int | None,
) -> dict[str, Any]:
    output = result_path(args.output_dir, context_length, block_size)
    if args.resume and (completed := load_completed(output)) is not None:
        print(f"Reusing {output}", flush=True)
        return completed

    command = [
        sys.executable,
        "student_scripts/a2k/checkpoint_benchmark.py",
        "--context-length",
        str(context_length),
        "--checkpoint-block-size",
        block_label(block_size),
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--minimum-free-mib",
        str(args.minimum_free_mib),
        "--output",
        output.as_posix(),
    ]
    print(f"Running context={context_length}, checkpoint_block_size={block_label(block_size)}", flush=True)
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def to_csv_row(result: dict[str, Any], source: Path) -> dict[str, Any]:
    config = result["config"]
    timing = result.get("timing", {})
    memory = result["memory"]
    allocator = result["allocator"]
    block_size = config["checkpoint_block_size"]
    return {
        "config_id": (f"medium-b1-t{config['context_length']}-bf16-checkpoint-{block_label(block_size)}"),
        "model_size": config["model_size"],
        "num_layers": config["num_layers"],
        "context_length": config["context_length"],
        "batch_size": config["batch_size"],
        "dtype": config["dtype"],
        "checkpoint_block_size": block_label(block_size),
        "nested": str(config["nested"]).lower(),
        "warmup_steps": config["warmup_steps"],
        "measurement_steps": config["measurement_steps"],
        "step_time_ms_samples": json.dumps(timing.get("step_time_ms_samples", []), separators=(",", ":")),
        "step_time_ms_p50": timing.get("step_time_ms_p50", ""),
        "peak_allocated_mib": memory["peak_allocated_mib"],
        "peak_reserved_mib": memory["peak_reserved_mib"],
        "allocator_limit_mib": allocator["allocator_limit_mib"],
        "allocator_fraction": allocator["allocator_fraction"],
        "within_24gib": str(memory["within_24gib"]).lower(),
        "status": result["status"],
        "failure_stage": result.get("failure_stage") or "",
        "error_type": result.get("error_type") or "",
        "source_json": source.name,
    }


def write_csv(output: Path, results: list[dict[str, Any]]) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for result in results:
            config = result["config"]
            source = result_path(
                output.parent,
                config["context_length"],
                config["checkpoint_block_size"],
            )
            writer.writerow(to_csv_row(result, source))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [run_case(args, 1024, block_size) for block_size in STANDARD_BLOCK_SIZES]
    successful_checkpointed = [result for result in results if result["status"] == "ok" and result["config"]["checkpoint_block_size"] is not None]
    if not successful_checkpointed:
        raise RuntimeError("no checkpointed context-1024 configuration succeeded")

    best = min(
        successful_checkpointed,
        key=lambda result: result["memory"]["peak_allocated_mib"],
    )
    best_block_size = best["config"]["checkpoint_block_size"]
    results.append(run_case(args, 2048, None))
    results.append(run_case(args, 2048, best_block_size))

    csv_output = args.output_dir / "checkpointing.csv"
    write_csv(csv_output, results)
    print(f"Lowest-memory context-1024 checkpoint block size: {best_block_size}")
    print(f"Wrote {len(results)} rows to {csv_output}")


if __name__ == "__main__":
    main()
