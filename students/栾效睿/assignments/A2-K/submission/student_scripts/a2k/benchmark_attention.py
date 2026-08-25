from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import scaled_dot_product_attention
from student_scripts.a2k.utils import (
    ALLOCATOR_LIMIT_BYTES,
    ATTENTION_QUANTILES,
    ATTENTION_REP_MS,
    ATTENTION_WARMUP_MS,
    MIB,
    allocator_evidence,
    benchmark_cuda,
    configure_cuda,
    cuda_peak_mib,
    is_cuda_oom,
    latency_columns,
    load_json,
    measure_cuda_peak,
    refresh_memory_summary,
    seed_all,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "attention_baseline.csv"
METADATA_PATH = ROOT / "results" / "run_metadata.json"
MEMORY_PATH = ROOT / "results" / "memory_evidence.json"

SEED = 336
BATCH_SIZE = 1
SEQUENCE_LENGTHS = (512, 2_048, 8_192)
HEAD_DIMS = (64, 128)
PHASES = ("forward", "backward", "forward_backward")
CSV_FIELDS = (
    "config_id",
    "sequence_length",
    "head_dim",
    "batch_size",
    "dtype",
    "is_causal",
    "implementation",
    *(f"{phase}_ms_p{percentile}" for phase in PHASES for percentile in (20, 50, 80)),
    "warmup_ms",
    "measurement_ms",
    "quantiles",
    "timer",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark explicit causal PyTorch attention for A2-K task 2.")
    parser.add_argument("--_sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_head-dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def row_template(sequence_length: int, head_dim: int) -> dict[str, Any]:
    return {
        "config_id": f"attention_eager_s{sequence_length}_d{head_dim}",
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "batch_size": BATCH_SIZE,
        "dtype": "bfloat16",
        "is_causal": True,
        "implementation": "explicit_pytorch_eager",
        **{f"{phase}_ms_p{percentile}": "" for phase in PHASES for percentile in (20, 50, 80)},
        "warmup_ms": ATTENTION_WARMUP_MS,
        "measurement_ms": ATTENTION_REP_MS,
        "quantiles": json.dumps(ATTENTION_QUANTILES),
        "timer": "triton.testing.do_bench",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "not_started",
        "error": "",
    }


def run_one(sequence_length: int, head_dim: int, output: Path) -> None:
    row, metadata = row_template(sequence_length, head_dim), {}
    try:
        metadata = configure_cuda()
        seed_all(SEED)
        q, k, v = (torch.randn((BATCH_SIZE, sequence_length, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
        causal_mask = torch.ones((sequence_length, sequence_length), device="cuda", dtype=torch.bool).tril_()
        output_gradient, inputs = torch.randn_like(q), (q, k, v)

        def forward() -> torch.Tensor:
            return scaled_dot_product_attention(q, k, v, causal_mask)

        row.update(latency_columns("forward", benchmark_cuda(forward)))
        backward_output = forward()

        def backward(saved_output: torch.Tensor = backward_output) -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(saved_output, inputs, output_gradient, retain_graph=True)

        row.update(latency_columns("backward", benchmark_cuda(backward)))
        del backward, backward_output
        gc.collect()

        def forward_backward() -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(forward(), inputs, output_gradient)

        row.update(latency_columns("forward_backward", benchmark_cuda(forward_backward)))
        peak_allocated, peak_reserved = measure_cuda_peak(forward_backward)
        row.update(
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
            status="success" if peak_reserved <= ALLOCATOR_LIMIT_BYTES / MIB else "invalid_allocator_limit",
        )
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        row.update(
            status="oom" if is_cuda_oom(error) else f"error:{type(error).__name__}",
            peak_allocated_mib=cuda_peak_mib(),
            peak_reserved_mib=cuda_peak_mib(reserved=True),
            error=type(error).__name__,
        )
    write_json(output, {"row": row, "metadata": metadata})


def launch(sequence_length: int, head_dim: int, output: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.benchmark_attention",
        "--_sequence-length",
        str(sequence_length),
        "--_head-dim",
        str(head_dim),
        "--_result",
        str(output),
    ]
    print(f"running {row_template(sequence_length, head_dim)['config_id']}", flush=True)
    process = subprocess.run(command, check=False, capture_output=True, text=True)
    if not output.exists():
        return {
            "row": {
                **row_template(sequence_length, head_dim),
                "status": f"error:child_exit_{process.returncode}",
                "error": "child produced no result",
            },
            "metadata": {},
        }
    result = load_json(output)
    print(f"  {result['row']['status']}", flush=True)
    if str(result["row"]["status"]).startswith("error:"):
        print(f"  {process.stderr.strip() or result['row']['error']}", file=sys.stderr)
    return result


def write_outputs(results: list[dict[str, Any]]) -> None:
    rows = [result["row"] for result in results]
    write_csv(CSV_PATH, rows, CSV_FIELDS)
    metadata = load_json(METADATA_PATH)
    metadata.setdefault("commands", {})["attention_baseline"] = "uv run python -m student_scripts.a2k.benchmark_attention"
    metadata["attention_baseline"] = {
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "sequence_lengths": SEQUENCE_LENGTHS,
        "head_dims": HEAD_DIMS,
        "dtype": "bfloat16",
        "causal": True,
        "input_shape": "[batch, sequence_length, head_dim]",
        "warmup_ms": ATTENTION_WARMUP_MS,
        "measurement_ms": ATTENTION_REP_MS,
        "quantiles": ATTENTION_QUANTILES,
        "timer": "triton.testing.do_bench",
        "peak_memory_scope": "one steady-state forward-backward call after empty_cache and reset_peak_memory_stats",
        "process_isolation": "one fresh Python process per shape",
        "per_process_start_free_memory_mib": {row["config_id"]: result.get("metadata", {}).get("gpu", {}).get("start_free_memory_mib") for row, result in zip(rows, results)},
    }
    first_metadata = next((result["metadata"] for result in results if result["metadata"]), {})
    metadata.update({key: value for key, value in first_metadata.items() if key not in metadata})
    write_json(METADATA_PATH, metadata)

    successful = [row for row in rows if row["status"] == "success"]
    evidence = load_json(MEMORY_PATH)
    evidence["allocator"] = allocator_evidence(metadata.get("allocator", {}).get("fraction"))
    evidence["attention_baseline"] = {
        "highest_peak_allocated_mib": max((float(row["peak_allocated_mib"]) for row in successful), default=None),
        "highest_peak_reserved_mib": max((float(row["peak_reserved_mib"]) for row in successful), default=None),
        "within_23gib_allocator": bool(successful) and max(float(row["peak_reserved_mib"]) for row in successful) <= ALLOCATOR_LIMIT_BYTES / MIB,
        "config_status": {row["config_id"]: row["status"] for row in rows},
    }
    refresh_memory_summary(evidence)
    write_json(MEMORY_PATH, evidence)


def run_matrix() -> int:
    with tempfile.TemporaryDirectory(prefix="a2k_attention_") as temp_dir:
        results = [launch(sequence_length, head_dim, Path(temp_dir) / f"s{sequence_length}_d{head_dim}.json") for sequence_length, head_dim in product(SEQUENCE_LENGTHS, HEAD_DIMS)]
    write_outputs(results)
    for path in (CSV_PATH, METADATA_PATH, MEMORY_PATH):
        print(f"wrote {path}")
    return int(any(result["row"]["status"] not in {"success", "oom"} for result in results))


def main() -> int:
    args = parse_args()
    if args._sequence_length is not None:
        if args._head_dim is None or args._result is None:
            raise ValueError("Missing internal benchmark arguments")
        run_one(args._sequence_length, args._head_dim, args._result)
        return 0
    return run_matrix()


if __name__ == "__main__":
    raise SystemExit(main())
