from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from student_scripts.a2k.compile_cases import (
    ATTENTION_CONFIGS,
    BATCH_SIZE,
    MODEL_CONFIG,
    MODEL_COMPARE_ATOL,
    MODEL_COMPARE_RTOL,
    MODEL_CONTEXT,
    MODEL_MEASUREMENT_STEPS,
    MODEL_WARMUP_STEPS,
    dynamo_stats,
    run_attention,
    run_model,
)
from student_scripts.a2k.utils import (
    ALLOCATOR_LIMIT_BYTES,
    ATTENTION_QUANTILES,
    ATTENTION_REP_MS,
    ATTENTION_WARMUP_MS,
    MIB,
    allocator_evidence,
    configure_cuda,
    cuda_peak_mib,
    is_cuda_oom,
    load_json,
    refresh_memory_summary,
    seed_all,
    write_csv,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "compile_comparison.csv"
METADATA_PATH = ROOT / "results" / "run_metadata.json"
MEMORY_PATH = ROOT / "results" / "memory_evidence.json"

SEED = 336
IMPLEMENTATIONS = ("eager", "compiled")
PHASES = ("forward", "backward", "forward_backward", "training_step")
CSV_FIELDS = (
    "config_id",
    "benchmark_type",
    "implementation",
    "model_size",
    "sequence_length",
    "head_dim",
    "batch_size",
    "dtype",
    "is_causal",
    "compile_backend",
    "compile_dynamic",
    "compile_fullgraph",
    "forward_cold_start_ms",
    "backward_cold_start_ms",
    "total_cold_start_ms",
    *(f"{phase}_ms_p{percentile}" for phase in PHASES for percentile in (20, 50, 80)),
    "warmup",
    "measurement",
    "measurement_unit",
    "timer",
    "graph_break_count",
    "unique_graph_count",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eager and torch.compile for A2-K task 2.")
    parser.add_argument("--_kind", choices=("attention", "model"), help=argparse.SUPPRESS)
    parser.add_argument("--_implementation", choices=IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--_sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_head-dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def row_template(kind: str, implementation: str, sequence_length: int, head_dim: int) -> dict[str, Any]:
    compiled, model = implementation == "compiled", kind == "model"
    return {
        "config_id": f"{kind}_{implementation}_s{sequence_length}_d{head_dim}",
        "benchmark_type": kind,
        "implementation": implementation,
        "model_size": "small" if model else "",
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "batch_size": BATCH_SIZE,
        "dtype": "bf16_autocast_fp32_params" if model else "bfloat16",
        "is_causal": True,
        "compile_backend": "inductor" if compiled else "",
        "compile_dynamic": False if compiled else "",
        "compile_fullgraph": (not model) if compiled else "",
        "forward_cold_start_ms": "",
        "backward_cold_start_ms": "",
        "total_cold_start_ms": "",
        **{f"{phase}_ms_p{percentile}": "" for phase in PHASES for percentile in (20, 50, 80)},
        "warmup": MODEL_WARMUP_STEPS if model else ATTENTION_WARMUP_MS,
        "measurement": MODEL_MEASUREMENT_STEPS if model else ATTENTION_REP_MS,
        "measurement_unit": "steps" if model else "milliseconds",
        "timer": "torch.cuda.Event" if model else "triton.testing.do_bench",
        "graph_break_count": "",
        "unique_graph_count": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "status": "not_started",
        "error": "",
    }


def run_one(kind: str, implementation: str, sequence_length: int, head_dim: int, output: Path) -> None:
    row, metadata = row_template(kind, implementation, sequence_length, head_dim), {}
    try:
        metadata = configure_cuda()
        seed_all(SEED)
        row = run_attention(row) if kind == "attention" else run_model(row)
        row["graph_break_count"], row["unique_graph_count"] = dynamo_stats() if implementation == "compiled" else (0, 0)
        row["status"] = "success" if float(row["peak_reserved_mib"]) <= ALLOCATOR_LIMIT_BYTES / MIB else "invalid_allocator_limit"
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        row.update(
            status="oom" if is_cuda_oom(error) else f"error:{type(error).__name__}",
            peak_allocated_mib=cuda_peak_mib(),
            peak_reserved_mib=cuda_peak_mib(reserved=True),
            error=type(error).__name__,
        )
    write_json(output, {"row": row, "metadata": metadata})


def launch(kind: str, implementation: str, sequence_length: int, head_dim: int, output: Path, cache: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.benchmark_compile",
        "--_kind",
        kind,
        "--_implementation",
        implementation,
        "--_sequence-length",
        str(sequence_length),
        "--_head-dim",
        str(head_dim),
        "--_result",
        str(output),
    ]
    environment = os.environ.copy()
    if implementation == "compiled":
        cache.mkdir(parents=True, exist_ok=True)
        environment.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
    label = row_template(kind, implementation, sequence_length, head_dim)["config_id"]
    print(f"running {label}", flush=True)
    process = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    if not output.exists():
        return {
            "row": {
                **row_template(kind, implementation, sequence_length, head_dim),
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
    metadata.setdefault("commands", {})["compile_comparison"] = "uv run python -m student_scripts.a2k.benchmark_compile"
    metadata["compile_comparison"] = {
        "seed": SEED,
        "attention": {
            "batch_size": BATCH_SIZE,
            "configs": ATTENTION_CONFIGS,
            "dtype": "bfloat16",
            "causal": True,
            "warmup_ms": ATTENTION_WARMUP_MS,
            "measurement_ms": ATTENTION_REP_MS,
            "quantiles": ATTENTION_QUANTILES,
            "timer": "triton.testing.do_bench",
        },
        "model": {
            "size": "small",
            "config": MODEL_CONFIG,
            "context_length": MODEL_CONTEXT,
            "batch_size": BATCH_SIZE,
            "dtype": "bf16_autocast_fp32_params",
            "optimizer": "torch.optim.AdamW(lr=1e-3, weight_decay=0.01)",
            "warmup_steps": MODEL_WARMUP_STEPS,
            "measurement_steps": MODEL_MEASUREMENT_STEPS,
            "timer": "torch.cuda.Event",
            "correctness_tolerance": {"rtol": MODEL_COMPARE_RTOL, "atol": MODEL_COMPARE_ATOL},
        },
        "compile": {
            "backend": "inductor",
            "dynamic": False,
            "attention_fullgraph": True,
            "model_fullgraph": False,
            "cold_start": "first compiled forward and first compiled backward, timed separately",
            "cache_policy": "fresh TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR per compiled child process",
        },
        "process_isolation": "one fresh Python process per benchmark type, implementation, and shape",
        "peak_memory_scope": "maximum peak across steady-state phases",
        "per_process_start_free_memory_mib": {row["config_id"]: result.get("metadata", {}).get("gpu", {}).get("start_free_memory_mib") for row, result in zip(rows, results)},
    }
    first_metadata = next((result["metadata"] for result in results if result["metadata"]), {})
    metadata.update({key: value for key, value in first_metadata.items() if key not in metadata})
    write_json(METADATA_PATH, metadata)

    successful = [row for row in rows if row["status"] == "success"]
    evidence = load_json(MEMORY_PATH)
    evidence["allocator"] = allocator_evidence(metadata.get("allocator", {}).get("fraction"))
    evidence["compile_comparison"] = {
        "highest_peak_allocated_mib": max((float(row["peak_allocated_mib"]) for row in successful), default=None),
        "highest_peak_reserved_mib": max((float(row["peak_reserved_mib"]) for row in successful), default=None),
        "within_23gib_allocator": bool(successful) and max(float(row["peak_reserved_mib"]) for row in successful) <= ALLOCATOR_LIMIT_BYTES / MIB,
        "config_status": {row["config_id"]: row["status"] for row in rows},
    }
    refresh_memory_summary(evidence)
    write_json(MEMORY_PATH, evidence)


def run_matrix() -> int:
    with tempfile.TemporaryDirectory(prefix="a2k_compile_") as temp_dir:
        temp = Path(temp_dir)
        cases = [
            *(("attention", implementation, sequence_length, head_dim) for sequence_length, head_dim in ATTENTION_CONFIGS for implementation in IMPLEMENTATIONS),
            *(("model", implementation, MODEL_CONTEXT, MODEL_CONFIG["d_model"] // MODEL_CONFIG["num_heads"]) for implementation in IMPLEMENTATIONS),
        ]
        results = [
            launch(kind, implementation, sequence_length, head_dim, temp / f"result_{index}.json", temp / f"cache_{index}")
            for index, (kind, implementation, sequence_length, head_dim) in enumerate(cases)
        ]
    write_outputs(results)
    for path in (CSV_PATH, METADATA_PATH, MEMORY_PATH):
        print(f"wrote {path}")
    return int(any(result["row"]["status"] not in {"success", "oom"} for result in results))


def main() -> int:
    args = parse_args()
    if args._kind is not None:
        if args._implementation is None or args._sequence_length is None or args._head_dim is None or args._result is None:
            raise ValueError("Missing internal benchmark arguments")
        run_one(args._kind, args._implementation, args._sequence_length, args._head_dim, args._result)
        return 0
    return run_matrix()


if __name__ == "__main__":
    raise SystemExit(main())
