from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.a2k.attention import FlashAttentionTriton, TRITON_NUM_STAGES, TRITON_NUM_WARPS, TRITON_TILE_SIZE
from student_scripts.a2k.flash_common import causal_mask, make_attention_inputs
from student_scripts.a2k.utils import ALLOCATOR_LIMIT_BYTES, ATTENTION_QUANTILES, ATTENTION_REP_MS, ATTENTION_WARMUP_MS, MIB, allocator_evidence, benchmark_cuda, configure_cuda, cuda_peak_mib, is_cuda_oom, latency_columns, load_json, measure_cuda_peak, refresh_memory_summary, write_csv, write_json


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "flash_benchmark.csv"
METADATA_PATH = ROOT / "results" / "run_metadata.json"
MEMORY_PATH = ROOT / "results" / "memory_evidence.json"
SEED = 336
CORE_CONFIGS = tuple(product((512, 2_048, 8_192), (64, 128)))
BOUNDARY_CONFIGS = tuple(product((16_384,), (64, 128)))
PHASES = ("forward", "backward", "forward_backward")
CORE_IMPLEMENTATIONS = ("explicit_pytorch_eager", "compiled_pytorch", "triton_flashattention")
BOUNDARY_IMPLEMENTATIONS = ("explicit_pytorch_eager", "triton_flashattention")
CSV_FIELDS = ("config_id", "implementation", "sequence_length", "head_dim", "batch_size", "dtype", "is_causal", "phase", "latency_ms_p20", "latency_ms_p50", "latency_ms_p80", "peak_allocated_mib", "peak_reserved_mib", "speedup_to_eager", "query_tile", "key_tile", "num_warps", "num_stages", "warmup_ms", "measurement_ms", "quantiles", "timer", "status", "error")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark A2-K FlashAttention implementations.")
    parser.add_argument("--_implementation", choices=CORE_IMPLEMENTATIONS, help=argparse.SUPPRESS)
    parser.add_argument("--_sequence-length", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_head-dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def row_template(implementation: str, sequence_length: int, head_dim: int, phase: str) -> dict[str, Any]:
    triton = implementation == "triton_flashattention"
    return {
        "config_id": f"flash_{implementation}_s{sequence_length}_d{head_dim}_{phase}", "implementation": implementation, "sequence_length": sequence_length, "head_dim": head_dim, "batch_size": 1, "dtype": "bfloat16", "is_causal": True, "phase": phase,
        "latency_ms_p20": "", "latency_ms_p50": "", "latency_ms_p80": "", "peak_allocated_mib": "", "peak_reserved_mib": "", "speedup_to_eager": "", "query_tile": TRITON_TILE_SIZE if triton else "", "key_tile": TRITON_TILE_SIZE if triton else "", "num_warps": TRITON_NUM_WARPS if triton else "", "num_stages": TRITON_NUM_STAGES if triton else "", "warmup_ms": ATTENTION_WARMUP_MS, "measurement_ms": ATTENTION_REP_MS, "quantiles": json.dumps(ATTENTION_QUANTILES), "timer": "triton.testing.do_bench", "status": "not_started", "error": "",
    }


def forward_function(implementation: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> Callable[[], torch.Tensor]:
    if implementation == "triton_flashattention":
        def triton_forward() -> torch.Tensor:
            return FlashAttentionTriton.apply(q, k, v, True)

        return triton_forward

    def eager() -> torch.Tensor:
        return scaled_dot_product_attention(q, k, v, mask)

    if implementation == "compiled_pytorch":
        compiled = torch.compile(eager, backend="inductor", fullgraph=True, dynamic=False)
        compiled()
        torch.cuda.synchronize()
        return compiled
    return eager


def measure_phase(phase: str, forward: Callable[[], torch.Tensor], inputs: tuple[torch.Tensor, ...], do: torch.Tensor) -> dict[str, float]:
    if phase == "forward":
        step = forward
    elif phase == "backward":
        output = forward()

        def step() -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(output, inputs, do, retain_graph=True)
    else:
        def step() -> tuple[torch.Tensor, ...]:
            return torch.autograd.grad(forward(), inputs, do)
    latency, peak = benchmark_cuda(step), measure_cuda_peak(step)
    return {**latency_columns("latency", latency), "peak_allocated_mib": peak[0], "peak_reserved_mib": peak[1]}


def run_one(implementation: str, sequence_length: int, head_dim: int, output: Path) -> None:
    rows, metadata = [row_template(implementation, sequence_length, head_dim, phase) for phase in PHASES], {}
    try:
        metadata = configure_cuda()
        q, k, v, do = make_attention_inputs(SEED, sequence_length, head_dim, torch.bfloat16)
        forward, inputs = forward_function(implementation, q, k, v, causal_mask(sequence_length)), (q, k, v)
        for row in rows:
            row.update(measure_phase(row["phase"], forward, inputs, do))
            row["status"] = "success" if float(row["peak_reserved_mib"]) <= ALLOCATOR_LIMIT_BYTES / MIB else "invalid_allocator_limit"
            gc.collect()
    except Exception as error:
        for row in rows:
            if row["status"] == "not_started":
                row.update(status="oom" if is_cuda_oom(error) else f"error:{type(error).__name__}", peak_allocated_mib=cuda_peak_mib(), peak_reserved_mib=cuda_peak_mib(reserved=True), error=type(error).__name__)
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
    write_json(output, {"rows": rows, "metadata": metadata})


def launch(implementation: str, sequence_length: int, head_dim: int, output: Path, cache: Path) -> dict[str, Any]:
    command = [sys.executable, "-m", "student_scripts.a2k.benchmark_flash_attention", "--_implementation", implementation, "--_sequence-length", str(sequence_length), "--_head-dim", str(head_dim), "--_result", str(output)]
    environment = os.environ.copy()
    if implementation == "compiled_pytorch":
        cache.mkdir(parents=True, exist_ok=True)
        environment.update(TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), TRITON_CACHE_DIR=str(cache / "triton"))
    print(f"running flash_{implementation}_s{sequence_length}_d{head_dim}", flush=True)
    process = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    if output.exists():
        result = load_json(output)
        print(f"  {', '.join(row['status'] for row in result['rows'])}", flush=True)
        if process.returncode and process.stderr:
            print(process.stderr.strip(), file=sys.stderr)
        return result
    return {"rows": [{**row_template(implementation, sequence_length, head_dim, phase), "status": f"error:child_exit_{process.returncode}", "error": "child produced no result"} for phase in PHASES], "metadata": {}}


def add_speedups(rows: list[dict[str, Any]]) -> None:
    eager = {(row["sequence_length"], row["head_dim"], row["phase"]): row for row in rows if row["implementation"] == "explicit_pytorch_eager" and row["status"] == "success"}
    for row in rows:
        reference = eager.get((row["sequence_length"], row["head_dim"], row["phase"]))
        if row["status"] == "success" and reference is not None:
            row["speedup_to_eager"] = float(reference["latency_ms_p50"]) / float(row["latency_ms_p50"])


def write_outputs(results: list[dict[str, Any]]) -> None:
    rows = [row for result in results for row in result["rows"]]
    add_speedups(rows)
    write_csv(CSV_PATH, rows, CSV_FIELDS)
    metadata, successful = load_json(METADATA_PATH), [row for row in rows if row["status"] == "success"]
    metadata.setdefault("commands", {})["flash_benchmark"] = "uv run python -m student_scripts.a2k.benchmark_flash_attention"
    metadata["flash_benchmark"] = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(), "seed": SEED, "batch_size": 1, "dtype": "bfloat16", "causal": True, "core_configs": CORE_CONFIGS, "boundary_configs": BOUNDARY_CONFIGS, "phases": PHASES, "implementations": CORE_IMPLEMENTATIONS, "timer": "triton.testing.do_bench", "warmup_ms": ATTENTION_WARMUP_MS, "measurement_ms": ATTENTION_REP_MS, "quantiles": ATTENTION_QUANTILES, "process_isolation": "one fresh Python process per implementation and shape", "peak_memory_scope": "one steady-state phase call after empty_cache and reset_peak_memory_stats"}
    first_metadata = next((result["metadata"] for result in results if result["metadata"]), {})
    metadata.update({key: value for key, value in first_metadata.items() if key not in metadata})
    write_json(METADATA_PATH, metadata)
    evidence = load_json(MEMORY_PATH)
    evidence["allocator"] = allocator_evidence(metadata.get("allocator", {}).get("fraction"))
    evidence["flash_benchmark"] = {"highest_peak_allocated_mib": max((float(row["peak_allocated_mib"]) for row in successful), default=None), "highest_peak_reserved_mib": max((float(row["peak_reserved_mib"]) for row in successful), default=None), "within_23gib_allocator": all(float(row["peak_reserved_mib"]) <= ALLOCATOR_LIMIT_BYTES / MIB for row in successful), "config_status": {row["config_id"]: row["status"] for row in rows}}
    evidence["hard_limit_mib"] = 24 * 1024
    refresh_memory_summary(evidence)
    write_json(MEMORY_PATH, evidence)


def run_matrix() -> int:
    cases = [*( (implementation, sequence_length, head_dim) for sequence_length, head_dim in CORE_CONFIGS for implementation in CORE_IMPLEMENTATIONS), *( (implementation, sequence_length, head_dim) for sequence_length, head_dim in BOUNDARY_CONFIGS for implementation in BOUNDARY_IMPLEMENTATIONS)]
    with tempfile.TemporaryDirectory(prefix="a2k_flash_") as temp_dir:
        temporary = Path(temp_dir)
        results = [launch(implementation, sequence_length, head_dim, temporary / f"result_{index}.json", temporary / f"cache_{index}") for index, (implementation, sequence_length, head_dim) in enumerate(cases)]
    write_outputs(results)
    print(f"wrote {CSV_PATH}, {METADATA_PATH}, and {MEMORY_PATH}")
    return int(any(row["status"] not in {"success", "oom"} for result in results for row in result["rows"]))


def main() -> int:
    args = parse_args()
    if args._implementation is None:
        return run_matrix()
    if args._sequence_length is None or args._head_dim is None or args._result is None:
        raise ValueError("Missing internal benchmark arguments")
    run_one(args._implementation, args._sequence_length, args._head_dim, args._result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
