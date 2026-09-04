"""Run the required A2-K activation-checkpointing experiment matrix.

The script writes only measurements it actually performed.  It refuses a
formal run unless the process owns a suitably idle single RTX 4090 and sets the
23 GiB PyTorch allocator guard before constructing a CUDA model or tensor.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch
import torch.nn.functional as functional

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k.checkpointing import checkpointed_blocks

try:
    from .common import (
        CudaPreflightError,
        append_memory_observation,
        append_run_metadata,
        configure_cuda,
        record_preflight_failure,
        stderr,
        write_csv as write_common_csv,
    )
except ImportError:  # pragma: no cover - direct-script fallback.
    from common import (  # type: ignore[no-redef]
        CudaPreflightError,
        append_memory_observation,
        append_run_metadata,
        configure_cuda,
        record_preflight_failure,
        stderr,
        write_csv as write_common_csv,
    )


MIB = 1024**2
ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
VOCAB_SIZE = 10_000
MEDIUM_CONFIG = {
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
}
CSV_FIELDS = [
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
    "allocator_fraction",
    "allocator_limit_mib",
    "status",
    "error_kind",
    "formal",
]


class CheckpointedMediumLM(BasicsTransformerLM):
    """Stanford medium LM whose layer stack can be checkpointed in segments."""

    def __init__(self, context_length: int, checkpoint_block_size: int | None) -> None:
        super().__init__(
            vocab_size=VOCAB_SIZE,
            context_length=context_length,
            **MEDIUM_CONFIG,
        )
        self.checkpoint_block_size = checkpoint_block_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_embeddings(token_ids)
        hidden_states = checkpointed_blocks(self.layers, hidden_states, self.checkpoint_block_size)
        hidden_states = self.ln_final(hidden_states)
        return self.lm_head(hidden_states)


def _run_training_step(
    model: CheckpointedMediumLM,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(token_ids)
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()


def _peak_mib() -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated() / MIB,
        torch.cuda.max_memory_reserved() / MIB,
    )


def measure_configuration(
    *,
    context_length: int,
    checkpoint_block_size: int | None,
    seed: int,
    warmup_steps: int,
    measurement_steps: int,
    formal: bool,
    allocator_fraction: float,
    allocator_limit_mib: float,
) -> dict[str, Any]:
    """Measure one complete training-step configuration and retain OOM evidence."""

    checkpoint_label = "none" if checkpoint_block_size is None else str(checkpoint_block_size)
    record: dict[str, Any] = {
        "config_id": f"medium_ctx{context_length}_block{checkpoint_label}",
        "model_size": "medium",
        "num_layers": MEDIUM_CONFIG["num_layers"],
        "context_length": context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_parameters",
        "checkpoint_block_size": checkpoint_label,
        "nested": False,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
        "step_time_ms_samples": "[]",
        "step_time_ms_p50": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "allocator_fraction": allocator_fraction,
        "allocator_limit_mib": allocator_limit_mib,
        "status": "runtime_error",
        "error_kind": "runtime_error",
        "formal": formal,
    }
    model: CheckpointedMediumLM | None = None
    optimizer: torch.optim.Optimizer | None = None
    token_ids: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    peak_allocated: list[float] = []
    peak_reserved: list[float] = []

    try:
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CheckpointedMediumLM(context_length, checkpoint_block_size).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
        token_ids = torch.randint(VOCAB_SIZE, (1, context_length), device="cuda", dtype=torch.long)
        targets = torch.randint(VOCAB_SIZE, (1, context_length), device="cuda", dtype=torch.long)

        for _ in range(warmup_steps):
            _run_training_step(model, optimizer, token_ids, targets)
        torch.cuda.synchronize()

        samples_ms: list[float] = []
        for _ in range(measurement_steps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            _run_training_step(model, optimizer, token_ids, targets)
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - started) * 1_000.0)
            allocated_mib, reserved_mib = _peak_mib()
            peak_allocated.append(allocated_mib)
            peak_reserved.append(reserved_mib)

        max_allocated_mib = max(peak_allocated)
        max_reserved_mib = max(peak_reserved)
        status = "success" if max_reserved_mib <= ALLOCATOR_LIMIT_MIB else "allocator_limit_exceeded"
        record.update(
            {
                "step_time_ms_samples": json.dumps(samples_ms),
                "step_time_ms_p50": statistics.median(samples_ms),
                "peak_allocated_mib": max_allocated_mib,
                "peak_reserved_mib": max_reserved_mib,
                "status": status,
                "error_kind": "" if status == "success" else "allocator_limit_exceeded",
            }
        )
    except torch.OutOfMemoryError:
        allocated_mib, reserved_mib = _peak_mib()
        record.update(
            {
                "peak_allocated_mib": allocated_mib,
                "peak_reserved_mib": reserved_mib,
                "status": "oom",
                "error_kind": "oom",
            }
        )
    except RuntimeError:
        allocated_mib, reserved_mib = _peak_mib()
        record.update(
            {
                "peak_allocated_mib": allocated_mib,
                "peak_reserved_mib": reserved_mib,
                "status": "runtime_error",
                "error_kind": "runtime_error",
            }
        )
    finally:
        del model, optimizer, token_ids, targets
        gc.collect()
        torch.cuda.empty_cache()
    return record


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    """Atomically persist the fixed checkpointing CSV schema."""

    write_common_csv(path, CSV_FIELDS, records)


def matrix_completed(records: list[dict[str, Any]]) -> bool:
    """Return whether the required matrix finished with its allowed boundary OOM.

    The 2048 no-checkpoint row is an intentional boundary experiment. The
    assignment permits it to OOM, provided every 1024 configuration and the
    selected 2048 checkpoint fallback complete within the allocator budget.
    Any other non-success status remains a failed/incomplete matrix.
    """

    expected_standard_blocks = ("none", "1", "2", "4", "8")
    by_configuration = {
        (record.get("context_length"), str(record.get("checkpoint_block_size"))): record
        for record in records
    }
    if len(by_configuration) != len(records):
        return False

    standard_complete = all(
        by_configuration.get((1024, block), {}).get("status") == "success"
        for block in expected_standard_blocks
    )
    boundary_baseline = by_configuration.get((2048, "none"), {})
    boundary_baseline_allowed = boundary_baseline.get("status") in {"success", "oom"}
    boundary_checkpoint_records = [
        record
        for (context_length, checkpoint_block_size), record in by_configuration.items()
        if context_length == 2048 and checkpoint_block_size != "none"
    ]
    boundary_checkpoint_complete = (
        len(boundary_checkpoint_records) == 1 and boundary_checkpoint_records[0].get("status") == "success"
    )
    return standard_complete and boundary_baseline_allowed and boundary_checkpoint_complete


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/checkpointing.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("local_results/a2k/checkpointing_metadata.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--checkpoint-block-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--formal",
        action="store_true",
        help="explicitly request the default formal RTX 4090 measurement mode",
    )
    mode_group.add_argument(
        "--nonformal",
        action="store_true",
        help="allow a CUDA smoke run on a non-4090 GPU; records are tagged formal=false",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup_steps < 3 or args.measurement_steps < 5:
        raise ValueError("the required matrix needs at least 3 warm-up and 5 measurement steps")
    if any(block_size <= 0 for block_size in args.checkpoint_block_sizes):
        raise ValueError("checkpoint block sizes must be positive")

    formal = not args.nonformal
    if formal and args.checkpoint_block_sizes != [1, 2, 4, 8]:
        raise ValueError("Formal checkpointing mode requires block sizes 1, 2, 4, and 8 in that order.")
    run_configuration = {
        "model_size": "medium",
        "num_layers": MEDIUM_CONFIG["num_layers"],
        "context_lengths": [1024, 2048],
        "checkpoint_block_sizes": list(args.checkpoint_block_sizes),
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_parameters",
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
    }
    try:
        runtime = configure_cuda("cuda:0", formal=formal)
    except CudaPreflightError as error:
        preflight_record: dict[str, Any] = {
            "config_id": "medium_checkpointing_not_run",
            "model_size": "medium",
            "num_layers": MEDIUM_CONFIG["num_layers"],
            "context_length": None,
            "batch_size": 1,
            "dtype": "bf16_autocast_fp32_parameters",
            "checkpoint_block_size": "not_run",
            "nested": False,
            "warmup_steps": args.warmup_steps,
            "measurement_steps": args.measurement_steps,
            "step_time_ms_samples": "[]",
            "step_time_ms_p50": None,
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
            "allocator_fraction": None,
            "allocator_limit_mib": None,
            "status": error.status,
            "error_kind": "cuda_preflight",
            "formal": formal,
        }
        write_csv(args.output, [preflight_record])
        record_preflight_failure(
            args.output.parent,
            script_name="benchmark_checkpointing.py",
            formal=formal,
            configuration=run_configuration,
            error=error,
        )
        stderr(error.public_reason)
        return 2

    safe_gpu_metadata = {
        "gpu_name": runtime.gpu_name,
        "gpu_total_mib": runtime.total_memory_mib,
        "gpu_free_mib": runtime.free_memory_mib,
        "driver_version": runtime.driver_version,
        "power_limit_watts": runtime.power_limit_watts,
        "pstate": runtime.pstate,
    }
    allocator_metadata = {
        "allocator_fraction": runtime.allocator_fraction,
        "allocator_limit_mib": float(ALLOCATOR_LIMIT_MIB),
        "device_total_mib": runtime.total_memory_mib,
    }
    records: list[dict[str, Any]] = []

    for block_size in [None, *args.checkpoint_block_sizes]:
        records.append(
            measure_configuration(
                context_length=1024,
                checkpoint_block_size=block_size,
                seed=args.seed,
                warmup_steps=args.warmup_steps,
                measurement_steps=args.measurement_steps,
                formal=formal,
                allocator_fraction=runtime.allocator_fraction,
                allocator_limit_mib=float(ALLOCATOR_LIMIT_MIB),
            )
        )

    checkpoint_successes = [
        record
        for record in records
        if record["context_length"] == 1024
        and record["checkpoint_block_size"] != "none"
        and record["status"] == "success"
        and isinstance(record["peak_allocated_mib"], (float, int))
    ]
    records.append(
        measure_configuration(
            context_length=2048,
            checkpoint_block_size=None,
            seed=args.seed,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps,
            formal=formal,
            allocator_fraction=runtime.allocator_fraction,
            allocator_limit_mib=float(ALLOCATOR_LIMIT_MIB),
        )
    )
    if checkpoint_successes:
        best_record = min(checkpoint_successes, key=lambda record: float(record["peak_allocated_mib"]))
        records.append(
            measure_configuration(
                context_length=2048,
                checkpoint_block_size=int(best_record["checkpoint_block_size"]),
                seed=args.seed,
                warmup_steps=args.warmup_steps,
                measurement_steps=args.measurement_steps,
                formal=formal,
                allocator_fraction=runtime.allocator_fraction,
                allocator_limit_mib=float(ALLOCATOR_LIMIT_MIB),
            )
        )
    else:
        records.append(
            {
                "config_id": "medium_ctx2048_best_1024_checkpoint",
                "model_size": "medium",
                "num_layers": MEDIUM_CONFIG["num_layers"],
                "context_length": 2048,
                "batch_size": 1,
                "dtype": "bf16_autocast_fp32_parameters",
                "checkpoint_block_size": "unavailable",
                "nested": False,
                "warmup_steps": args.warmup_steps,
                "measurement_steps": args.measurement_steps,
                "step_time_ms_samples": "[]",
                "step_time_ms_p50": None,
                "peak_allocated_mib": None,
                "peak_reserved_mib": None,
                "allocator_fraction": runtime.allocator_fraction,
                "allocator_limit_mib": float(ALLOCATOR_LIMIT_MIB),
                "status": "not_run_missing_1024_checkpoint_success",
                "error_kind": "missing_prerequisite",
                "formal": formal,
            }
        )

    write_csv(args.output, records)
    observed_allocated = [
        float(record["peak_allocated_mib"])
        for record in records
        if isinstance(record.get("peak_allocated_mib"), (float, int))
    ]
    observed_reserved = [
        float(record["peak_reserved_mib"])
        for record in records
        if isinstance(record.get("peak_reserved_mib"), (float, int))
    ]
    status = "success" if matrix_completed(records) else "incomplete"
    boundary_baseline = next(
        (record for record in records if record["context_length"] == 2048 and record["checkpoint_block_size"] == "none"),
        None,
    )
    metadata = {
        "schema_version": 1,
        "task": "a2k_checkpointing",
        "status": status,
        "formal": formal,
        "seed": args.seed,
        "model": {"size": "medium", "vocab_size": VOCAB_SIZE, **MEDIUM_CONFIG},
        "allocator": allocator_metadata,
        "hard_limit_mib": HARD_LIMIT_MIB,
        "gpu": safe_gpu_metadata,
        "dtype": "bf16_autocast_fp32_parameters",
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "boundary_baseline_status": boundary_baseline["status"] if boundary_baseline is not None else "missing",
        "boundary_baseline_oom_allowed": True,
        "command": "python -m student_scripts.a2k.benchmark_checkpointing",
    }
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_run_metadata(
        args.output.parent,
        script_name="benchmark_checkpointing.py",
        runtime=runtime,
        status=status,
        formal=formal,
        configuration=run_configuration,
    )
    append_memory_observation(
        args.output.parent,
        script_name="benchmark_checkpointing.py",
        runtime=runtime,
        status=status,
        peak_allocated_mib=max(observed_allocated) if observed_allocated else None,
        peak_reserved_mib=max(observed_reserved) if observed_reserved else None,
        formal=formal,
    )
    if status != "success":
        stderr(
            "Checkpointing matrix is incomplete: all 1024 rows and the 2048 checkpoint fallback must succeed; "
            "only the 2048 no-checkpoint boundary row may be OOM. Inspect checkpointing.csv."
        )
        return 1
    print(f"Wrote {len(records)} real checkpointing rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
