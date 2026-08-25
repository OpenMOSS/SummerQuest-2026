#!/usr/bin/env python3
"""Run the A2-K activation-checkpointing matrix in isolated processes.

The public entry point is the coordinator.  It never creates a CUDA tensor;
instead, every configuration is executed by a fresh invocation of this module.
This is important because allocator peaks and OOMs are otherwise contaminated by
objects left alive by an earlier configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from student_scripts.a2k.runtime import (
    ALLOCATOR_LIMIT_MIB,
    assert_public_payload,
    atomic_write_csv,
    atomic_write_json,
    child_process_environment,
    peak_memory_mib,
    prepare_runtime,
    public_error,
    release_memory,
    reset_peak_memory,
    synchronize,
)

FORMAL_BLOCK_SIZES = (0, 1, 2, 4, 8)
PUBLIC_COLUMNS = (
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
    "status",
    "seed",
    "allocator_limit_mib",
    "allocator_fraction",
    "within_allocator_limit",
    "error_type",
    "error_summary",
)


def _write_public_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    public_rows = [{field: row.get(field) for field in PUBLIC_COLUMNS} for row in rows]
    atomic_write_csv(path, public_rows, fieldnames=PUBLIC_COLUMNS)


def _autocast(device: Any, dtype: str):
    import contextlib
    import torch

    if dtype == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _run_training_step(
    model: Any,
    optimizer: Any,
    input_ids: Any,
    targets: Any,
    *,
    device: Any,
    dtype: str,
) -> float:
    import torch.nn.functional as F

    synchronize(device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with _autocast(device, dtype):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    optimizer.step()
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1_000
    del logits, loss
    return elapsed_ms


def _base_record(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_id": spec["config_id"],
        "model_size": spec["model_size"],
        "num_layers": spec["num_layers"],
        "context_length": spec["context_length"],
        "batch_size": spec["batch_size"],
        "dtype": spec["dtype"],
        "checkpoint_block_size": spec["checkpoint_block_size"],
        "nested": False,
        "warmup_steps": spec["warmup_steps"],
        "measurement_steps": spec["measurement_steps"],
        "step_time_ms_samples": [],
        "step_time_ms_p50": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "status": "running",
        "seed": spec["seed"],
        "allocator_limit_mib": None if spec["dry_run"] else ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": None,
        "within_allocator_limit": None,
        "error_type": None,
        "error_summary": None,
    }


def _validate_success_measurements(record: dict[str, Any], *, dry_run: bool) -> None:
    """Reject a nominally successful row with incomplete measurement evidence."""

    samples = record.get("step_time_ms_samples")
    measurement_steps = record.get("measurement_steps")
    if not isinstance(samples, list) or not isinstance(measurement_steps, int) or len(samples) != measurement_steps:
        raise RuntimeError("checkpoint success row lacks the required raw latency samples")
    if not samples or any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in samples):
        raise RuntimeError("checkpoint success row contains an invalid latency sample")
    p50 = record.get("step_time_ms_p50")
    if not isinstance(p50, (int, float)) or isinstance(p50, bool) or not math.isfinite(p50) or p50 < 0:
        raise RuntimeError("checkpoint success row lacks a valid p50 latency")
    if not math.isclose(float(p50), statistics.median(samples), rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError("checkpoint p50 does not match its raw latency samples")

    if dry_run:
        return
    allocated = record.get("peak_allocated_mib")
    reserved = record.get("peak_reserved_mib")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in (allocated, reserved)):
        raise RuntimeError("checkpoint success row lacks valid CUDA peak-memory evidence")
    if float(allocated) > float(reserved) or float(reserved) > ALLOCATOR_LIMIT_MIB:
        raise RuntimeError("checkpoint success row violates the allocator-memory contract")
    if record.get("within_allocator_limit") is not True:
        raise RuntimeError("checkpoint success row lacks an affirmative allocator-limit result")


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute exactly one checkpoint configuration in the current process."""

    import torch

    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
    from cs336_systems.a2k.checkpointing import CheckpointedTransformerLM

    record = _base_record(spec)
    guard = None
    model = None
    optimizer = None
    try:
        guard = prepare_runtime(
            dry_run=spec["dry_run"],
            tf32_enabled=False,
            development_cuda=spec["development_cuda"],
        )
        device = guard.device
        record["runtime"] = guard.metadata
        record["allocator_fraction"] = guard.metadata["allocator"]["allocator_fraction"]
        record["allocator_limit_mib"] = guard.metadata["allocator"]["allocator_limit_mib"]

        torch.manual_seed(spec["seed"])
        if device.type == "cuda":
            torch.cuda.manual_seed_all(spec["seed"])
        else:
            torch.set_num_threads(1)

        model = BasicsTransformerLM(
            vocab_size=spec["vocab_size"],
            context_length=spec["context_length"],
            d_model=spec["d_model"],
            num_layers=spec["num_layers"],
            num_heads=spec["num_heads"],
            d_ff=spec["d_ff"],
        ).to(device)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("checkpoint benchmark parameters must remain FP32")
        record["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        record["parameter_dtype"] = "fp32"
        model = CheckpointedTransformerLM(model, spec["checkpoint_block_size"])
        model.train()
        optimizer = AdamW(model.parameters(), lr=spec["learning_rate"])

        input_ids = torch.randint(
            0,
            spec["vocab_size"],
            (spec["batch_size"], spec["context_length"]),
            device=device,
        )
        targets = torch.randint(
            0,
            spec["vocab_size"],
            (spec["batch_size"], spec["context_length"]),
            device=device,
        )

        for _ in range(spec["warmup_steps"]):
            _run_training_step(model, optimizer, input_ids, targets, device=device, dtype=spec["dtype"])

        optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        reset_peak_memory(device)
        samples = [_run_training_step(model, optimizer, input_ids, targets, device=device, dtype=spec["dtype"]) for _ in range(spec["measurement_steps"])]
        memory = peak_memory_mib(device)
        reserved = memory["peak_reserved_mib"]
        within_limit = None if reserved is None else reserved <= ALLOCATOR_LIMIT_MIB
        record.update(
            step_time_ms_samples=[round(value, 6) for value in samples],
            step_time_ms_p50=round(statistics.median(samples), 6),
            **memory,
            within_allocator_limit=within_limit,
        )
        _validate_success_measurements(record, dry_run=spec["dry_run"])
        record["status"] = "ok"
    except Exception as exc:  # The matrix must retain honest OOM/failure rows.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        error = public_error(exc)
        record.update(
            status="oom" if error["category"] == "out_of_memory" else "error",
            error_type=error["type"],
            error_summary=error["message"],
        )
        if guard is not None:
            record.update(peak_memory_mib(guard.device))
    finally:
        optimizer = None
        model = None
        if guard is not None:
            release_memory(guard.device)
    return record


def _case_spec(args: argparse.Namespace, *, context_length: int, block_size: int) -> dict[str, Any]:
    if args.dry_run:
        dimensions = {"model_size": "dry_run", "d_model": 32, "d_ff": 64, "num_layers": 8, "num_heads": 4, "vocab_size": 64}
    else:
        dimensions = {"model_size": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16, "vocab_size": 10_000}
    config_id = f"{dimensions['model_size']}_ctx{context_length}_ckpt{block_size}"
    return {
        "config_id": config_id,
        **dimensions,
        "context_length": context_length,
        "batch_size": 1,
        "dtype": "fp32" if args.dry_run else "bf16",
        "checkpoint_block_size": block_size,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "dry_run": args.dry_run,
        "development_cuda": args.development_cuda,
    }


def _run_isolated(spec: dict[str, Any], *, runtime_root: Path, case_namespace: str) -> dict[str, Any]:
    # The per-run namespace prevents a crashed retry from being mistaken for a
    # valid result left by an earlier invocation.
    case_runtime = runtime_root / "cases" / case_namespace
    result_path = case_runtime / "result.json"
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.checkpointing",
        "--worker-spec",
        json.dumps(spec, separators=(",", ":")),
        "--worker-result",
        str(result_path),
    ]
    environment = child_process_environment(case_namespace)
    if spec["dry_run"]:
        environment["OMP_NUM_THREADS"] = "1"
    case_runtime.mkdir(parents=True, exist_ok=True)
    with (case_runtime / "private_stderr.txt").open("w", encoding="utf-8") as private_stderr:
        completed = subprocess.run(
            command,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=private_stderr,
            check=False,
        )
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except (OSError, json.JSONDecodeError):
            pass
    record = _base_record(spec)
    record.update(
        status="error",
        error_type="WorkerProcessError",
        error_summary=f"isolated worker did not produce a valid result (exit code {completed.returncode})",
    )
    return record


def _select_lowest_peak_checkpoint(rows: list[dict[str, Any]], *, dry_run: bool) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["checkpoint_block_size"] > 0 and row["status"] == "ok"]
    numeric = [row for row in candidates if isinstance(row.get("peak_allocated_mib"), (int, float))]
    if numeric:
        return min(numeric, key=lambda row: (row["peak_allocated_mib"], row["checkpoint_block_size"]))
    if dry_run and candidates:
        return min(candidates, key=lambda row: row["checkpoint_block_size"])
    return None


def _valid_boundary_results(rows: list[dict[str, Any]], *, context_length: int) -> bool:
    """Require one honest baseline and one successful checkpoint boundary row."""

    boundary = [row for row in rows if row.get("context_length") == context_length]
    baselines = [row for row in boundary if row.get("checkpoint_block_size") == 0]
    checkpointed = [row for row in boundary if isinstance(row.get("checkpoint_block_size"), int) and row["checkpoint_block_size"] > 0]
    return len(boundary) == 2 and len(baselines) == 1 and baselines[0].get("status") in {"ok", "oom"} and len(checkpointed) == 1 and checkpointed[0].get("status") == "ok"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A2-K activation-checkpointing benchmark")
    parser.add_argument("--runtime-dir", type=Path, default=Path("local_results/a2k/checkpointing-runtime"))
    parser.add_argument("--json-output", type=Path, default=Path("local_results/a2k/checkpointing.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("local_results/a2k/checkpointing.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument(
        "--development-cuda",
        action="store_true",
        help="allow a larger single GPU for non-authoritative development evidence",
    )
    parser.add_argument("--dry-run", action="store_true", help="run a tiny FP32 CPU matrix; never formal evidence")
    parser.add_argument("--worker-spec", help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.seed < 0 or args.learning_rate <= 0:
        parser.error("--seed must be non-negative and --learning-rate must be positive")
    if args.dry_run:
        args.warmup_steps = 1
        args.measurement_steps = 1
    elif args.warmup_steps < 3 or args.measurement_steps < 5:
        parser.error("formal runs require at least 3 warm-up and 5 measurement steps")
    if args.dry_run and args.development_cuda:
        parser.error("--dry-run and --development-cuda are mutually exclusive")


def _worker_main(args: argparse.Namespace) -> int:
    if args.worker_result is None:
        raise SystemExit("--worker-result is required with --worker-spec")
    try:
        spec = json.loads(args.worker_spec)
        if not isinstance(spec, dict):
            raise ValueError("worker spec must be an object")
        record = run_worker(spec)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        error = public_error(exc)
        record = {"status": "error", "error_type": error["type"], "error_summary": error["message"]}
    assert_public_payload(record)
    atomic_write_json(args.worker_result, record)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.worker_spec is not None:
        return _worker_main(args)
    _validate_args(parser, args)

    runtime_root = args.runtime_dir.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    run_namespace = f"checkpoint-{time.time_ns()}"
    context_1024 = 8 if args.dry_run else 1024
    context_2048 = 16 if args.dry_run else 2048
    for block_size in FORMAL_BLOCK_SIZES:
        spec = _case_spec(args, context_length=context_1024, block_size=block_size)
        row = _run_isolated(spec, runtime_root=runtime_root, case_namespace=f"{run_namespace}-{block_size}")
        rows.append(row)
        print(f"{row.get('config_id', spec['config_id'])}: {row['status']}")

    selected = _select_lowest_peak_checkpoint(rows, dry_run=args.dry_run)
    boundary_blocks = [0]
    if selected is not None:
        boundary_blocks.append(int(selected["checkpoint_block_size"]))
    for block_size in boundary_blocks:
        spec = _case_spec(args, context_length=context_2048, block_size=block_size)
        row = _run_isolated(spec, runtime_root=runtime_root, case_namespace=f"{run_namespace}-boundary-{block_size}")
        rows.append(row)
        print(f"{row.get('config_id', spec['config_id'])}: {row['status']}")

    payload = {
        "schema_version": 1,
        "benchmark": "activation_checkpointing",
        "formal_evidence": (not args.dry_run and not args.development_cuda and all(row.get("runtime", {}).get("authoritative") is True for row in rows)),
        "process_isolation": "one fresh Python process per configuration, executed serially",
        "selection": {
            "context_2048_checkpoint_block_size": selected["checkpoint_block_size"] if selected else None,
            "criterion": "lowest successful context-1024 checkpointed peak_allocated_mib",
        },
        "measurement_contract": {
            "model_parameters": "fp32",
            "autocast": "bf16" if not args.dry_run else "disabled_cpu_dry_run",
            "optimizer": "cs336_basics.optimizer.AdamW",
            "initialization_and_input_generation_timed": False,
            "synchronize_before_and_after_each_step": not args.dry_run,
            "peak_reset_after_warmup": True,
        },
        "results": rows,
    }
    assert_public_payload(payload)
    atomic_write_json(args.json_output, payload)
    _write_public_csv(args.csv_output, rows)
    print(f"wrote {len(rows)} rows")

    valid = selected is not None and _valid_boundary_results(rows, context_length=context_2048)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
