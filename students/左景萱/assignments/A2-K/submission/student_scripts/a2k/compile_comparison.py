#!/usr/bin/env python3
"""Measure eager and ``torch.compile`` cold/steady behavior for A2-K.

Every workload/shape/implementation/phase tuple runs in a fresh process with a
private compiler cache.  Consequently ``cold_start_ms`` cannot silently become
a disk-cache hit from a preceding row, while initialization and random input
creation remain outside both cold and steady timing regions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
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


ATTENTION_SHAPES = ((512, 64), (2048, 128), (8192, 128))
ATTENTION_PHASES = ("forward", "backward", "forward_backward")
MODEL_PHASES = ("forward", "forward_backward", "training_step")
IMPLEMENTATIONS = ("eager", "compiled")
ATTENTION_WARMUP_MS = 100
ATTENTION_REP_MS = 300
PUBLIC_COLUMNS = (
    "config_id",
    "workload",
    "model_size",
    "batch_size",
    "sequence_length",
    "context_length",
    "head_dim",
    "dtype",
    "causal",
    "implementation",
    "phase",
    "warmup_steps",
    "measurement_steps",
    "timer",
    "warmup_ms",
    "rep_ms",
    "measurement_count",
    "first_call_ms",
    "cold_start_ms",
    "cold_forward_setup_ms",
    "cold_phase_ms",
    "steady_ms_samples",
    "steady_ms_p20",
    "steady_ms_p50",
    "steady_ms_p80",
    "cold_peak_allocated_mib",
    "cold_peak_reserved_mib",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "compile_backend",
    "compile_fullgraph",
    "compile_dynamic",
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


def _quantile(samples: list[float], probability: float) -> float:
    if not samples:
        raise ValueError("quantile requires at least one sample")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _time_call(function: Callable[[], Any], *, device: Any) -> tuple[float, Any]:
    synchronize(device)
    start = time.perf_counter()
    result = function()
    synchronize(device)
    return (time.perf_counter() - start) * 1_000, result


def _clear_tensor_gradients(tensors: list[Any]) -> None:
    for tensor in tensors:
        tensor.grad = None


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _validate_success_measurements(record: dict[str, Any], *, dry_run: bool) -> None:
    """Reject an ``ok`` row unless its timing and memory evidence is complete."""

    samples = record.get("steady_ms_samples")
    count = record.get("measurement_count")
    if not isinstance(samples, list) or not isinstance(count, int) or count <= 0 or len(samples) != count:
        raise RuntimeError("compile success row lacks the required steady latency samples")
    if any(not _finite_nonnegative(value) for value in samples):
        raise RuntimeError("compile success row contains an invalid steady latency sample")
    quantiles = [record.get(name) for name in ("steady_ms_p20", "steady_ms_p50", "steady_ms_p80")]
    if any(not _finite_nonnegative(value) for value in quantiles) or quantiles != sorted(quantiles):
        raise RuntimeError("compile success row lacks valid ordered latency quantiles")
    if not _finite_nonnegative(record.get("first_call_ms")) or not _finite_nonnegative(record.get("cold_phase_ms")):
        raise RuntimeError("compile success row lacks cold-call timing evidence")
    if record.get("phase") == "backward" and not _finite_nonnegative(record.get("cold_forward_setup_ms")):
        raise RuntimeError("compile backward row lacks cold forward-setup timing")
    if record.get("implementation") == "compiled" and not _finite_nonnegative(record.get("cold_start_ms")):
        raise RuntimeError("compiled success row lacks cold-start timing")

    if dry_run:
        return
    for prefix in ("cold_peak", "peak"):
        allocated = record.get(f"{prefix}_allocated_mib")
        reserved = record.get(f"{prefix}_reserved_mib")
        if not _finite_nonnegative(allocated) or not _finite_nonnegative(reserved):
            raise RuntimeError("compile success row lacks valid CUDA peak-memory evidence")
        if float(allocated) > float(reserved) or float(reserved) > ALLOCATOR_LIMIT_MIB:
            raise RuntimeError("compile success row violates the allocator-memory contract")
    if record.get("within_allocator_limit") is not True:
        raise RuntimeError("compile success row lacks an affirmative allocator-limit result")


def _release_cold_allocator_cache(device: Any) -> None:
    """Drop unused cold-start blocks before the steady-state warm-up."""

    if device.type != "cuda":
        return
    import torch

    synchronize(device)
    torch.cuda.empty_cache()
    synchronize(device)


def _base_record(spec: dict[str, Any]) -> dict[str, Any]:
    compiled = spec["implementation"] == "compiled"
    return {
        "config_id": spec["config_id"],
        "workload": spec["workload"],
        "model_size": spec.get("model_size"),
        "batch_size": spec["batch_size"],
        "sequence_length": spec.get("sequence_length"),
        "context_length": spec.get("context_length"),
        "head_dim": spec.get("head_dim"),
        "dtype": spec["dtype"],
        "causal": spec.get("causal", False),
        "implementation": spec["implementation"],
        "phase": spec["phase"],
        "warmup_steps": spec["warmup_steps"],
        "measurement_steps": spec["measurement_steps"],
        "timer": spec["timer"],
        "warmup_ms": spec["warmup_ms"],
        "rep_ms": spec["rep_ms"],
        "measurement_count": None,
        "first_call_ms": None,
        "cold_start_ms": None,
        "cold_forward_setup_ms": None,
        "cold_phase_ms": None,
        "steady_ms_samples": [],
        "steady_ms_p20": None,
        "steady_ms_p50": None,
        "steady_ms_p80": None,
        "cold_peak_allocated_mib": None,
        "cold_peak_reserved_mib": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "compile_backend": spec["compile_backend"] if compiled else None,
        "compile_fullgraph": spec["compile_fullgraph"] if compiled else False,
        "compile_dynamic": False,
        "status": "running",
        "seed": spec["seed"],
        "allocator_limit_mib": None if spec["dry_run"] else ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": None,
        "within_allocator_limit": None,
        "error_type": None,
        "error_summary": None,
    }


class _AttentionWorkload:
    def __init__(self, spec: dict[str, Any], device: Any) -> None:
        import torch

        from cs336_systems.a2k.attention import eager_attention

        self.device = device
        self.phase = spec["phase"]
        dtype = torch.bfloat16 if spec["dtype"] == "bf16" else torch.float32
        shape = (spec["batch_size"], spec["sequence_length"], spec["head_dim"])
        requires_grad = self.phase != "forward"
        self.q = torch.randn(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
        self.k = torch.randn(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
        self.v = torch.randn(shape, dtype=dtype, device=self.device, requires_grad=requires_grad)
        self.grad_output = torch.randn(shape, dtype=dtype, device=self.device) if requires_grad else None
        self.inputs = [self.q, self.k, self.v]

        def operation(q: Any, k: Any, v: Any) -> Any:
            return eager_attention(q, k, v, is_causal=True)

        self.operation = (
            torch.compile(
                operation,
                backend=spec["compile_backend"],
                fullgraph=spec["compile_fullgraph"],
                dynamic=False,
            )
            if spec["implementation"] == "compiled"
            else operation
        )

    def forward(self) -> Any:
        import torch

        with torch.inference_mode():
            return self.operation(self.q, self.k, self.v)

    def forward_backward(self) -> None:
        _clear_tensor_gradients(self.inputs)
        self.forward_backward_prepared()

    def forward_backward_prepared(self) -> None:
        output = self.operation(self.q, self.k, self.v)
        output.backward(self.grad_output)

    def prepare_backward(self) -> Any:
        _clear_tensor_gradients(self.inputs)
        output = self.operation(self.q, self.k, self.v)
        synchronize(self.device)
        return output

    def backward(self, output: Any) -> None:
        output.backward(self.grad_output)


class _ModelWorkload:
    def __init__(self, spec: dict[str, Any], device: Any) -> None:
        import torch

        from cs336_basics.model import BasicsTransformerLM
        from cs336_basics.optimizer import AdamW

        config = spec["model_config"]
        self.device = device
        self.dtype = spec["dtype"]
        self.phase = spec["phase"]
        model = BasicsTransformerLM(
            vocab_size=spec["vocab_size"],
            context_length=spec["context_length"],
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
        ).to(self.device)
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise RuntimeError("model parameters must remain FP32 under BF16 autocast")
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.model = (
            torch.compile(
                model,
                backend=spec["compile_backend"],
                fullgraph=spec["compile_fullgraph"],
                dynamic=False,
            )
            if spec["implementation"] == "compiled"
            else model
        )
        self.model.train(self.phase != "forward")
        self.input_ids = torch.randint(
            0,
            spec["vocab_size"],
            (spec["batch_size"], spec["context_length"]),
            device=self.device,
        )
        self.targets = torch.randint(
            0,
            spec["vocab_size"],
            (spec["batch_size"], spec["context_length"]),
            device=self.device,
        )
        self.optimizer = AdamW(self.model.parameters(), lr=spec["learning_rate"]) if self.phase == "training_step" else None

    def _autocast(self):
        import contextlib
        import torch

        if self.dtype == "bf16":
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def forward(self) -> Any:
        import torch

        with torch.inference_mode(), self._autocast():
            return self.model(self.input_ids)

    def forward_backward(self) -> None:
        import torch.nn.functional as F

        self.model.zero_grad(set_to_none=True)
        with self._autocast():
            logits = self.model(self.input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.targets.reshape(-1))
        loss.backward()

    def training_step(self) -> None:
        import torch.nn.functional as F

        if self.optimizer is None:
            raise RuntimeError("training_step requires an optimizer")
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            logits = self.model(self.input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.targets.reshape(-1))
        loss.backward()
        self.optimizer.step()


def _first_call(workload: Any, phase: str, *, device: Any) -> dict[str, float]:
    """Time cold execution while exposing backward's untimed steady setup."""

    if phase == "backward":
        synchronize(device)
        overall_start = time.perf_counter()
        setup_start = overall_start
        output = workload.prepare_backward()
        setup_ms = (time.perf_counter() - setup_start) * 1_000
        phase_start = time.perf_counter()
        workload.backward(output)
        synchronize(device)
        phase_ms = (time.perf_counter() - phase_start) * 1_000
        return {
            "first_call_ms": (time.perf_counter() - overall_start) * 1_000,
            "cold_forward_setup_ms": setup_ms,
            "cold_phase_ms": phase_ms,
        }
    elapsed, result = _time_call(getattr(workload, phase), device=device)
    del result
    return {"first_call_ms": elapsed, "cold_phase_ms": elapsed}


def _steady_call(workload: Any, phase: str, *, device: Any) -> float:
    if phase == "backward":
        output = workload.prepare_backward()
        elapsed, _ = _time_call(lambda: workload.backward(output), device=device)
        return elapsed
    elapsed, result = _time_call(getattr(workload, phase), device=device)
    del result
    return elapsed


def _cuda_event_sample(workload: _AttentionWorkload, phase: str, *, device: Any) -> float:
    """Measure one attention phase with a fresh CUDA-event interval."""

    import torch

    if device.type != "cuda":
        raise RuntimeError("CUDA event timing requires CUDA")
    backward_output = None
    if phase == "backward":
        backward_output = workload.prepare_backward()
    elif phase == "forward_backward":
        _clear_tensor_gradients(workload.inputs)

    synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    if phase == "backward":
        workload.backward(backward_output)
        result = None
    elif phase == "forward_backward":
        result = workload.forward_backward_prepared()
    else:
        result = workload.forward()
    end.record()
    end.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    del result, backward_output
    return elapsed_ms


def _cuda_event_window(
    workload: _AttentionWorkload,
    phase: str,
    *,
    device: Any,
    duration_ms: int,
) -> list[float]:
    """Collect samples until measured CUDA event time reaches the contract."""

    if duration_ms <= 0:
        raise ValueError("CUDA event duration must be positive")
    samples: list[float] = []
    measured_ms = 0.0
    while measured_ms < duration_ms:
        sample = _cuda_event_sample(workload, phase, device=device)
        if not math.isfinite(sample) or sample <= 0:
            raise RuntimeError("CUDA event returned a non-positive or non-finite duration")
        samples.append(sample)
        measured_ms += sample
    return samples


def run_worker(spec: dict[str, Any]) -> dict[str, Any]:
    """Execute one isolated compile-comparison case."""

    import torch

    record = _base_record(spec)
    guard = None
    workload = None
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

        workload = _AttentionWorkload(spec, device) if spec["workload"] == "attention" else _ModelWorkload(spec, device)
        if isinstance(workload, _ModelWorkload):
            record["parameter_count"] = workload.parameter_count
            record["parameter_dtype"] = "fp32"

        reset_peak_memory(device)
        first = _first_call(workload, spec["phase"], device=device)
        record.update({key: round(value, 6) for key, value in first.items()})
        if spec["implementation"] == "compiled":
            record["cold_start_ms"] = record["first_call_ms"]
        cold_memory = peak_memory_mib(device)
        record["cold_peak_allocated_mib"] = cold_memory["peak_allocated_mib"]
        record["cold_peak_reserved_mib"] = cold_memory["peak_reserved_mib"]
        _release_cold_allocator_cache(device)

        if spec["timer"] == "cuda_event_duration":
            if not isinstance(workload, _AttentionWorkload):
                raise RuntimeError("CUDA event duration protocol is reserved for attention microbenchmarks")
            _cuda_event_window(
                workload,
                spec["phase"],
                device=device,
                duration_ms=spec["warmup_ms"],
            )
            reset_peak_memory(device)
            samples = _cuda_event_window(
                workload,
                spec["phase"],
                device=device,
                duration_ms=spec["rep_ms"],
            )
        else:
            for _ in range(spec["warmup_steps"]):
                _steady_call(workload, spec["phase"], device=device)
            reset_peak_memory(device)
            samples = [_steady_call(workload, spec["phase"], device=device) for _ in range(spec["measurement_steps"])]
        memory = peak_memory_mib(device)
        reserved = memory["peak_reserved_mib"]
        cold_reserved = record["cold_peak_reserved_mib"]
        observed_reserved = [value for value in (reserved, cold_reserved) if value is not None]
        within_limit = None if not observed_reserved else all(value <= ALLOCATOR_LIMIT_MIB for value in observed_reserved)
        record.update(
            steady_ms_samples=[round(value, 6) for value in samples],
            steady_ms_p20=round(_quantile(samples, 0.2), 6),
            steady_ms_p50=round(statistics.median(samples), 6),
            steady_ms_p80=round(_quantile(samples, 0.8), 6),
            measurement_count=len(samples),
            **memory,
            within_allocator_limit=within_limit,
        )
        _validate_success_measurements(record, dry_run=spec["dry_run"])
        record["status"] = "ok"
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        error = public_error(exc)
        record.update(
            status="oom" if error["category"] == "out_of_memory" else "compile_error" if spec["implementation"] == "compiled" else "error",
            error_type=error["type"],
            error_summary=error["message"],
        )
        if guard is not None:
            record.update(peak_memory_mib(guard.device))
    finally:
        workload = None
        if guard is not None:
            release_memory(guard.device)
    return record


def _common_spec(args: argparse.Namespace, *, config_id: str, implementation: str, phase: str) -> dict[str, Any]:
    return {
        "config_id": config_id,
        "implementation": implementation,
        "phase": phase,
        "dtype": "fp32" if args.dry_run else "bf16",
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "timer": "synchronized_perf_counter",
        "warmup_ms": None,
        "rep_ms": None,
        "seed": args.seed,
        "compile_backend": "eager" if args.dry_run else args.compile_backend,
        "compile_fullgraph": args.compile_fullgraph,
        "dry_run": args.dry_run,
        "development_cuda": args.development_cuda,
    }


def build_case_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    attention_shapes = ((8, 4),) if args.dry_run else ATTENTION_SHAPES
    for sequence_length, head_dim in attention_shapes:
        for implementation in IMPLEMENTATIONS:
            for phase in ATTENTION_PHASES:
                config_id = f"attention_s{sequence_length}_d{head_dim}_{implementation}_{phase}"
                specs.append(
                    {
                        **_common_spec(args, config_id=config_id, implementation=implementation, phase=phase),
                        "workload": "attention",
                        "model_size": None,
                        "batch_size": 1,
                        "sequence_length": sequence_length,
                        "context_length": None,
                        "head_dim": head_dim,
                        "causal": True,
                        "timer": "synchronized_perf_counter" if args.dry_run else "cuda_event_duration",
                        "warmup_steps": args.warmup_steps if args.dry_run else None,
                        "measurement_steps": args.measurement_steps if args.dry_run else None,
                        "warmup_ms": None if args.dry_run else ATTENTION_WARMUP_MS,
                        "rep_ms": None if args.dry_run else ATTENTION_REP_MS,
                    }
                )

    if args.dry_run:
        model_size = "dry_run"
        context_length = 8
        model_config = {"d_model": 32, "d_ff": 64, "num_layers": 2, "num_heads": 4}
        vocab_size = 64
    else:
        model_size = "small"
        context_length = 512
        model_config = {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12}
        vocab_size = 10_000
    for implementation in IMPLEMENTATIONS:
        for phase in MODEL_PHASES:
            config_id = f"model_{model_size}_ctx{context_length}_{implementation}_{phase}"
            specs.append(
                {
                    **_common_spec(args, config_id=config_id, implementation=implementation, phase=phase),
                    "workload": "transformer",
                    "model_size": model_size,
                    "batch_size": 1,
                    "sequence_length": None,
                    "context_length": context_length,
                    "head_dim": None,
                    "causal": True,
                    "model_config": model_config,
                    "vocab_size": vocab_size,
                    "learning_rate": args.learning_rate,
                }
            )
    return specs


def _run_isolated(spec: dict[str, Any], *, runtime_root: Path, case_namespace: str) -> dict[str, Any]:
    # Never accept a stale row if a later isolated worker dies before writing.
    case_runtime = runtime_root / "cases" / case_namespace
    result_path = case_runtime / "result.json"
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.compile_comparison",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A2-K eager/torch.compile comparison")
    parser.add_argument("--runtime-dir", type=Path, default=Path("local_results/a2k/compile-runtime"))
    parser.add_argument("--json-output", type=Path, default=Path("local_results/a2k/compile_comparison.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("local_results/a2k/compile_comparison.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--compile-backend", default="inductor")
    parser.add_argument("--compile-fullgraph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--development-cuda",
        action="store_true",
        help="allow a larger single GPU for non-authoritative development evidence",
    )
    parser.add_argument("--dry-run", action="store_true", help="run tiny FP32 CPU cases with the eager compile backend")
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
    run_namespace = f"compile-{time.time_ns()}"
    for case_index, spec in enumerate(build_case_specs(args), start=1):
        row = _run_isolated(spec, runtime_root=runtime_root, case_namespace=f"{run_namespace}-{case_index:02d}")
        rows.append(row)
        print(f"{row.get('config_id', spec['config_id'])}: {row['status']}")

    payload = {
        "schema_version": 1,
        "benchmark": "torch_compile_comparison",
        "formal_evidence": (not args.dry_run and not args.development_cuda and all(row.get("runtime", {}).get("authoritative") is True for row in rows)),
        "process_isolation": "one fresh Python process and private compiler cache per row, executed serially",
        "measurement_contract": {
            "initialization_and_input_generation_timed": False,
            "first_compiled_execution_reported_as_cold_start": True,
            "cold_start_excluded_from_warmup_and_steady_samples": True,
            "backward_steady_forward_setup_timed": False,
            "cuda_synchronize_at_timing_boundaries": not args.dry_run,
            "attention_steady_timer": "synchronized CUDA events" if not args.dry_run else "synchronized perf_counter CPU dry-run",
            "attention_warmup_ms": ATTENTION_WARMUP_MS if not args.dry_run else None,
            "attention_rep_ms": ATTENTION_REP_MS if not args.dry_run else None,
            "attention_duration_accumulation": ("sum of measured CUDA event intervals; untimed backward setup excluded" if not args.dry_run else None),
            "model_warmup_steps": args.warmup_steps,
            "model_measurement_steps": args.measurement_steps,
            "compiled_scope": "explicit attention operation or Transformer forward; cs336_basics AdamW remains eager",
            "training_step_cold_includes_lazy_optimizer_state_initialization": True,
            "dtype": "bf16" if not args.dry_run else "fp32_cpu_dry_run",
            "compile_dynamic": False,
            "cold_allocator_cache_released_before_steady_warmup": not args.dry_run,
        },
        "results": rows,
    }
    assert_public_payload(payload)
    atomic_write_json(args.json_output, payload)
    _write_public_csv(args.csv_output, rows)
    print(f"wrote {len(rows)} rows")
    return 0 if rows and all(row["status"] == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
