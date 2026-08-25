#!/usr/bin/env python3
"""Run the A2-P profiling matrix once across an isolated single-node allocation.

This program never submits a cluster job.  It consumes an existing allocation,
assigns at most one child to each visible GPU selector, and records no selector
or device UUID in its manifest.  Every case is launched exactly once; the only
conditional work is a new, explicitly named memory fallback case after an XL
context-2048 training OOM.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "cs336.a2p.run-suite.v1"
MANIFEST_NAME = "manifest.json"
SUCCESS_MARKER = Path("markers/suite.success.json")
COMPLETION_MARKER = Path("markers/suite.complete.json")
MODEL_SIZES = ("small", "medium", "large", "xl", "10b")
PROFILE_MODELS = ("small", "xl")
PROFILE_CONTEXTS = (256, 512, 1_024)
MEMORY_CONTEXTS = (128, 2_048)
MEMORY_MODES = ("forward", "train_step")
PRECISIONS = ("fp32", "bf16")
SAFE_CASE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SAFE_GPU_SELECTOR = re.compile(r"^(?:0|[1-9][0-9]*|GPU-[A-Za-z0-9][A-Za-z0-9._-]*|MIG-[A-Za-z0-9][A-Za-z0-9._:/-]*)$")

PASSTHROUGH_VARIABLES = (
    "CUDA_DEVICE_ORDER",
    "CUDA_MODULE_LOADING",
    "LANG",
    "LC_ALL",
    "MKL_NUM_THREADS",
    "NVIDIA_DRIVER_CAPABILITIES",
    "OMP_NUM_THREADS",
    "TERM",
    "TZ",
)

ISOLATED_PATHS = {
    "HOME": "home",
    "XDG_CACHE_HOME": "cache/xdg",
    "XDG_CONFIG_HOME": "cache/xdg-config",
    "XDG_DATA_HOME": "cache/xdg-data",
    "XDG_STATE_HOME": "cache/xdg-state",
    "TMPDIR": "tmp",
    "TMP": "tmp",
    "TEMP": "tmp",
    "PYTHONPYCACHEPREFIX": "cache/pycache",
    "CUDA_CACHE_PATH": "cache/cuda",
    "TORCHINDUCTOR_CACHE_DIR": "cache/torchinductor",
    "TRITON_CACHE_DIR": "cache/triton",
    "TORCH_EXTENSIONS_DIR": "cache/torch-extensions",
    "TORCH_HOME": "cache/torch",
    "HF_HOME": "cache/huggingface",
    "HF_DATASETS_CACHE": "cache/huggingface/datasets",
    "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub",
    "TRANSFORMERS_CACHE": "cache/huggingface/transformers",
    "MPLCONFIGDIR": "cache/matplotlib",
    "NUMBA_CACHE_DIR": "cache/numba",
    "PIP_CACHE_DIR": "cache/pip",
    "PIP_CONFIG_FILE": "config/pip.conf",
    "UV_CACHE_DIR": "cache/uv",
    "WANDB_DIR": "cache/wandb",
    "CCACHE_DIR": "cache/ccache",
    "CUPY_CACHE_DIR": "cache/cupy",
    "JAX_COMPILATION_CACHE_DIR": "cache/jax",
}


@dataclass(frozen=True)
class SuiteConfig:
    runtime_root: Path
    python: Path
    project_root: Path
    expected_gpus: int
    seed: int
    timeout_seconds: float
    termination_grace_seconds: float
    dry_run: bool
    visible_gpu_count: int
    visibility_status: str


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    stage: str
    command: tuple[str, ...]
    case_root: Path
    workdir: Path
    output_json: Path
    configuration: dict[str, Any]
    fallback_for: str | None = None


CaseRunner = Callable[[CaseSpec, str, SuiteConfig, Mapping[str, str]], dict[str, Any]]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _is_below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_root(path: Path, label: str) -> None:
    root = Path("/root")
    if path == root or _is_below(path, root):
        raise ValueError(f"{label} must not be under /root")


def resolve_runtime_root(raw: str) -> Path:
    if not raw.strip():
        raise ValueError("--runtime-root must be non-empty")
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise ValueError("--runtime-root must be an absolute path")
    resolved = requested.resolve()
    if requested != resolved:
        raise ValueError("--runtime-root must be canonical and symlink-free")
    _reject_root(resolved, "--runtime-root")
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError("--runtime-root must name a new directory")
    return resolved


def resolve_python(raw: str) -> Path:
    if not raw.strip():
        raise ValueError("--python must be non-empty")
    requested = Path(raw).expanduser()
    if not requested.is_absolute():
        raise ValueError("--python must be an absolute executable path")
    resolved = requested.resolve()
    _reject_root(resolved, "--python")
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("--python is missing or is not executable")
    # Preserve a virtual-environment symlink so Python activates that prefix.
    return requested.absolute()


def _split_visible_selectors(raw: str) -> tuple[str, ...]:
    selectors = tuple(item.strip() for item in raw.split(","))
    if not selectors or any(not selector for selector in selectors):
        raise ValueError("visible GPU selectors contain an empty entry")
    if any(SAFE_GPU_SELECTOR.fullmatch(selector) is None for selector in selectors):
        raise ValueError("visible GPU selectors contain an unsupported value")
    normalized = [selector.casefold() for selector in selectors]
    if len(set(normalized)) != len(normalized):
        raise ValueError("visible GPU selectors must be unique")
    return selectors


def discover_visible_gpu_selectors(
    source: Mapping[str, str],
    *,
    expected_gpus: int,
    dry_run: bool,
) -> tuple[tuple[str, ...], int, str]:
    if expected_gpus <= 0:
        raise ValueError("--expected-gpus must be positive")
    raw = source.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw or raw.casefold() == "all":
        fallback = source.get("NVIDIA_VISIBLE_DEVICES", "").strip()
        raw = fallback if fallback and fallback.casefold() != "all" else ""
    if not raw:
        if dry_run:
            # These opaque values are never executed or serialized.
            return tuple(f"dry-run-slot-{index}" for index in range(expected_gpus)), 0, "deferred_dry_run"
        raise ValueError("an explicit CUDA_VISIBLE_DEVICES allocation is required")
    selectors = _split_visible_selectors(raw)
    if len(selectors) != expected_gpus:
        raise ValueError(f"expected exactly {expected_gpus} visible GPUs, got {len(selectors)}")
    return selectors, len(selectors), "validated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the isolated single-node A2-P collection matrix without submitting a job.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runtime-root", required=True, help="new absolute directory for all suite state")
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=6 * 60 * 60)
    parser.add_argument("--termination-grace-seconds", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="write and print the plan without launching children")
    return parser


def resolve_config(
    args: argparse.Namespace,
    *,
    source_environment: Mapping[str, str] | None = None,
) -> tuple[SuiteConfig, tuple[str, ...]]:
    if args.expected_gpus <= 0:
        raise ValueError("--expected-gpus must be positive")
    if not math.isfinite(args.timeout_seconds) or args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be finite and positive")
    if not math.isfinite(args.termination_grace_seconds) or args.termination_grace_seconds < 0:
        raise ValueError("--termination-grace-seconds must be finite and non-negative")
    project_root = PROJECT_ROOT.resolve()
    for name in ("benchmark.py", "mixed_precision.py", "memory_snapshot.py"):
        script = project_root / "profiling" / name
        if not script.is_file() or script.is_symlink():
            raise FileNotFoundError(f"required profiling entry point is missing or unsafe: {name}")
    source = os.environ if source_environment is None else source_environment
    selectors, visible_count, visibility_status = discover_visible_gpu_selectors(
        source,
        expected_gpus=args.expected_gpus,
        dry_run=args.dry_run,
    )
    return (
        SuiteConfig(
            runtime_root=resolve_runtime_root(args.runtime_root),
            python=resolve_python(args.python),
            project_root=project_root,
            expected_gpus=args.expected_gpus,
            seed=args.seed,
            timeout_seconds=args.timeout_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
            dry_run=args.dry_run,
            visible_gpu_count=visible_count,
            visibility_status=visibility_status,
        ),
        selectors,
    )


def _case(
    config: SuiteConfig,
    *,
    stage: str,
    case_id: str,
    script_name: str,
    arguments: Sequence[str],
    output_relative: Path,
    configuration: dict[str, Any],
    fallback_for: str | None = None,
) -> CaseSpec:
    if SAFE_CASE_ID.fullmatch(case_id) is None:
        raise ValueError(f"unsafe case id: {case_id}")
    case_root = config.runtime_root / "commands" / stage / case_id
    workdir = case_root / "work"
    output_json = workdir / output_relative
    if not _is_below(output_json, workdir):
        raise AssertionError("case output escapes its work directory")
    module_name = f"profiling.{Path(script_name).stem}"
    command = (str(config.python), "-m", module_name, *arguments)
    return CaseSpec(
        case_id=case_id,
        stage=stage,
        command=command,
        case_root=case_root,
        workdir=workdir,
        output_json=output_json,
        configuration=configuration,
        fallback_for=fallback_for,
    )


def benchmark_cases(config: SuiteConfig) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for mode in ("forward", "forward_backward", "train_step"):
        case_id = f"small_ctx512_{mode}_warm5"
        cases.append(
            _case(
                config,
                stage="benchmark",
                case_id=case_id,
                script_name="benchmark.py",
                arguments=(
                    "--model-size",
                    "small",
                    "--batch-size",
                    "4",
                    "--context-length",
                    "512",
                    "--mode",
                    mode,
                    "--warmup",
                    "5",
                    "--steps",
                    "10",
                    "--dtype",
                    "fp32",
                    "--seed",
                    str(config.seed),
                    "--device",
                    "cuda",
                    "--profile",
                    "none",
                    "--output",
                    "results/result.json",
                ),
                output_relative=Path("results/result.json"),
                configuration={
                    "model_size": "small",
                    "batch_size": 4,
                    "context_length": 512,
                    "mode": mode,
                    "warmup": 5,
                    "steps": 10,
                    "dtype": "fp32",
                },
            )
        )
    cases.append(
        _case(
            config,
            stage="benchmark",
            case_id="small_ctx512_train_step_warm0",
            script_name="benchmark.py",
            arguments=(
                "--model-size",
                "small",
                "--batch-size",
                "4",
                "--context-length",
                "512",
                "--mode",
                "train_step",
                "--warmup",
                "0",
                "--steps",
                "10",
                "--dtype",
                "fp32",
                "--seed",
                str(config.seed),
                "--device",
                "cuda",
                "--profile",
                "none",
                "--output",
                "results/result.json",
            ),
            output_relative=Path("results/result.json"),
            configuration={
                "model_size": "small",
                "batch_size": 4,
                "context_length": 512,
                "mode": "train_step",
                "warmup": 0,
                "steps": 10,
                "dtype": "fp32",
            },
        )
    )
    return cases


def torch_profile_cases(config: SuiteConfig) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for model_size in PROFILE_MODELS:
        for context_length in PROFILE_CONTEXTS:
            case_id = f"{model_size}_ctx{context_length}_train_step"
            cases.append(
                _case(
                    config,
                    stage="torch_profile",
                    case_id=case_id,
                    script_name="benchmark.py",
                    arguments=(
                        "--model-size",
                        model_size,
                        "--batch-size",
                        "4",
                        "--context-length",
                        str(context_length),
                        "--mode",
                        "train_step",
                        "--warmup",
                        "5",
                        "--steps",
                        "1",
                        "--dtype",
                        "fp32",
                        "--seed",
                        str(config.seed),
                        "--device",
                        "cuda",
                        "--profile",
                        "torch",
                        "--output",
                        "results/result.json",
                    ),
                    output_relative=Path("results/result.json"),
                    configuration={
                        "model_size": model_size,
                        "batch_size": 4,
                        "context_length": context_length,
                        "mode": "train_step",
                        "warmup": 5,
                        "steps": 1,
                        "dtype": "fp32",
                        "profile": "torch",
                    },
                )
            )
    return cases


def mixed_precision_cases(config: SuiteConfig) -> list[CaseSpec]:
    return [
        _case(
            config,
            stage="mixed_precision",
            case_id=f"mixed_{model_size}",
            script_name="mixed_precision.py",
            arguments=(
                "--device",
                "cuda",
                "--model-size",
                model_size,
                "--seed",
                str(config.seed),
                "--output",
                "results/mixed_precision.json",
            ),
            output_relative=Path("results/mixed_precision.json"),
            configuration={"model_size": model_size, "precisions": ["fp32", "bf16"]},
        )
        for model_size in MODEL_SIZES
    ]


def memory_case(
    config: SuiteConfig,
    *,
    model_size: str,
    context_length: int,
    mode: str,
    dtype: str,
    fallback_for: str | None = None,
) -> CaseSpec:
    prefix = "fallback_" if fallback_for is not None else ""
    case_id = f"{prefix}{model_size}_ctx{context_length}_{mode}_{dtype}"
    saved_tensors_block = fallback_for is None and model_size == "xl" and context_length == 128 and mode == "train_step" and dtype == "fp32"
    memory_viz_requested = dtype == "fp32" and ((model_size == "xl" and context_length == 2_048) or fallback_for is not None)
    arguments = [
        "--model-size",
        model_size,
        "--batch-size",
        "1",
        "--context-length",
        str(context_length),
        "--mode",
        mode,
        "--dtype",
        dtype,
        "--warmup",
        "5",
        "--steps",
        "1",
        "--seed",
        str(config.seed),
        "--output",
        "results/result.json",
        "--snapshot-output",
        "private/snapshot.pickle",
        "--timeline-output",
        "results/timeline.png",
    ]
    if memory_viz_requested:
        arguments.extend(("--memory-viz-output", "private/active_memory_timeline.html"))
    if saved_tensors_block:
        arguments.append("--saved-tensors-block")
    return _case(
        config,
        stage="memory_snapshot",
        case_id=case_id,
        script_name="memory_snapshot.py",
        arguments=arguments,
        output_relative=Path("results/result.json"),
        configuration={
            "model_size": model_size,
            "batch_size": 1,
            "context_length": context_length,
            "mode": mode,
            "dtype": dtype,
            "warmup": 5,
            "steps": 1,
            "saved_tensors_block": saved_tensors_block,
            "memory_viz_requested": memory_viz_requested,
        },
        fallback_for=fallback_for,
    )


def memory_cases(config: SuiteConfig) -> list[CaseSpec]:
    return [
        memory_case(
            config,
            model_size="xl",
            context_length=context_length,
            mode=mode,
            dtype=dtype,
        )
        for context_length in MEMORY_CONTEXTS
        for mode in MEMORY_MODES
        for dtype in PRECISIONS
    ]


def build_stage_matrix(config: SuiteConfig) -> dict[str, list[CaseSpec]]:
    stages = {
        "benchmark": benchmark_cases(config),
        "torch_profile": torch_profile_cases(config),
        "mixed_precision": mixed_precision_cases(config),
        "memory_snapshot": memory_cases(config),
    }
    identifiers = [case.case_id for cases in stages.values() for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("suite case identifiers must be globally unique")
    return stages


def _filtered_library_path(source: Mapping[str, str]) -> str | None:
    conda_prefix = source.get("CONDA_PREFIX")
    conda_root = Path(conda_prefix).expanduser().resolve() if conda_prefix else None
    retained: list[str] = []
    for item in source.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        if not item:
            continue
        path = Path(item).expanduser().resolve()
        if path == Path("/root") or _is_below(path, Path("/root")):
            continue
        if conda_root is not None and (path == conda_root or _is_below(path, conda_root)):
            continue
        retained.append(str(path))
    return os.pathsep.join(retained) or None


def build_case_environment(
    spec: CaseSpec,
    *,
    gpu_selector: str,
    config: SuiteConfig,
    source_environment: Mapping[str, str],
) -> dict[str, str]:
    if SAFE_GPU_SELECTOR.fullmatch(gpu_selector) is None:
        raise ValueError("unsafe GPU selector")
    if spec.case_root.exists() or spec.case_root.is_symlink():
        raise FileExistsError(f"case runtime already exists: {spec.case_id}")
    spec.workdir.mkdir(parents=True, mode=0o700, exist_ok=False)
    isolation = spec.case_root / "isolation"
    environment = {key: source_environment[key] for key in PASSTHROUGH_VARIABLES if source_environment.get(key)}
    library_path = _filtered_library_path(source_environment)
    if library_path:
        environment["LD_LIBRARY_PATH"] = library_path
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu_selector,
            "PATH": os.pathsep.join(
                (
                    str(config.python.parent),
                    "/usr/local/cuda/bin",
                    "/usr/local/sbin",
                    "/usr/local/bin",
                    "/usr/sbin",
                    "/usr/bin",
                    "/sbin",
                    "/bin",
                )
            ),
            "PYTHONPATH": os.pathsep.join((str(config.project_root), str(config.project_root / "cs336-basics"))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    for variable, relative in ISOLATED_PATHS.items():
        path = isolation / relative
        if variable == "PIP_CONFIG_FILE":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=False)
        else:
            path.mkdir(parents=True, exist_ok=True)
        environment[variable] = str(path)
    forbidden = [
        key
        for key in environment
        if key.startswith("CONDA_")
        or key
        in {
            "PYTHONHOME",
            "PYTHONUSERBASE",
            "PYTHONSTARTUP",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT",
        }
    ]
    if forbidden:
        raise AssertionError(f"forbidden inherited Python environment: {sorted(forbidden)}")
    for variable in ISOLATED_PATHS:
        if not _is_below(Path(environment[variable]).resolve(), config.runtime_root):
            raise AssertionError(f"{variable} escapes the suite runtime root")
    return environment


def _nonempty_regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file() and path.stat().st_size > 0


def _load_case_result_status(spec: CaseSpec) -> str:
    """Validate the stage-specific evidence contract, not only an exit code."""

    path = spec.output_json
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError("command did not create a non-empty regular result JSON")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("command result is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("status"), str):
        raise ValueError("command result JSON has no string status")
    status = payload["status"]
    expected = spec.configuration

    if spec.stage in {"benchmark", "torch_profile"}:
        configuration = payload.get("configuration")
        results = payload.get("results")
        if payload.get("authoritative") is not True or not isinstance(configuration, dict):
            raise ValueError("benchmark result is not authoritative CUDA evidence")
        if not isinstance(results, list) or len(results) != 1 or results[0].get("status") != "ok":
            raise ValueError("benchmark result does not contain exactly one successful mode")
        checks = {
            "requested_model_size": expected["model_size"],
            "batch_size": expected["batch_size"],
            "context_length": expected["context_length"],
            "modes": [expected["mode"]],
            "warmup_steps": expected["warmup"],
            "measurement_steps": expected["steps"],
            "dtype": expected["dtype"],
            "device": "cuda",
        }
        if any(configuration.get(key) != value for key, value in checks.items()):
            raise ValueError("benchmark result configuration differs from the planned case")
        raw_steps = results[0].get("raw_steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != expected["steps"]:
            raise ValueError("benchmark result has an incomplete raw timing vector")
        if spec.stage == "torch_profile":
            profile = payload.get("profile")
            if not isinstance(profile, dict) or profile.get("tool") != "torch.profiler":
                raise ValueError("profile result has no torch.profiler metadata")
            if profile.get("measured_steps") != 1 or profile.get("warmup_steps_before_measurement", 0) < 5:
                raise ValueError("profile result violates the one-step post-warmup contract")
            trace = path.parent / str(profile.get("trace_file", ""))
            summary = path.parent / str(profile.get("summary_file", ""))
            if trace.parent != path.parent or summary.parent != path.parent:
                raise ValueError("profile artifact path escapes the case result directory")
            if not _nonempty_regular_file(trace) or not _nonempty_regular_file(summary):
                raise ValueError("profile trace or summary artifact is missing")

    elif spec.stage == "mixed_precision":
        benchmark = payload.get("language_model_benchmark")
        toy = payload.get("toy_model")
        accumulation = payload.get("accumulation")
        if payload.get("authoritative") is not True or not isinstance(benchmark, dict):
            raise ValueError("mixed-precision result is not authoritative CUDA evidence")
        if benchmark.get("requested_cuda_model_size") != expected["model_size"]:
            raise ValueError("mixed-precision model size differs from the planned case")
        records = benchmark.get("records")
        if not isinstance(records, dict) or set(records) != set(PRECISIONS):
            raise ValueError("mixed-precision result does not contain both precisions")
        precision_statuses = {record.get("status") for record in records.values() if isinstance(record, dict)}
        if len(precision_statuses) == 0 or not precision_statuses <= {"ok", "oom"}:
            raise ValueError("mixed-precision precision-level status is invalid")
        expected_status = "oom" if "oom" in precision_statuses else "ok"
        if status != expected_status:
            raise ValueError("mixed-precision top-level status hides a precision-level OOM")
        if not isinstance(toy, dict) or toy.get("authoritative_cuda_bf16") is not True:
            raise ValueError("mixed-precision result lacks authoritative ToyModel BF16 evidence")
        accumulation_records = accumulation.get("records") if isinstance(accumulation, dict) else None
        if not isinstance(accumulation_records, list) or len(accumulation_records) != 4:
            raise ValueError("mixed-precision result lacks the four accumulation cases")

    elif spec.stage == "memory_snapshot":
        configuration = payload.get("configuration")
        if payload.get("authoritative") is not True or not isinstance(configuration, dict):
            raise ValueError("memory result is not authoritative CUDA evidence")
        checks = {
            "model_size": expected["model_size"],
            "batch_size": expected["batch_size"],
            "context_length": expected["context_length"],
            "mode": expected["mode"],
            "dtype": expected["dtype"],
        }
        if any(configuration.get(key) != value for key, value in checks.items()):
            raise ValueError("memory result configuration differs from the planned case")
        if status == "ok":
            artifacts = payload.get("artifacts")
            if not isinstance(artifacts, dict) or artifacts.get("snapshot_generated") is not True:
                raise ValueError("successful memory result has no allocator snapshot")
            if artifacts.get("timeline_generated") is not True:
                raise ValueError("successful memory result has no active-memory timeline")
            if artifacts.get("timeline_derived_from_same_snapshot") is not True:
                raise ValueError("memory timeline is not recorded as derived from the allocator snapshot")
            if not _nonempty_regular_file(spec.workdir / "private" / "snapshot.pickle"):
                raise ValueError("memory snapshot artifact is missing")
            if not _nonempty_regular_file(spec.workdir / "results" / "timeline.png"):
                raise ValueError("memory timeline artifact is missing")
            if expected.get("memory_viz_requested"):
                if artifacts.get("memory_viz_generated") is not True:
                    raise ValueError("designated memory case has no official private memory visualization")
                if artifacts.get("memory_viz_basename") != "active_memory_timeline.html":
                    raise ValueError("official memory visualization metadata is malformed")
                if not _nonempty_regular_file(spec.workdir / "private" / "active_memory_timeline.html"):
                    raise ValueError("official private memory visualization artifact is missing")
            if expected.get("saved_tensors_block"):
                saved = payload.get("saved_tensors_block")
                if not isinstance(saved, dict) or saved.get("authoritative") is not True:
                    raise ValueError("designated memory case lacks authoritative saved-tensor evidence")
        elif status == "oom":
            exception = payload.get("exception")
            if not isinstance(exception, dict) or not isinstance(exception.get("type"), str):
                raise ValueError("memory OOM result lacks a typed failure record")
        else:
            raise ValueError("memory result status is neither ok nor oom")
    else:
        raise ValueError(f"unknown suite stage: {spec.stage}")
    return status


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute_case_once(
    spec: CaseSpec,
    gpu_selector: str,
    config: SuiteConfig,
    source_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Launch one child process exactly once and return a selector-free record."""

    started = utc_now()
    log_path = spec.case_root / "logs" / "command.log"
    outcome: dict[str, Any] = {
        "attempts": 1,
        "automatic_retry": False,
        "started_at_utc": started,
        "finished_at_utc": None,
        "returncode": None,
        "status": "launch_error",
        "reported_status": None,
        "log": str(log_path.relative_to(config.runtime_root)),
    }
    try:
        environment = build_case_environment(
            spec,
            gpu_selector=gpu_selector,
            config=config,
            source_environment=source_environment,
        )
        log_path.parent.mkdir(parents=True, exist_ok=False)
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                list(spec.command),
                cwd=spec.workdir,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired:
                _terminate_process_group(process, config.termination_grace_seconds)
                outcome.update(returncode=process.returncode, status="timed_out")
                return outcome
        outcome["returncode"] = returncode
        if returncode != 0:
            outcome["status"] = "failed"
            return outcome
        reported_status = _load_case_result_status(spec)
        outcome["reported_status"] = reported_status
        if reported_status == "ok":
            outcome["status"] = "success"
        elif reported_status == "oom":
            outcome["status"] = "oom"
        else:
            outcome["status"] = "failed"
        return outcome
    except Exception as exc:  # noqa: BLE001 - local log contains private detail; manifest is generic
        outcome.update(
            status="launch_error",
            error_type=type(exc).__name__,
            error="command launch or result validation failed; inspect the local case log",
        )
        return outcome
    finally:
        outcome["finished_at_utc"] = utc_now()


def run_cases_in_batches(
    cases: Sequence[CaseSpec],
    selectors: Sequence[str],
    *,
    config: SuiteConfig,
    source_environment: Mapping[str, str],
    case_runner: CaseRunner = execute_case_once,
) -> dict[str, dict[str, Any]]:
    if len(selectors) != config.expected_gpus:
        raise ValueError("selector count differs from the suite GPU contract")
    outcomes: dict[str, dict[str, Any]] = {}
    for offset in range(0, len(cases), config.expected_gpus):
        batch = list(cases[offset : offset + config.expected_gpus])
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = [
                executor.submit(
                    case_runner,
                    spec,
                    selectors[worker_slot],
                    config,
                    source_environment,
                )
                for worker_slot, spec in enumerate(batch)
            ]
            for worker_slot, (spec, future) in enumerate(zip(batch, futures, strict=True)):
                outcome = future.result()
                outcome["worker_slot"] = worker_slot
                outcomes[spec.case_id] = outcome
    if len(outcomes) != len(cases):
        raise AssertionError("not every case produced exactly one outcome")
    return outcomes


def case_manifest(spec: CaseSpec, config: SuiteConfig) -> dict[str, Any]:
    return {
        "case_id": spec.case_id,
        "stage": spec.stage,
        "configuration": spec.configuration,
        "fallback_for": spec.fallback_for,
        "command": list(spec.command),
        "cwd": str(spec.workdir.relative_to(config.runtime_root)),
        "output_json": str(spec.output_json.relative_to(config.runtime_root)),
        "attempts": 0,
        "automatic_retry": False,
        "worker_slot": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "returncode": None,
        "reported_status": None,
        "status": "planned",
        "log": None,
    }


def build_manifest(config: SuiteConfig, matrix: Mapping[str, Sequence[CaseSpec]]) -> dict[str, Any]:
    expected_matrix = {
        name: {
            "base_command_count": len(cases),
            "case_ids": [case.case_id for case in cases],
        }
        for name, cases in matrix.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned",
        "dry_run": config.dry_run,
        "created_at_utc": utc_now(),
        "finished_at_utc": None,
        "runtime_root": str(config.runtime_root),
        "python": str(config.python),
        "expected_matrix": {
            "base_command_count": sum(len(cases) for cases in matrix.values()),
            "stages": expected_matrix,
            "conditional_memory_fallbacks_excluded": True,
        },
        "success_marker": {
            "path": SUCCESS_MARKER.as_posix(),
            "created": False,
        },
        "completion_marker": {
            "path": COMPLETION_MARKER.as_posix(),
            "created": False,
        },
        "gpu": {
            "expected_count": config.expected_gpus,
            "visible_count": config.visible_gpu_count,
            "visibility_status": config.visibility_status,
            "selectors_recorded": False,
        },
        "execution_contract": {
            "single_node": True,
            "one_process_per_gpu": True,
            "batch_parallelism": config.expected_gpus,
            "each_command_launched_once": True,
            "automatic_retry": False,
            "credentials_inherited": False,
            "conda_and_user_python_state_inherited": False,
            "case_local_home_tmp_and_caches": True,
            "formal_child_commands_use_dry_run": False,
            "project_root_is_first_on_pythonpath": True,
        },
        "fallback_policy": {
            "trigger": "each required XL context-2048 status=oom, independently per mode and precision",
            "round_1": "same mode/precision at XL context-1024, once",
            "round_2": "same mode/precision at Large context-2048, once only if round_1 also reports oom",
            "original_case_retried": False,
        },
        "stages": [
            {
                "name": name,
                "status": "planned",
                "started_at_utc": None,
                "finished_at_utc": None,
                "commands": [case_manifest(spec, config) for spec in cases],
            }
            for name, cases in matrix.items()
        ],
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _stage_record(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [stage for stage in manifest["stages"] if stage["name"] == name]
    if len(matches) != 1:
        raise AssertionError(f"manifest has no unique stage {name}")
    return matches[0]


def _apply_outcomes(stage: dict[str, Any], outcomes: Mapping[str, Mapping[str, Any]]) -> None:
    records = {record["case_id"]: record for record in stage["commands"]}
    if set(outcomes) - set(records):
        raise AssertionError("outcome references an unplanned case")
    for case_id, outcome in outcomes.items():
        records[case_id].update(outcome)


def _run_stage(
    name: str,
    cases: Sequence[CaseSpec],
    *,
    manifest: dict[str, Any],
    manifest_path: Path,
    selectors: Sequence[str],
    config: SuiteConfig,
    source_environment: Mapping[str, str],
    case_runner: CaseRunner,
) -> dict[str, dict[str, Any]]:
    stage = _stage_record(manifest, name)
    stage.update(status="running", started_at_utc=stage["started_at_utc"] or utc_now(), finished_at_utc=None)
    manifest["status"] = "running"
    atomic_json(manifest_path, manifest)
    outcomes = run_cases_in_batches(
        cases,
        selectors,
        config=config,
        source_environment=source_environment,
        case_runner=case_runner,
    )
    _apply_outcomes(stage, outcomes)
    stage["status"] = "success" if all(outcome["status"] == "success" for outcome in outcomes.values()) else "fail"
    stage["finished_at_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    return outcomes


def _run_memory_fallbacks(
    *,
    base_outcomes: Mapping[str, Mapping[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
    selectors: Sequence[str],
    config: SuiteConfig,
    source_environment: Mapping[str, str],
    case_runner: CaseRunner,
) -> dict[str, dict[str, Any]]:
    stage = _stage_record(manifest, "memory_snapshot")
    all_outcomes: dict[str, dict[str, Any]] = {}
    round_one: list[CaseSpec] = []
    for mode in MEMORY_MODES:
        for dtype in PRECISIONS:
            original_id = f"xl_ctx2048_{mode}_{dtype}"
            if base_outcomes.get(original_id, {}).get("status") == "oom":
                round_one.append(
                    memory_case(
                        config,
                        model_size="xl",
                        context_length=1_024,
                        mode=mode,
                        dtype=dtype,
                        fallback_for=original_id,
                    )
                )
    if not round_one:
        return all_outcomes

    stage["commands"].extend(case_manifest(spec, config) for spec in round_one)
    stage.update(status="running_fallback", finished_at_utc=None)
    atomic_json(manifest_path, manifest)
    first_outcomes = run_cases_in_batches(
        round_one,
        selectors,
        config=config,
        source_environment=source_environment,
        case_runner=case_runner,
    )
    _apply_outcomes(stage, first_outcomes)
    all_outcomes.update(first_outcomes)

    round_two: list[CaseSpec] = []
    for first in round_one:
        if first_outcomes[first.case_id]["status"] == "oom":
            dtype = str(first.configuration["dtype"])
            mode = str(first.configuration["mode"])
            round_two.append(
                memory_case(
                    config,
                    model_size="large",
                    context_length=2_048,
                    mode=mode,
                    dtype=dtype,
                    fallback_for=first.case_id,
                )
            )
    if round_two:
        stage["commands"].extend(case_manifest(spec, config) for spec in round_two)
        atomic_json(manifest_path, manifest)
        second_outcomes = run_cases_in_batches(
            round_two,
            selectors,
            config=config,
            source_environment=source_environment,
            case_runner=case_runner,
        )
        _apply_outcomes(stage, second_outcomes)
        all_outcomes.update(second_outcomes)
    command_statuses = {record["status"] for record in stage["commands"]}
    stage["status"] = "fail" if command_statuses - {"success", "oom"} else "complete_with_oom" if "oom" in command_statuses else "success"
    stage["finished_at_utc"] = utc_now()
    atomic_json(manifest_path, manifest)
    return all_outcomes


def run_suite(
    config: SuiteConfig,
    selectors: Sequence[str],
    *,
    source_environment: Mapping[str, str] | None = None,
    case_runner: CaseRunner = execute_case_once,
) -> dict[str, Any]:
    source = os.environ if source_environment is None else source_environment
    matrix = build_stage_matrix(config)
    config.runtime_root.mkdir(parents=True, mode=0o700, exist_ok=False)
    manifest_path = config.runtime_root / MANIFEST_NAME
    manifest = build_manifest(config, matrix)
    atomic_json(manifest_path, manifest)
    if config.dry_run:
        manifest.update(status="dry_run", finished_at_utc=utc_now())
        for stage in manifest["stages"]:
            stage["status"] = "dry_run"
        atomic_json(manifest_path, manifest)
        return manifest

    for stage_name in ("benchmark", "torch_profile", "mixed_precision"):
        _run_stage(
            stage_name,
            matrix[stage_name],
            manifest=manifest,
            manifest_path=manifest_path,
            selectors=selectors,
            config=config,
            source_environment=source,
            case_runner=case_runner,
        )
    memory_outcomes = _run_stage(
        "memory_snapshot",
        matrix["memory_snapshot"],
        manifest=manifest,
        manifest_path=manifest_path,
        selectors=selectors,
        config=config,
        source_environment=source,
        case_runner=case_runner,
    )
    _run_memory_fallbacks(
        base_outcomes=memory_outcomes,
        manifest=manifest,
        manifest_path=manifest_path,
        selectors=selectors,
        config=config,
        source_environment=source,
        case_runner=case_runner,
    )
    all_commands = [record for stage in manifest["stages"] for record in stage["commands"]]
    for stage in manifest["stages"]:
        command_statuses = {record["status"] for record in stage["commands"]}
        stage["status"] = "fail" if command_statuses - {"success", "oom"} else "complete_with_oom" if "oom" in command_statuses else "success"
    hard_failures = [record for record in all_commands if record["status"] not in {"success", "oom"}]
    manifest["status"] = "fail" if hard_failures else "complete_with_oom" if any(record["status"] == "oom" for record in all_commands) else "success"
    manifest["finished_at_utc"] = utc_now()
    manifest["summary"] = {
        "command_count": len(all_commands),
        "success_count": sum(record["status"] == "success" for record in all_commands),
        "oom_count": sum(record["status"] == "oom" for record in all_commands),
        "failure_count": sum(record["status"] not in {"success", "oom"} for record in all_commands),
        "fallback_count": sum(record["fallback_for"] is not None for record in all_commands),
    }
    if manifest["status"] in {"success", "complete_with_oom"}:
        completion = {
            "schema_version": SCHEMA_VERSION,
            "status": manifest["status"],
            "manifest": MANIFEST_NAME,
            "finished_at_utc": manifest["finished_at_utc"],
            "expected_base_command_count": manifest["expected_matrix"]["base_command_count"],
            "actual_command_count": len(all_commands),
        }
        atomic_json(config.runtime_root / COMPLETION_MARKER, completion)
        manifest["completion_marker"]["created"] = True
    if manifest["status"] == "success":
        marker = {
            "schema_version": SCHEMA_VERSION,
            "status": "success",
            "manifest": MANIFEST_NAME,
            "finished_at_utc": manifest["finished_at_utc"],
            "expected_base_command_count": manifest["expected_matrix"]["base_command_count"],
            "actual_command_count": len(all_commands),
        }
        atomic_json(config.runtime_root / SUCCESS_MARKER, marker)
        manifest["success_marker"]["created"] = True
    atomic_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, selectors = resolve_config(args)
        manifest = run_suite(config, selectors)
    except Exception as exc:  # noqa: BLE001 - worker-visible error contains no credentials
        print(f"A2-P suite failed before completion: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if config.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": MANIFEST_NAME,
                "summary": manifest.get("summary"),
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] in {"success", "complete_with_oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
