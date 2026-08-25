"""Shared, fail-closed runtime helpers for the A2-K formal GPU experiments.

The public records produced through this module intentionally omit argv, paths,
environment variables, host/user names, CUDA UUIDs, process information, and
raw exception text.  A CPU dry-run validates control flow only; it never falls
back from a requested formal CUDA experiment.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any


SCHEMA_VERSION = "cs336.a2k.runtime.v1"
STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
MIN_FREE_MIB = 22 * 1024
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
MIN_TOTAL_MIB = 24_000
MAX_TOTAL_MIB = 25_000
MIB = 1024**2
GPU_MEMORY_REPORT_REL_TOL = 0.02
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BASE_RUNTIME_ROOT = (_PROJECT_ROOT / ".runtime" / "a2k").resolve()


class RuntimeValidationError(RuntimeError):
    """Raised before a formal run when its evidence contract is not satisfied."""


def _configure_project_runtime() -> None:
    """Keep compiler and temporary state below the assignment worktree."""

    requested = os.environ.get("A2K_RUNTIME_DIR")
    root = Path(requested).expanduser().resolve() if requested else _BASE_RUNTIME_ROOT
    try:
        root.relative_to(_BASE_RUNTIME_ROOT)
    except ValueError as exc:
        raise RuntimeError("A2K_RUNTIME_DIR must stay inside the project A2-K runtime root") from exc
    paths = {
        "CUDA_CACHE_PATH": root / "cuda-cache",
        "HOME": root / "home",
        "PYTHONPYCACHEPREFIX": root / "pycache",
        "TEMP": root / "tmp",
        "TMP": root / "tmp",
        "TMPDIR": root / "tmp",
        "TORCH_EXTENSIONS_DIR": root / "torch-extensions",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "TRITON_CACHE_DIR": root / "triton",
        "XDG_CACHE_HOME": root / "xdg-cache",
    }
    for variable, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)


_configure_project_runtime()

# The cache environment must be fixed before importing Torch or Triton. Merely
# importing torch does not allocate CUDA tensors; the formal guard below runs
# before the scripts import the student attention implementation or make data.
import torch  # noqa: E402


@dataclass(frozen=True)
class RuntimeGuard:
    """Validated execution mode and its sanitized public metadata."""

    device: torch.device
    authoritative: bool
    metadata: dict[str, Any]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _mib(byte_count: int | float) -> float:
    return float(byte_count) / MIB


def _finite_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise RuntimeValidationError(f"invalid {label}")
    return value


def validate_formal_device_facts(
    *,
    device_count: int,
    gpu_name: str,
    total_mib: float,
    free_mib: float,
    prior_allocated_bytes: int,
) -> None:
    """Validate facts without side effects, making the hard gate unit-testable."""

    if device_count != 1:
        raise RuntimeValidationError("formal execution requires exactly one visible CUDA device")
    if " ".join(gpu_name.split()) != EXPECTED_GPU_NAME:
        raise RuntimeValidationError("formal execution requires an NVIDIA GeForce RTX 4090")
    _finite_positive(total_mib, "total memory")
    _finite_positive(free_mib, "free memory")
    if not MIN_TOTAL_MIB <= total_mib <= MAX_TOTAL_MIB:
        raise RuntimeValidationError("visible GPU is not in the expected 24 GiB memory range")
    if free_mib < MIN_FREE_MIB:
        raise RuntimeValidationError("formal execution requires at least 22 GiB free memory")
    if prior_allocated_bytes != 0:
        raise RuntimeValidationError("CUDA memory was allocated before the 23 GiB allocator guard")


def validate_development_device_facts(
    *,
    device_count: int,
    total_mib: float,
    free_mib: float,
    prior_allocated_bytes: int,
) -> None:
    """Validate a larger development GPU without calling it formal evidence."""

    if device_count != 1:
        raise RuntimeValidationError("development CUDA execution requires exactly one visible device")
    _finite_positive(total_mib, "total memory")
    _finite_positive(free_mib, "free memory")
    if total_mib < ALLOCATOR_LIMIT_MIB:
        raise RuntimeValidationError("development GPU cannot enforce the 23 GiB allocator budget")
    if free_mib < MIN_FREE_MIB:
        raise RuntimeValidationError("development CUDA execution requires at least 22 GiB free memory")
    if prior_allocated_bytes != 0:
        raise RuntimeValidationError("CUDA memory was allocated before the 23 GiB allocator guard")


def _query_nvidia_smi(device_identifier: str) -> dict[str, Any]:
    """Return public fields for the physical GPU selected as logical cuda:0.

    The required public query intentionally contains no UUID.  If nvidia-smi
    exposes several host GPUs despite ``CUDA_VISIBLE_DEVICES``, a second,
    private UUID-only query maps PyTorch's device to the corresponding row.
    Neither the selector nor UUID output is returned or persisted.
    """

    if not device_identifier or any(character.isspace() for character in device_identifier):
        raise RuntimeValidationError("CUDA device selector is unavailable")

    public_command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            public_command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeValidationError("required public nvidia-smi metadata is unavailable") from exc
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeValidationError("nvidia-smi did not return any device metadata")
    selected_index = 0
    if len(lines) > 1:
        try:
            identities = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeValidationError("required private GPU identity mapping is unavailable") from exc
        uuids = [line.strip() for line in identities.stdout.splitlines() if line.strip()]
        if len(uuids) != len(lines):
            raise RuntimeValidationError("nvidia-smi device identity rows are inconsistent")

        def normalized(value: str) -> str:
            return value.strip().lower().removeprefix("gpu-")

        wanted = normalized(device_identifier)
        matches = [index for index, value in enumerate(uuids) if normalized(value) == wanted]
        if len(matches) != 1:
            raise RuntimeValidationError("the PyTorch device could not be mapped to one nvidia-smi row")
        selected_index = matches[0]

    fields = [field.strip() for field in lines[selected_index].split(",")]
    if len(fields) != 6:
        raise RuntimeValidationError("nvidia-smi returned an unexpected public metadata schema")
    name, total, free, driver, power, pstate = fields
    try:
        total_mib = float(total)
        free_mib = float(free)
        power_limit_w = float(power)
    except ValueError as exc:
        raise RuntimeValidationError("nvidia-smi returned non-numeric memory or power data") from exc
    return {
        "name": name,
        "memory_total_mib": total_mib,
        "memory_free_mib": free_mib,
        "driver_version": driver,
        "power_limit_w": power_limit_w,
        "pstate": pstate,
    }


def _software_metadata() -> dict[str, Any]:
    try:
        import triton

        triton_version: str | None = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton_version,
    }


def set_tf32(enabled: bool) -> None:
    """Set the public TF32 policy with the current and legacy Torch APIs."""

    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        matmul.fp32_precision = "tf32" if enabled else "ieee"
    else:  # pragma: no cover - compatibility with older pinned Torch builds.
        matmul.allow_tf32 = enabled
    if hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
        torch.backends.cudnn.conv.fp32_precision = "tf32" if enabled else "ieee"
    elif hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = enabled


def tf32_policy() -> str:
    matmul = torch.backends.cuda.matmul
    if hasattr(matmul, "fp32_precision"):
        return str(matmul.fp32_precision)
    return "tf32" if bool(matmul.allow_tf32) else "ieee"


def prepare_runtime(
    *,
    dry_run: bool,
    tf32_enabled: bool = False,
    development_cuda: bool = False,
) -> RuntimeGuard:
    """Prepare CPU/dev CUDA, or enforce the authoritative formal 4090 gate.

    For a formal run this function must be called before creating any CUDA
    tensor, model, or optimizer. It never selects CPU as a fallback.
    """

    if dry_run and development_cuda:
        raise RuntimeValidationError("CPU dry-run and development CUDA modes are mutually exclusive")
    set_tf32(tf32_enabled)
    if dry_run:
        return RuntimeGuard(
            device=torch.device("cpu"),
            authoritative=False,
            metadata={
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": utc_now(),
                "starter_commit": STARTER_COMMIT,
                "authoritative": False,
                "device_type": "cpu",
                "non_authoritative_reason": "CPU dry-run; not valid GPU correctness, performance, or allocator evidence",
                "software": _software_metadata(),
                "tf32_policy": tf32_policy(),
                "allocator": {
                    "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
                    "allocator_fraction": None,
                    "guard_applied": False,
                },
            },
        )

    if not torch.cuda.is_available():
        raise RuntimeValidationError("formal CUDA execution was requested but CUDA is unavailable")
    device_count = torch.cuda.device_count()
    if device_count != 1:
        mode = "development CUDA" if development_cuda else "formal"
        raise RuntimeValidationError(f"{mode} execution requires exactly one visible CUDA device")

    device = torch.device("cuda", 0)
    prior_allocated = int(torch.cuda.memory_allocated(device))
    properties = torch.cuda.get_device_properties(device)
    total_mib = _mib(properties.total_memory)

    # Apply the allocator budget before mem_get_info and, critically, before
    # the first PyTorch CUDA tensor/model/optimizer allocation.
    fraction = min(1.0, (ALLOCATOR_LIMIT_MIB * MIB) / properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    actual_fraction = float(torch.cuda.get_per_process_memory_fraction(device))
    free_bytes, runtime_total_bytes = torch.cuda.mem_get_info(device)
    free_mib = _mib(free_bytes)
    runtime_total_mib = _mib(runtime_total_bytes)

    if development_cuda:
        validate_development_device_facts(
            device_count=device_count,
            total_mib=total_mib,
            free_mib=free_mib,
            prior_allocated_bytes=prior_allocated,
        )
    else:
        validate_formal_device_facts(
            device_count=device_count,
            gpu_name=properties.name,
            total_mib=total_mib,
            free_mib=free_mib,
            prior_allocated_bytes=prior_allocated,
        )
    if abs(runtime_total_mib - total_mib) > 32:
        raise RuntimeValidationError("CUDA APIs disagree on total device memory")
    if not math.isclose(actual_fraction, fraction, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeValidationError("the 23 GiB allocator fraction was not applied")

    device_identifier = str(getattr(properties, "uuid", ""))
    smi = _query_nvidia_smi(device_identifier)
    if development_cuda:
        validate_development_device_facts(
            device_count=device_count,
            total_mib=float(smi["memory_total_mib"]),
            free_mib=float(smi["memory_free_mib"]),
            prior_allocated_bytes=prior_allocated,
        )
    else:
        validate_formal_device_facts(
            device_count=device_count,
            gpu_name=str(smi["name"]),
            total_mib=float(smi["memory_total_mib"]),
            free_mib=float(smi["memory_free_mib"]),
            prior_allocated_bytes=prior_allocated,
        )
    # On some 48 GiB 4090 environments, NVML reports the physical framebuffer
    # while CUDA exposes slightly less memory to the process (observed delta:
    # 621 MiB, 1.3%).  Reject a genuinely different device, but tolerate this
    # documented driver/runtime reservation.
    if not math.isclose(
        float(smi["memory_total_mib"]),
        total_mib,
        rel_tol=GPU_MEMORY_REPORT_REL_TOL,
        abs_tol=64,
    ):
        raise RuntimeValidationError("CUDA and nvidia-smi disagree on total device memory")

    return RuntimeGuard(
        device=device,
        authoritative=not development_cuda,
        metadata={
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "starter_commit": STARTER_COMMIT,
            "authoritative": not development_cuda,
            "device_type": "cuda",
            "execution_mode": "development_cuda" if development_cuda else "formal_cuda",
            "non_authoritative_reason": ("development CUDA run on non-standard hardware; not valid RTX 4090 24GB evidence" if development_cuda else None),
            "gpu": smi,
            "software": _software_metadata(),
            "tf32_policy": tf32_policy(),
            "allocator": {
                "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
                "allocator_fraction": actual_fraction,
                "guard_applied": True,
                "applied_before_first_cuda_allocation": True,
                "prior_allocated_bytes": prior_allocated,
            },
        },
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    synchronize(device)
    return {
        "peak_allocated_mib": _mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": _mib(torch.cuda.max_memory_reserved(device)),
    }


def release_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def public_error(exc: BaseException) -> dict[str, str]:
    """Describe failures without copying arbitrary path-bearing messages."""

    return {
        "category": "out_of_memory" if is_oom_error(exc) else "runtime_error",
        "type": type(exc).__name__,
        "message": "allocator OOM; case retained without fallback" if is_oom_error(exc) else "case failed; inspect private logs",
    }


_PRIVATE_ENV_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|COOKIE|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|AUTH|USERNAME|USER_ID|APP_ID|CLIENT_ID)",
    re.IGNORECASE,
)
_CASE_NAMESPACE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


def child_process_environment(case_namespace: str) -> dict[str, str]:
    """Build a credential-stripped child env with a private per-case cache.

    The returned values are for ``subprocess`` only and must never be copied to
    public metadata. A fresh interpreter sees ``A2K_RUNTIME_DIR`` before Torch
    import, preventing compile cold-start cases from sharing disk caches.
    """

    if _CASE_NAMESPACE.fullmatch(case_namespace) is None:
        raise ValueError("case namespace must use only public alphanumeric, dot, dash, or underscore characters")
    root = (_BASE_RUNTIME_ROOT / "cases" / case_namespace).resolve()
    root.relative_to(_BASE_RUNTIME_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    environment = {name: value for name, value in os.environ.items() if not _PRIVATE_ENV_NAME.search(name)}
    environment["A2K_RUNTIME_DIR"] = str(root)
    return environment


def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    materialized = [dict(row) for row in rows]
    if fieldnames is None:
        fieldnames = sorted({key for row in materialized for key in row})
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), extrasaction="raise")
    writer.writeheader()
    for row in materialized:
        writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    _atomic_write_text(path, buffer.getvalue())


_SENSITIVE_TEXT = re.compile(
    r"(?<![A-Za-z0-9])/(?:[^/\s\"]+/)+[^/\s\"]+|\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[0-9a-f]{8}-[0-9a-f-]{27,}\b",
    re.IGNORECASE,
)


def assert_public_payload(payload: Any) -> None:
    """Reject common path/IP/UUID leaks before making a result public."""

    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    if _SENSITIVE_TEXT.search(encoded):
        raise RuntimeValidationError("public payload contains a forbidden path, IP address, or UUID")


__all__ = [
    "ALLOCATOR_LIMIT_MIB",
    "HARD_LIMIT_MIB",
    "MIN_FREE_MIB",
    "RuntimeGuard",
    "RuntimeValidationError",
    "STARTER_COMMIT",
    "assert_public_payload",
    "atomic_write_csv",
    "atomic_write_json",
    "child_process_environment",
    "is_oom_error",
    "peak_memory_mib",
    "prepare_runtime",
    "public_error",
    "release_memory",
    "reset_peak_memory",
    "set_tf32",
    "synchronize",
    "tf32_policy",
    "utc_now",
    "validate_development_device_facts",
    "validate_formal_device_facts",
]
