"""Shared, reproducible measurement utilities for the A2-K experiment scripts.

The helpers in this module deliberately keep benchmark plumbing separate from
the attention implementations.  They set the required allocator guard before
any CUDA tensor is created, use CUDA synchronization around measured work, and
only write values that were observed in the current process.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, cast

import torch
from torch import Tensor


# The formal CUDA runtime uses Python 3.10, where datetime.UTC is unavailable.
UTC_TZ = timezone.utc  # noqa: UP017

MIB: Final[int] = 1024**2
GIB: Final[int] = 1024**3
ALLOCATOR_LIMIT_MIB: Final[int] = 23 * 1024
ALLOCATOR_LIMIT_BYTES: Final[int] = 23 * GIB
HARD_MEMORY_LIMIT_MIB: Final[int] = 24 * 1024
MINIMUM_FREE_MEMORY_BYTES: Final[int] = 22 * GIB
METADATA_SCHEMA_VERSION: Final[int] = 1

CsvValue = str | int | float | bool | None
CsvRow = dict[str, CsvValue]
JsonObject = dict[str, object]
Phase = Literal["forward", "backward", "forward_backward"]


class A2KScriptError(RuntimeError):
    """Base error for a benchmark that cannot produce a valid measurement."""


class CudaPreflightError(A2KScriptError):
    """Raised when CUDA or the requested formal environment is unavailable."""

    def __init__(self, reason: str, *, status: str = "unavailable") -> None:
        super().__init__(reason)
        self.status = status
        self.public_reason = reason


class MeasurementError(A2KScriptError):
    """Raised when a timing backend returns an invalid result."""


@dataclass(frozen=True)
class CudaRuntime:
    """Sanitized CUDA state collected after installing the allocator guard."""

    device: torch.device
    device_index: int
    gpu_name: str
    total_memory_mib: float
    free_memory_mib: float | None
    driver_version: str | None
    power_limit_watts: float | None
    pstate: str | None
    allocator_fraction: float
    visible_device_count: int


@dataclass(frozen=True)
class LatencySummary:
    """Latency quantiles measured over a real CUDA timing interval."""

    p20_ms: float
    p50_ms: float
    p80_ms: float
    sample_count: int | None
    measured_duration_ms: float
    timer: str


@dataclass(frozen=True)
class PhaseMeasurement:
    """Latency and allocator peaks for one benchmark configuration."""

    status: str
    error_kind: str | None
    latency: LatencySummary | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None


@dataclass(frozen=True)
class FlashImplementation:
    """A callable FlashAttention autograd implementation exposed by the adapter."""

    name: str
    apply: Callable[[Tensor, Tensor, Tensor, bool], Tensor]
    kernel_config: dict[str, int | None]


def default_output_dir() -> Path:
    """Return the local-only default result directory for A2-K experiments."""

    return Path("local_results") / "a2k"


def repository_root() -> Path:
    """Resolve the repository root without ever writing its path to metadata."""

    return Path(__file__).resolve().parents[2]


def utc_timestamp() -> str:
    """Return a timezone-explicit, non-identifying timestamp."""

    return datetime.now(UTC_TZ).replace(microsecond=0).isoformat()


def _numeric_prefix(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match is not None else None


def _cuda_visible_device_selector(device_index: int) -> str | None:
    """Return the physical GPU selector exported by a scheduler, if present."""

    raw_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw_value is None:
        return None
    selectors = [item.strip() for item in raw_value.split(",")]
    if device_index >= len(selectors):
        return None
    selector = selectors[device_index]
    return selector if selector and selector != "-1" else None


def _query_nvidia_smi(device_index: int, *, expected_uuid: str | None) -> dict[str, str | float] | None:
    """Return public metadata for the CUDA-visible GPU without persisting UUIDs."""

    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,memory.free,driver_version,power.limit,pstate",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    parsed_rows = [next(csv.reader([row]), []) for row in rows]
    fields: list[str] | None = None
    if expected_uuid is not None:
        fields = next((row for row in parsed_rows if len(row) == 7 and row[0].strip() == expected_uuid), None)
    else:
        selector = _cuda_visible_device_selector(device_index)
        if selector is None:
            physical_index = device_index
            if physical_index < len(parsed_rows):
                fields = parsed_rows[physical_index]
        elif selector.startswith("GPU-"):
            fields = next((row for row in parsed_rows if len(row) == 7 and row[0].strip() == selector), None)
        elif selector.isdecimal():
            physical_index = int(selector)
            if physical_index < len(parsed_rows):
                fields = parsed_rows[physical_index]
    if fields is None or len(fields) != 7:
        return None

    total_memory = _numeric_prefix(fields[2])
    free_memory = _numeric_prefix(fields[3])
    power_limit = _numeric_prefix(fields[5])
    if total_memory is None or free_memory is None:
        return None
    result: dict[str, str | float] = {
        "name": fields[1].strip(),
        "total_memory_mib": total_memory,
        "free_memory_mib": free_memory,
        "driver_version": fields[4].strip() or "unavailable",
        "pstate": fields[6].strip() or "unavailable",
    }
    if power_limit is not None:
        result["power_limit_watts"] = power_limit
    return result


def _nvidia_smi_selector(device_index: int, *, expected_uuid: str | None) -> str:
    """Return a selector that addresses the CUDA-visible physical GPU."""

    if expected_uuid is not None:
        return expected_uuid
    visible_selector = _cuda_visible_device_selector(device_index)
    return visible_selector if visible_selector is not None else str(device_index)


def _query_compute_process_ids(device_index: int, *, expected_uuid: str | None) -> set[int] | None:
    """Return compute PIDs for one GPU without persisting process information.

    ``None`` means that the query could not be trusted. Formal measurements
    reject that state rather than inferring process isolation from free memory.
    """

    command = [
        "nvidia-smi",
        "-i",
        _nvidia_smi_selector(device_index, expected_uuid=expected_uuid),
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None

    process_ids: set[int] = set()
    for line in completed.stdout.splitlines():
        value = line.strip()
        if not value or value.lower() == "no running processes found":
            continue
        fields = next(csv.reader([value]), [])
        if len(fields) != 1:
            return None
        process_id = fields[0].strip()
        if not process_id.isdecimal():
            return None
        process_ids.add(int(process_id))
    return process_ids


def configure_cuda(device_name: str, *, formal: bool) -> CudaRuntime:
    """Install the 23 GiB allocator guard and validate the requested CUDA device.

    This function must be called before creating tensors, models, optimizers, or
    triggering compilation.  CUDA property queries may initialize a context but
    do not allocate through the PyTorch caching allocator; the guard is set
    immediately after the total-memory query and before any tensor allocation.
    """

    if not torch.cuda.is_available():
        raise CudaPreflightError("CUDA is not available; no GPU measurement was performed.")

    device = torch.device(device_name)
    if device.type != "cuda":
        raise CudaPreflightError(f"Requested device {device_name!r} is not a CUDA device.")
    device_index = 0 if device.index is None else device.index
    visible_device_count = torch.cuda.device_count()
    if device_index < 0 or device_index >= visible_device_count:
        raise CudaPreflightError(f"Requested CUDA device index {device_index} is not visible.")
    if formal and visible_device_count != 1:
        raise CudaPreflightError(
            "Formal mode requires exactly one visible CUDA device; refusing a multi-device measurement.",
            status="nonstandard_hardware",
        )

    torch.cuda.set_device(device_index)
    properties = torch.cuda.get_device_properties(device_index)
    allocator_fraction = min(1.0, ALLOCATOR_LIMIT_BYTES / properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=device_index)

    raw_uuid = getattr(properties, "uuid", None)
    expected_uuid = raw_uuid if isinstance(raw_uuid, str) and raw_uuid.startswith("GPU-") else None
    smi = _query_nvidia_smi(device_index, expected_uuid=expected_uuid)
    if formal and smi is None:
        raise CudaPreflightError(
            "Formal mode requires nvidia-smi metadata for Driver, power limit, and P-state.",
            status="metadata_unavailable",
        )
    free_memory_mib: float | None = None
    if smi is not None:
        gpu_name = str(smi["name"])
        total_memory_mib = float(smi["total_memory_mib"])
        free_memory_mib = float(smi["free_memory_mib"])
        driver_version = str(smi["driver_version"])
        power_limit_watts = cast(float | None, smi.get("power_limit_watts"))
        pstate = str(smi["pstate"])
    else:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        gpu_name = properties.name
        total_memory_mib = total_bytes / MIB
        free_memory_mib = free_bytes / MIB
        driver_version = None
        power_limit_watts = None
        pstate = None

    runtime = CudaRuntime(
        device=torch.device("cuda", device_index),
        device_index=device_index,
        gpu_name=gpu_name,
        total_memory_mib=total_memory_mib,
        free_memory_mib=free_memory_mib,
        driver_version=driver_version,
        power_limit_watts=power_limit_watts,
        pstate=pstate,
        allocator_fraction=allocator_fraction,
        visible_device_count=visible_device_count,
    )
    if formal and "RTX 4090" not in runtime.gpu_name.upper():
        raise CudaPreflightError(
            f"Formal mode requires an NVIDIA GeForce RTX 4090, found {runtime.gpu_name!r}.",
            status="nonstandard_hardware",
        )
    if formal and runtime.free_memory_mib is not None and runtime.free_memory_mib * MIB < MINIMUM_FREE_MEMORY_BYTES:
        raise CudaPreflightError(
            "Formal mode requires at least 22 GiB free before the matrix starts; no shape was reduced.",
            status="insufficient_free_memory",
        )
    if formal:
        compute_process_ids = _query_compute_process_ids(device_index, expected_uuid=expected_uuid)
        if compute_process_ids is None:
            raise CudaPreflightError(
                "Formal mode requires a trustworthy nvidia-smi compute-process query for the allocated GPU.",
                status="metadata_unavailable",
            )
        if compute_process_ids - {os.getpid()}:
            raise CudaPreflightError(
                "Formal mode requires no other GPU compute process; waiting for the allocated GPU to become idle.",
                status="gpu_busy",
            )
    return runtime


def set_tf32(enabled: bool) -> None:
    """Set both relevant PyTorch TF32 controls after CUDA preflight."""

    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def _safe_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if re.fullmatch(r"[0-9a-f]{7,64}", commit) else None


def _triton_version() -> str | None:
    try:
        import triton
    except ModuleNotFoundError:
        return None
    version = getattr(triton, "__version__", None)
    return str(version) if version is not None else "unknown"


def _runtime_payload(runtime: CudaRuntime | None) -> dict[str, object]:
    if runtime is None:
        return {"cuda_available": False}
    return {
        "cuda_available": True,
        "device": f"cuda:{runtime.device_index}",
        "gpu_name": runtime.gpu_name,
        "total_memory_mib": round(runtime.total_memory_mib, 3),
        "free_memory_mib": round(runtime.free_memory_mib, 3) if runtime.free_memory_mib is not None else None,
        "driver_version": runtime.driver_version,
        "power_limit_watts": runtime.power_limit_watts,
        "pstate": runtime.pstate,
        "visible_device_count": runtime.visible_device_count,
    }


def _software_payload() -> dict[str, object]:
    return {
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": _triton_version(),
    }


def _read_json(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise A2KScriptError(f"Cannot safely merge existing JSON result file {path.name!r}.") from error
    if not isinstance(value, dict):
        raise A2KScriptError(f"Existing JSON result file {path.name!r} must contain an object.")
    return cast(JsonObject, value)


def _object_records(payload: Mapping[str, object], *, field: str, artifact_name: str) -> list[JsonObject]:
    """Read a JSON list while keeping its untyped boundary explicit."""

    raw_records = payload.get(field)
    if raw_records is None:
        return []
    if not isinstance(raw_records, list):
        raise A2KScriptError(f"{artifact_name} has an invalid {field!r} field.")
    records: list[JsonObject] = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            raise A2KScriptError(f"{artifact_name} contains a non-object {field!r} entry.")
        records.append(cast(JsonObject, raw_record))
    return records


def _numeric_record_values(records: Sequence[Mapping[str, object]], field: str) -> list[float]:
    """Extract a numeric JSON field without trusting unvalidated JSON values."""

    values: list[float] = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically write a compact, public JSON result artifact."""

    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, CsvValue]]) -> None:
    """Atomically write rows with a stable schema and empty cells for nulls."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as temporary:
            writer = csv.DictWriter(
                temporary,
                fieldnames=list(fieldnames),
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fieldnames})
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _reproduction_command(script_name: str, *, formal: bool) -> str:
    """Return a sanitized command that reproduces the recorded execution mode."""

    script_stem = Path(script_name).stem
    if formal:
        mode_flag = "--formal"
    elif Path(script_name).name == "benchmark_checkpointing.py":
        mode_flag = "--nonformal"
    else:
        mode_flag = "--non-formal"
    return f"python -m student_scripts.a2k.{script_stem} {mode_flag}"


def append_run_metadata(
    output_dir: Path,
    *,
    script_name: str,
    runtime: CudaRuntime | None,
    status: str,
    formal: bool,
    configuration: Mapping[str, object],
    reason: str | None = None,
) -> None:
    """Append one sanitized process record to ``run_metadata.json``."""

    path = output_dir / "run_metadata.json"
    payload = _read_json(path)
    existing_runs = _object_records(payload, field="runs", artifact_name="run_metadata.json")

    allocator: dict[str, object]
    if runtime is None:
        allocator = {"allocator_fraction": None, "allocator_limit_mib": None, "target_limit_mib": ALLOCATOR_LIMIT_MIB}
    else:
        allocator = {
            "allocator_fraction": runtime.allocator_fraction,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        }

    record: dict[str, object] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "timestamp_utc": utc_timestamp(),
        "script": script_name,
        "status": status,
        "formal": formal,
        "command": _reproduction_command(script_name, formal=formal),
        "commit": _safe_commit(),
        "cuda": _runtime_payload(runtime),
        "software": _software_payload(),
        "allocator": allocator,
        "tf32": {
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if runtime is not None else None,
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if runtime is not None else None,
        },
        "configuration": dict(configuration),
    }
    if reason is not None:
        record["reason"] = reason
    existing_runs.append(record)
    write_json(path, {"schema_version": METADATA_SCHEMA_VERSION, "runs": existing_runs})


def append_memory_observation(
    output_dir: Path,
    *,
    script_name: str,
    runtime: CudaRuntime | None,
    status: str,
    peak_allocated_mib: float | None,
    peak_reserved_mib: float | None,
    formal: bool,
) -> None:
    """Append observed allocator peaks and recompute the cross-process maximum."""

    path = output_dir / "memory_evidence.json"
    payload = _read_json(path)
    existing_runs = _object_records(payload, field="runs", artifact_name="memory_evidence.json")

    observation: dict[str, object] = {
        "timestamp_utc": utc_timestamp(),
        "script": script_name,
        "status": status,
        "formal": formal,
        "peak_allocated_mib": round(peak_allocated_mib, 3) if peak_allocated_mib is not None else None,
        "peak_reserved_mib": round(peak_reserved_mib, 3) if peak_reserved_mib is not None else None,
        "allocator_fraction": runtime.allocator_fraction if runtime is not None else None,
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB if runtime is not None else None,
    }
    existing_runs.append(observation)

    # Top-level evidence must describe only formal RTX 4090 measurements.
    # Keep development runs in the audit trail, but never let them determine a
    # claimed 24 GiB peak or allocator verdict.
    formal_runs = [record for record in existing_runs if record.get("formal") is True]
    allocated_values = _numeric_record_values(formal_runs, "peak_allocated_mib")
    reserved_values = _numeric_record_values(formal_runs, "peak_reserved_mib")
    max_allocated = max(allocated_values) if allocated_values else None
    max_reserved = max(reserved_values) if reserved_values else None
    allocator_fraction = runtime.allocator_fraction if formal and runtime is not None else None
    if allocator_fraction is None:
        for record in reversed(formal_runs):
            recorded_fraction = record.get("allocator_fraction")
            if isinstance(recorded_fraction, (float, int)) and not isinstance(recorded_fraction, bool):
                allocator_fraction = float(recorded_fraction)
                break

    result: dict[str, object] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "allocator": {
            "allocator_fraction": allocator_fraction,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB if allocator_fraction is not None else None,
            "target_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": HARD_MEMORY_LIMIT_MIB,
        "pytorch_peak_allocated_mib": round(max_allocated, 3) if max_allocated is not None else None,
        "pytorch_peak_reserved_mib": round(max_reserved, 3) if max_reserved is not None else None,
        "within_allocator_guard": None if max_reserved is None else max_reserved <= ALLOCATOR_LIMIT_MIB,
        "within_hard_limit": None if max_reserved is None else max_reserved <= HARD_MEMORY_LIMIT_MIB,
        # The formal A2-K criterion is stricter than physical 24 GiB: reserved
        # memory must stay within the configured 23 GiB allocator guard.
        "within_24gib": None if max_reserved is None else max_reserved <= ALLOCATOR_LIMIT_MIB,
        "formal_observation_count": len(formal_runs),
        "excluded_nonformal_observation_count": len(existing_runs) - len(formal_runs),
        "runs": existing_runs,
    }
    write_json(path, result)


def record_preflight_failure(
    output_dir: Path,
    *,
    script_name: str,
    formal: bool,
    configuration: Mapping[str, object],
    error: CudaPreflightError,
) -> None:
    """Persist a clear unavailable record instead of inventing GPU measurements."""

    append_run_metadata(
        output_dir,
        script_name=script_name,
        runtime=None,
        status=error.status,
        formal=formal,
        configuration=configuration,
        reason=error.public_reason,
    )
    append_memory_observation(
        output_dir,
        script_name=script_name,
        runtime=None,
        status=error.status,
        peak_allocated_mib=None,
        peak_reserved_mib=None,
        formal=formal,
    )


def parse_positive_ints(value: str, *, option: str) -> tuple[int, ...]:
    """Parse a comma-separated list without silently accepting invalid shapes."""

    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"{option} must be a comma-separated list of positive integers.") from error
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError(f"{option} must contain at least one positive integer.")
    return parsed


def parse_attention_shapes(value: str) -> tuple[tuple[int, int], ...]:
    """Parse ``sequence:head_dim`` pairs for compile microbenchmarks."""

    shapes: list[tuple[int, int]] = []
    for part in value.split(","):
        sequence_text, separator, head_dim_text = part.strip().partition(":")
        if not separator:
            raise ValueError("--attention-shapes must use comma-separated sequence:head_dim pairs.")
        try:
            sequence_length = int(sequence_text)
            head_dim = int(head_dim_text)
        except ValueError as error:
            raise ValueError("--attention-shapes must contain positive integers.") from error
        if sequence_length <= 0 or head_dim <= 0:
            raise ValueError("--attention-shapes must contain positive integers.")
        shapes.append((sequence_length, head_dim))
    if not shapes:
        raise ValueError("--attention-shapes must not be empty.")
    return tuple(shapes)


def dtype_from_name(name: str) -> torch.dtype:
    """Map intentionally small CLI spelling to a concrete torch dtype."""

    normalized = name.lower()
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype {name!r}; use fp32, bf16, or fp16.")


def dtype_name(dtype: torch.dtype) -> str:
    """Return a stable, concise dtype label for public result files."""

    names = {
        torch.float32: "float32",
        torch.bfloat16: "bfloat16",
        torch.float16: "float16",
    }
    return names.get(dtype, str(dtype).replace("torch.", ""))


def make_attention_inputs(
    *,
    batch_size: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Allocate deterministic Q/K/V and dO before entering a timing interval."""

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    shape = (batch_size, sequence_length, head_dim)
    q = torch.randn(shape, dtype=dtype, device=device, generator=generator).requires_grad_(True)
    k = torch.randn(shape, dtype=dtype, device=device, generator=generator).requires_grad_(True)
    v = torch.randn(shape, dtype=dtype, device=device, generator=generator).requires_grad_(True)
    grad_output = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    return q, k, v, grad_output


def explicit_attention_apply(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> Tensor:
    """Call the student-owned explicit baseline without any fused-attention API."""

    from cs336_systems.a2k.attention import explicit_attention

    return explicit_attention(q, k, v, is_causal=is_causal)


def explicit_attention_with_lse(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> tuple[Tensor, Tensor]:
    """Call the explicit reference and retain LSE for correctness comparisons."""

    from cs336_systems.a2k.attention import explicit_attention_with_lse as implementation

    return implementation(q, k, v, is_causal=is_causal)


def _kernel_config(implementation_type: type[torch.autograd.Function]) -> dict[str, int | None]:
    raw_config = getattr(implementation_type, "BENCHMARK_CONFIG", {})
    if not isinstance(raw_config, Mapping):
        raw_config = {}

    def integer_value(*keys: str) -> int | None:
        for key in keys:
            value = raw_config.get(key)
            if isinstance(value, int):
                return value
        return None

    return {
        "query_tile": integer_value("query_tile", "block_m", "BLOCK_M"),
        "key_tile": integer_value("key_tile", "block_n", "BLOCK_N"),
        "num_warps": integer_value("num_warps"),
        "num_stages": integer_value("num_stages"),
    }


def load_flash_implementations(*, include_pytorch: bool = True, include_triton: bool = True) -> list[FlashImplementation]:
    """Resolve the same autograd classes that official tests use via the adapter."""

    from tests.adapters import get_flashattention_autograd_function_pytorch, get_flashattention_autograd_function_triton

    implementations: list[FlashImplementation] = []
    if include_pytorch:
        pytorch_type = get_flashattention_autograd_function_pytorch()
        implementations.append(
            FlashImplementation(
                name="pytorch_tiled",
                apply=cast(Callable[[Tensor, Tensor, Tensor, bool], Tensor], pytorch_type.apply),
                kernel_config={"query_tile": None, "key_tile": None, "num_warps": None, "num_stages": None},
            )
        )
    if include_triton:
        triton_type = get_flashattention_autograd_function_triton()
        implementations.append(
            FlashImplementation(
                name="triton_flashattention2",
                apply=cast(Callable[[Tensor, Tensor, Tensor, bool], Tensor], triton_type.apply),
                kernel_config=_kernel_config(triton_type),
            )
        )
    return implementations


def make_attention_phase(
    apply: Callable[[Tensor, Tensor, Tensor, bool], Tensor],
    *,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    grad_output: Tensor,
    is_causal: bool,
    phase: Phase,
) -> Callable[[], object]:
    """Build a phase callable while keeping Q/K/V and random dO outside timing."""

    def forward() -> Tensor:
        return apply(q, k, v, is_causal)

    if phase == "forward":
        return forward
    if phase == "backward":
        output = forward()

        def backward() -> tuple[Tensor | None, ...]:
            return torch.autograd.grad(
                output,
                (q, k, v),
                grad_outputs=grad_output,
                retain_graph=True,
                allow_unused=False,
            )

        return backward
    if phase == "forward_backward":

        def forward_backward() -> tuple[Tensor | None, ...]:
            output = forward()
            return torch.autograd.grad(output, (q, k, v), grad_outputs=grad_output, allow_unused=False)

        return forward_backward
    raise ValueError(f"Unsupported phase {phase!r}.")


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise MeasurementError("Cannot calculate a percentile of zero timing samples.")
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def warm_up_cuda(workload: Callable[[], object], *, warmup_ms: int) -> None:
    """Warm a callable for at least ``warmup_ms`` of CUDA event time."""

    if warmup_ms <= 0:
        raise ValueError("warmup_ms must be positive.")
    elapsed_ms = 0.0
    torch.cuda.synchronize()
    while elapsed_ms < warmup_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):
            result = workload()
            del result
        end.record()
        end.synchronize()
        elapsed_ms += start.elapsed_time(end)


def _benchmark_with_events(workload: Callable[[], object], *, rep_ms: int) -> LatencySummary:
    if rep_ms <= 0:
        raise ValueError("rep_ms must be positive.")
    samples: list[float] = []
    elapsed_ms = 0.0
    torch.cuda.synchronize()
    while elapsed_ms < rep_ms:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = workload()
        del result
        end.record()
        end.synchronize()
        sample_ms = start.elapsed_time(end)
        samples.append(sample_ms)
        elapsed_ms += sample_ms
        if len(samples) > 1_000_000:
            raise MeasurementError("CUDA event timer produced too many samples before reaching the target duration.")
    return LatencySummary(
        p20_ms=_percentile(samples, 0.2),
        p50_ms=_percentile(samples, 0.5),
        p80_ms=_percentile(samples, 0.8),
        sample_count=len(samples),
        measured_duration_ms=elapsed_ms,
        timer="cuda_events",
    )


def benchmark_cuda(workload: Callable[[], object], *, warmup_ms: int, rep_ms: int) -> LatencySummary:
    """Use Triton's required `do_bench` protocol, with a CUDA-event fallback.

    The fallback still executes the same 100 ms warmup and 300 ms (by default)
    measurement interval with CUDA events, so it is appropriate when the Triton
    Python helper is unavailable but the measured implementation is otherwise
    valid CUDA code.
    """

    try:
        from triton.testing import do_bench
    except ModuleNotFoundError:
        return _benchmark_with_events(workload, rep_ms=rep_ms)

    torch.cuda.synchronize()
    timings = do_bench(workload, warmup=warmup_ms, rep=rep_ms, return_mode="all")
    torch.cuda.synchronize()
    if not isinstance(timings, (list, tuple)) or not timings:
        raise MeasurementError("triton.testing.do_bench returned no timing samples.")
    try:
        samples = [float(value) for value in timings]
    except (TypeError, ValueError) as error:
        raise MeasurementError("triton.testing.do_bench returned an invalid timing sample.") from error
    if any(not math.isfinite(sample) or sample < 0.0 for sample in samples):
        raise MeasurementError("triton.testing.do_bench returned a non-finite or negative timing sample.")
    return LatencySummary(
        p20_ms=_percentile(samples, 0.2),
        p50_ms=_percentile(samples, 0.5),
        p80_ms=_percentile(samples, 0.8),
        sample_count=len(samples),
        measured_duration_ms=sum(samples),
        timer="triton.testing.do_bench",
    )


def current_peak_memory_mib() -> tuple[float | None, float | None]:
    """Read PyTorch allocator peaks without inferring missing values."""

    try:
        allocated = torch.cuda.max_memory_allocated() / MIB
        reserved = torch.cuda.max_memory_reserved() / MIB
    except RuntimeError:
        return None, None
    return allocated, reserved


def measure_cuda_workload(
    workload: Callable[[], object],
    *,
    warmup_ms: int,
    rep_ms: int,
) -> PhaseMeasurement:
    """Warm, capture allocator peak, then time one CUDA workload honestly."""

    try:
        warm_up_cuda(workload, warmup_ms=warmup_ms)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result = workload()
        del result
        torch.cuda.synchronize()
        latency = benchmark_cuda(workload, warmup_ms=warmup_ms, rep_ms=rep_ms)
        # Lazy kernel workspaces and allocator growth can occur on a later
        # steady-state invocation; sample after the timed interval so the
        # reported peak cannot be lower than the actual workload peak.
        peak_allocated, peak_reserved = current_peak_memory_mib()
        status = "success"
        if peak_reserved is not None and peak_reserved > ALLOCATOR_LIMIT_MIB:
            status = "allocator_limit_exceeded"
        return PhaseMeasurement(
            status=status,
            error_kind=None,
            latency=latency,
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
        )
    except Exception as error:
        peak_allocated, peak_reserved = current_peak_memory_mib()
        if is_out_of_memory(error):
            return PhaseMeasurement(
                status="oom",
                error_kind="oom",
                latency=None,
                peak_allocated_mib=peak_allocated,
                peak_reserved_mib=peak_reserved,
            )
        return PhaseMeasurement(
            status="failed",
            error_kind=error_kind(error),
            latency=None,
            peak_allocated_mib=peak_allocated,
            peak_reserved_mib=peak_reserved,
        )


def is_out_of_memory(error: BaseException) -> bool:
    """Classify OOM without persisting an exception message to public results."""

    oom_type = getattr(torch, "OutOfMemoryError", RuntimeError)
    if isinstance(error, oom_type):
        return True
    return "out of memory" in str(error).lower()


def error_kind(error: BaseException) -> str:
    """Return a stable, non-sensitive exception category."""

    if is_out_of_memory(error):
        return "oom"
    if isinstance(error, (ImportError, ModuleNotFoundError)):
        return "dependency_error"
    if isinstance(error, (ValueError, TypeError)):
        return "configuration_error"
    if isinstance(error, RuntimeError):
        return "runtime_error"
    return type(error).__name__.lower()


def disable_aot_donated_buffers() -> bool:
    """Keep repeated compiled backward calls compatible with ``retain_graph=True``."""

    try:
        from torch._functorch import config as functorch_config
    except ImportError:
        return False
    functorch_config.donated_buffer = False
    return True


def cleanup_cuda() -> None:
    """Release Python references and allocator cache between independent rows."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def measurement_csv_fields(measurement: PhaseMeasurement) -> CsvRow:
    """Flatten a measured phase into common result columns."""

    latency = measurement.latency
    return {
        "timer": latency.timer if latency is not None else None,
        "measurement_sample_count": latency.sample_count if latency is not None else None,
        "measurement_duration_ms": round(latency.measured_duration_ms, 4) if latency is not None else None,
        "p20_ms": round(latency.p20_ms, 6) if latency is not None else None,
        "p50_ms": round(latency.p50_ms, 6) if latency is not None else None,
        "p80_ms": round(latency.p80_ms, 6) if latency is not None else None,
        "peak_allocated_mib": round(measurement.peak_allocated_mib, 3) if measurement.peak_allocated_mib is not None else None,
        "peak_reserved_mib": round(measurement.peak_reserved_mib, 3) if measurement.peak_reserved_mib is not None else None,
        "status": measurement.status,
        "error_kind": measurement.error_kind,
    }


def maximum_peak(measurements: Sequence[PhaseMeasurement]) -> tuple[float | None, float | None]:
    """Calculate real maxima while preserving missing metrics as missing."""

    allocated_values = [measurement.peak_allocated_mib for measurement in measurements if measurement.peak_allocated_mib is not None]
    reserved_values = [measurement.peak_reserved_mib for measurement in measurements if measurement.peak_reserved_mib is not None]
    return (
        max(allocated_values) if allocated_values else None,
        max(reserved_values) if reserved_values else None,
    )


def max_error(actual: Tensor, reference: Tensor) -> dict[str, float]:
    """Return max absolute and relative error with an explicit zero-safe denominator."""

    difference = (actual - reference).abs()
    denominator = reference.abs().clamp_min(torch.finfo(reference.dtype).eps)
    relative = difference / denominator
    return {
        "max_abs_error": float(difference.max().item()),
        "max_rel_error": float(relative.max().item()),
    }


def all_success(rows: Sequence[Mapping[str, CsvValue]]) -> bool:
    """Return whether every result row is an actual successful measurement."""

    return bool(rows) and all(row.get("status") == "success" for row in rows)


def stderr(message: str) -> None:
    """Emit user-facing status without polluting result artifacts."""

    print(message, file=sys.stderr)
