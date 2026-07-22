from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
SCHEMA_VERSION = "a2p-profiling-v1"
VOCAB_SIZE = 10_000
DEFAULT_SEED = 42
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Development-only smoke configuration. Never use it as a formal result.
    "tiny": ModelConfig(d_model=64, d_ff=128, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}

FORMAL_MODEL_SIZES = tuple(name for name in MODEL_CONFIGS if name != "tiny")
MODES = ("forward", "forward_backward", "train_step")
DTYPES = ("fp32", "bf16")
STAGE_NAMES = (
    "profile/warmup",
    "profile/measure",
    "zero_grad",
    "forward",
    "loss",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def model_config_dict(name: str) -> dict[str, int]:
    try:
        return asdict(MODEL_CONFIGS[name.lower()])
    except KeyError as exc:
        raise ValueError(f"unknown model size: {name}") from exc


def make_run_name(
    task: str,
    model_size: str,
    batch_size: int,
    context_length: int,
    mode: str,
    dtype: str,
    tool: str,
) -> str:
    fields = (
        task,
        model_size.lower(),
        f"bs{batch_size}",
        f"ctx{context_length}",
        mode,
        dtype,
        tool,
    )
    return "_".join(re.sub(r"[^a-zA-Z0-9_.-]+", "-", field) for field in fields)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def public_relative_path(path: str | Path, base: Path | None = None) -> str:
    base = (base or repo_root()).resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(base).as_posix()
    except ValueError:
        # Public metadata never needs a machine-specific directory. A basename is
        # still sufficient to identify a locally retained trace or snapshot.
        return resolved.name


def public_command(argv: list[str] | None = None) -> str:
    argv = list(sys.argv if argv is None else argv)
    sanitized: list[str] = []
    for index, argument in enumerate(argv):
        prefix = ""
        value = argument
        if argument.startswith("--") and "=" in argument:
            key, value = argument.split("=", 1)
            prefix = f"{key}="
        candidate = Path(value).expanduser()
        if index == 0 or candidate.is_absolute():
            value = public_relative_path(candidate)
        value = value.replace(str(Path.home()), "<home>")
        value = value.replace(str(repo_root()), "<repo>")
        sanitized.append((prefix + value)[:1000])
    return "python " + shlex.join(sanitized)


def git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _driver_version() -> str | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(values) or None


def environment_metadata(torch_module: Any | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "starter_commit": STARTER_COMMIT,
        "workspace_head": git_head(),
        "driver_version": _driver_version(),
    }
    if torch_module is None:
        return metadata

    torch = torch_module
    metadata.update(
        {
            "pytorch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda) if torch.version.cuda else None,
            "cuda_available": bool(torch.cuda.is_available()),
            "tf32_flags": {
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
            },
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }
    )
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_model": properties.name,
                "gpu_memory_mib": round(properties.total_memory / (1024**2), 2),
                "gpu_compute_capability": f"{properties.major}.{properties.minor}",
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            }
        )
    return metadata


_ABSOLUTE_PATH = re.compile(r"(?<![\w.-])/(?:[^\s:]+/)+[^\s:]*")
_PROCESS_ID = re.compile(r"(?i)process\s+\d+")


def safe_error_summary(exc: BaseException) -> str:
    if exc.__class__.__name__ in {"OutOfMemoryError", "CUDAOutOfMemoryError"}:
        return "CUDA out of memory"
    first_line = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    first_line = first_line.replace(str(Path.home()), "<home>")
    first_line = first_line.replace(str(repo_root()), "<repo>")
    first_line = _ABSOLUTE_PATH.sub("<path>", first_line)
    first_line = _PROCESS_ID.sub("process <redacted>", first_line)
    return first_line[:500]


def classify_error(exc: BaseException) -> str:
    name = exc.__class__.__name__
    message = str(exc).lower()
    if name in {"OutOfMemoryError", "CUDAOutOfMemoryError"} or "out of memory" in message:
        return "oom"
    if "cupti" in message:
        return "cupti_error"
    return "tool_error"


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def base_metadata(run_id: str, run_name: str, tool: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_name": run_name,
        "tool": tool,
        "status": "running",
        "failure_stage": None,
        "error_type": None,
        "error_summary": None,
        "command": public_command(),
        "started_at": utc_now(),
        "finished_at": None,
    }
