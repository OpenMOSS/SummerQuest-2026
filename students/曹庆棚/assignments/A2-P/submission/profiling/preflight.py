from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from profiling.config import environment_metadata, public_relative_path, utc_now, write_json


REQUIRED_PACKAGES = (
    "torch",
    "numpy",
    "pandas",
    "matplotlib",
    "humanfriendly",
    "regex",
    "tqdm",
    "wandb",
    "pytest",
    "ruff",
    "ty",
    "einops",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the A2-P runtime before formal experiments")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-cpu", action="store_true", help="permit a CPU-only development check")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def package_versions() -> tuple[dict[str, str | None], list[str]]:
    versions: dict[str, str | None] = {}
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
            missing.append(package)
    return versions, missing


def cuda_smoke(device: torch.device) -> dict[str, Any]:
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    left = torch.randn(64, 64, device=device)
    right = torch.randn(64, 64, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = left @ right
    torch.cuda.synchronize(device)

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities) as profiler:
        with torch.profiler.record_function("preflight/cuda"):
            probe = (left @ right).sum()
        torch.cuda.synchronize(device)

    events = profiler.events()
    has_cuda_event = any("cuda" in str(getattr(event, "device_type", "")).lower() for event in events)
    return {
        "bf16_matmul_dtype": str(result.dtype).removeprefix("torch."),
        "bf16_matmul_finite": bool(torch.isfinite(result).all().item()),
        "profiler_event_count": len(events),
        "profiler_has_cuda_event": has_cuda_event,
        "probe_value_finite": bool(torch.isfinite(probe).item()),
    }


def run_preflight(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    versions, missing = package_versions()
    device = torch.device(args.device)
    checks: dict[str, Any] = {
        "packages_complete": not missing,
        "torch_2_11": torch.__version__.split("+", 1)[0].startswith("2.11."),
        "profiler_api": hasattr(torch, "profiler"),
        "memory_snapshot_api": all(hasattr(torch.cuda.memory, name) for name in ("_record_memory_history", "_snapshot", "_dump_snapshot")),
        "cuda_requested": device.type == "cuda",
        "cuda_available": bool(torch.cuda.is_available()),
    }
    payload: dict[str, Any] = {
        "kind": "a2p_environment_preflight",
        "checked_at": utc_now(),
        "requested_device": str(device),
        "allow_cpu": args.allow_cpu,
        "package_versions": versions,
        "missing_packages": missing,
        "environment": environment_metadata(torch),
        "checks": checks,
        "cuda_smoke": None,
        "output_file": public_relative_path(args.output) if args.output else None,
    }

    if device.type == "cuda" and torch.cuda.is_available():
        checks["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        try:
            payload["cuda_smoke"] = cuda_smoke(device)
        except Exception as exc:
            checks["cuda_smoke"] = False
            payload["cuda_smoke_error"] = exc.__class__.__name__
        else:
            checks["cuda_smoke"] = all(
                (
                    payload["cuda_smoke"]["bf16_matmul_finite"],
                    payload["cuda_smoke"]["profiler_has_cuda_event"],
                    payload["cuda_smoke"]["probe_value_finite"],
                )
            )
    elif device.type == "cuda":
        checks["bf16_supported"] = False
        checks["cuda_smoke"] = False

    required = [
        checks["packages_complete"],
        checks["torch_2_11"],
        checks["profiler_api"],
        checks["memory_snapshot_api"],
    ]
    if device.type == "cuda":
        required.extend((checks["cuda_available"], checks["bf16_supported"], checks["cuda_smoke"]))
    elif not args.allow_cpu:
        required.append(False)

    payload["status"] = "success" if all(required) else "failed"
    if args.output:
        write_json(args.output, payload)
    return payload, 0 if payload["status"] == "success" else 1


def main() -> int:
    payload, return_code = run_preflight(parse_args())
    print(f"preflight_status={payload['status']}")
    for name, value in payload["checks"].items():
        print(f"{name}={value}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
