"""Run the A2-P experiment matrix as independent, resumable processes.

This orchestration layer intentionally keeps each benchmark/profile/memory
configuration in a fresh Python process.  It records failures instead of
silently dropping OOM or tooling-blocked configurations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Job:
    label: str
    module: str
    arguments: tuple[str, ...]
    output: Path


def _json_status(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    value = payload.get("status")
    return str(value) if value is not None else None


def _jobs(root: Path, device: str) -> list[Job]:
    jobs: list[Job] = []
    benchmark_root = root / "benchmark"
    for mode in ("forward", "forward_backward", "train_step"):
        output = benchmark_root / f"baseline_{mode}.json"
        jobs.append(
            Job(
                f"baseline-{mode}",
                "profiling.benchmark",
                (
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
                    "--device",
                    device,
                    "--output",
                    str(output),
                ),
                output,
            )
        )
    for warmup in (0, 5):
        output = benchmark_root / f"train_step_warmup_{warmup}.json"
        jobs.append(
            Job(
                f"train-step-warmup-{warmup}",
                "profiling.benchmark",
                (
                    "--model-size",
                    "small",
                    "--batch-size",
                    "4",
                    "--context-length",
                    "512",
                    "--mode",
                    "train_step",
                    "--warmup",
                    str(warmup),
                    "--steps",
                    "10",
                    "--dtype",
                    "fp32",
                    "--device",
                    device,
                    "--output",
                    str(output),
                ),
                output,
            )
        )

    profile_root = root / "profile"
    for model_size in ("small", "medium"):
        for context in (256, 512, 1024):
            stem = f"{model_size}_ctx{context}"
            output = profile_root / f"{stem}.json"
            jobs.append(
                Job(
                    f"profile-{stem}",
                    "profiling.profile",
                    (
                        "--model-size",
                        model_size,
                        "--batch-size",
                        "4",
                        "--context-length",
                        str(context),
                        "--warmup",
                        "5",
                        "--dtype",
                        "fp32",
                        "--device",
                        device,
                        "--output",
                        str(output),
                        "--trace-output",
                        str(profile_root / f"{stem}.json.trace"),
                        "--table-output",
                        str(profile_root / f"{stem}.operators.csv"),
                    ),
                    output,
                )
            )

    for model_size in ("small", "medium", "large", "xl", "10b"):
        output = root / f"mixed_precision_{model_size}.json"
        jobs.append(
            Job(
                f"mixed-precision-{model_size}",
                "profiling.mixed_precision",
                (
                    "--device",
                    device,
                    "--model-size",
                    model_size,
                    "--warmup",
                    "5",
                    "--steps",
                    "10",
                    "--output",
                    str(output),
                ),
                output,
            )
        )
    memory_root = root / "memory"
    # Keep the required XL matrix and every explicitly required fallback.
    memory_jobs = [
        ("xl", 128, 4, "forward"),
        ("xl", 128, 4, "train_step"),
        ("xl", 2048, 4, "forward"),
        ("xl", 2048, 4, "train_step"),
        # The lab-required fallback is explicitly batch 1; keep it as
        # separate rows rather than relabeling the original configuration.
        ("xl", 2048, 1, "forward"),
        ("xl", 2048, 1, "train_step"),
        ("xl", 1024, 4, "forward"),
        ("xl", 1024, 4, "train_step"),
        ("large", 2048, 4, "forward"),
        ("large", 2048, 4, "train_step"),
    ]
    for model_size, context, batch_size, mode in memory_jobs:
            stem = f"{model_size}_ctx{context}_b{batch_size}_{mode}"
            output = memory_root / f"{stem}.json"
            jobs.append(
                Job(
                    f"memory-{stem}",
                    "profiling.memory_snapshot",
                    (
                        "--model-size",
                        model_size,
                        "--batch-size",
                        str(batch_size),
                        "--context-length",
                        str(context),
                        "--mode",
                        mode,
                        "--warmup",
                        "5",
                        "--dtype",
                        "fp32",
                        "--device",
                        device,
                        "--snapshot",
                        str(memory_root / f"{stem}.pickle"),
                        "--output",
                        str(output),
                    ),
                    output,
                )
            )
    return jobs


def _run_job(job: Job, *, python: str, log_dir: Path, force: bool) -> dict[str, object]:
    previous = _json_status(job.output)
    if previous == "complete" and not force:
        return {"label": job.label, "status": "complete", "action": "skip-existing"}

    job.output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.label}.log"
    command = [python, "-m", job.module, *job.arguments]
    environment = dict(os.environ)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = _json_status(job.output) or ("complete" if completed.returncode == 0 else "failed")
    return {
        "label": job.label,
        "status": status,
        "returncode": completed.returncode,
        "output": job.output.name,
        "log": log_path.name,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.output_root if args.output_root.is_absolute() else REPO_ROOT / args.output_root
    jobs = _jobs(root, args.device)
    if args.list_only:
        for job in jobs:
            print(f"{job.label}: python -m {job.module} ... -> {job.output}")
        return 0

    manifest = root / "run_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for job in jobs:
        result = _run_job(job, python=args.python, log_dir=root / "logs", force=args.force)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    manifest.write_text(json.dumps({"device": args.device, "jobs": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(result["status"] in {"complete", "skipped", "skip-existing"} for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
