"""Append immutable Slurm submission records to the experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "experiment_manifest.jsonl"


def code_version() -> dict[str, str | bool | None]:
    digest = hashlib.sha256()
    versioned_paths = sorted(
        path
        for pattern in ("cs336_basics/*.py", "scripts/*.py", "slurm/*.sh")
        for path in REPO_ROOT.glob(pattern)
        if path.is_file()
    )
    for path in versioned_paths:
        digest.update(str(path.relative_to(REPO_ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    snapshot = digest.hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True))
        return {"git_revision": revision, "git_dirty": dirty, "code_sha256": snapshot}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None, "code_sha256": snapshot}


def append_submission(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    payload = config_path.read_bytes()
    record = {
        "event": "submitted",
        "submitted_at": datetime.now(UTC).isoformat(),
        "run_name": args.run_name,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(payload).hexdigest(),
        "job_id": args.job_id,
        "stdout_path": args.stdout,
        "stderr_path": args.stderr,
        "parent_run": args.parent_run,
        "resume_checkpoint": args.resume_checkpoint,
        "submit_host": os.uname().nodename,
        **code_version(),
    }
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stdout", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--parent-run")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    return parser.parse_args()


if __name__ == "__main__":
    append_submission(parse_args())
