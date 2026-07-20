#!/usr/bin/env python3
"""Run a sequence of config overrides and record success, failure, or OOM."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cs336_basics.config import project_root, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--set", dest="global_overrides", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    summary_mode = parser.add_mutually_exclusive_group()
    summary_mode.add_argument("--append-summary", action="store_true")
    summary_mode.add_argument("--overwrite-summary", action="store_true")
    return parser.parse_args()


def flatten(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(child_prefix, child)
    else:
        yield prefix, value


def main() -> None:
    args = parse_args()
    root = project_root()
    sweep_path = args.sweep if args.sweep.is_absolute() else root / args.sweep
    with sweep_path.open(encoding="utf-8") as sweep_file:
        sweep = json.load(sweep_file)
    base_config = resolve_project_path(sweep["base_config"], root=root)
    assert base_config is not None
    variants = sweep.get("variants", [])
    if not variants:
        raise ValueError("sweep contains no variants")

    summary_path = resolve_project_path(
        sweep.get("summary", f"artifacts/sweeps/{sweep_path.stem}_summary.jsonl"),
        root=root,
    )
    assert summary_path is not None
    if not args.dry_run:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        if summary_path.exists():
            if args.overwrite_summary:
                summary_path.unlink()
            elif not args.append_summary:
                raise FileExistsError(
                    f"{summary_path} already exists; preserve it, use --append-summary, or explicitly "
                    "replace it with --overwrite-summary after archiving the old run directories"
                )

    for variant in variants:
        name = str(variant["name"])
        command = [sys.executable, str(root / "scripts" / "train_lm.py"), "--config", str(base_config)]
        overrides = dict(variant.get("overrides", {}))
        overrides.setdefault("run_name", name)
        for key, value in flatten("", overrides):
            command.extend(["--set", f"{key}={json.dumps(value)}"])
        for override in args.global_overrides:
            command.extend(["--set", override])
        print(" ".join(command))
        if args.dry_run:
            continue

        started = time.time()
        completed = subprocess.run(command, cwd=root, check=False)
        status_by_return_code = {0: "completed", 74: "failed", 75: "oom", 76: "divergent", 130: "interrupted"}
        record = {
            "sweep": sweep_path.stem,
            "variant": name,
            "return_code": completed.returncode,
            "status": status_by_return_code.get(completed.returncode, "failed"),
            "elapsed_seconds": time.time() - started,
            "base_config": str(sweep["base_config"]),
            "overrides": overrides,
            "global_overrides": args.global_overrides,
        }
        with summary_path.open("a", encoding="utf-8") as summary_file:
            summary_file.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
