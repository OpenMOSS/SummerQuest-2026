from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experiment_utils import apply_sets, load_json, project_path


def flatten_sets(values: Mapping[str, Any], prefix: str = "") -> list[tuple[str, Any]]:
    pairs: list[tuple[str, Any]] = []
    for key, value in values.items():
        dotted = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping) and "." not in str(key):
            pairs.extend(flatten_sets(value, dotted))
        else:
            pairs.append((dotted, value))
    return pairs


def command_for_run(manifest: Mapping[str, Any], run: Mapping[str, Any]) -> list[str]:
    name = str(run["name"])
    sets = dict(flatten_sets(manifest.get("common_set", {})))
    sets.update(dict(flatten_sets(run.get("set", {}))))
    sets.update(
        {
            "run.name": name,
            "logging.path": f"{manifest.get('log_dir', 'logs')}/{name}.jsonl",
            "logging.summary_path": f"{manifest.get('log_dir', 'logs')}/{name}.summary.json",
            "checkpoint.dir": f"{manifest.get('checkpoint_dir', 'checkpoint')}/{name}",
        }
    )
    cmd = [sys.executable, str(project_path("scripts/train.py")), "--config", str(project_path(manifest["base_config"]))]
    for key, value in sorted(sets.items()):
        cmd.extend(["--set", f"{key}={json.dumps(value)}"])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a config-defined experiment suite.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--only", action="append", default=[], help="Run only this run name or 1-based index.")
    parser.add_argument("--set", action="append", default=[], help="Override common_set with dotted.path=JSON.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()

    manifest = load_json(args.config)
    if args.set:
        manifest["common_set"] = apply_sets(dict(manifest.get("common_set", {})), args.set)
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("suite config must contain a non-empty runs array")
    selected: list[Mapping[str, Any]] = []
    wanted = set(args.only)
    for index, run in enumerate(runs, start=1):
        if not wanted or str(index) in wanted or str(run.get("name")) in wanted:
            selected.append(run)
    if not selected:
        raise ValueError(f"no runs matched --only={sorted(wanted)}")

    code = 0
    for index, run in enumerate(selected, start=1):
        cmd = command_for_run(manifest, run)
        print(f"[{index}/{len(selected)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(cmd, cwd=project_path("."))
        if completed.returncode:
            code = completed.returncode
            if not args.keep_going:
                break
    return code


if __name__ == "__main__":
    raise SystemExit(main())
