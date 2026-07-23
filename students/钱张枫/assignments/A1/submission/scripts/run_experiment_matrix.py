from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from cs336_basics.Part5.configuration import load_experiment_config
from cs336_basics.Part5.training import resolve_project_path, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a named group from configs/experiment_matrix.json.")
    parser.add_argument("--manifest", default="configs/experiment_matrix.json")
    parser.add_argument("--group", required=True)
    parser.add_argument("--device", help="Optional device override for every run.")
    parser.add_argument("--max-steps", type=int, help="Optional short-run override for every config.")
    parser.add_argument("--max-learning-rate", type=float)
    parser.add_argument("--min-learning-rate", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = resolve_project_path(args.manifest).resolve()
    config_paths = _load_group(manifest_path, args.group)
    if args.dry_run:
        for config_path in config_paths:
            config = load_experiment_config(config_path)
            print(f"{config.experiment_name}: {config_path}")
        return 0

    results: list[dict[str, Any]] = []
    failures = 0
    for config_path in config_paths:
        try:
            config = load_experiment_config(config_path)
            summary = train_experiment(
                config,
                device_override=args.device,
                max_steps_override=args.max_steps,
                max_learning_rate_override=args.max_learning_rate,
                min_learning_rate_override=args.min_learning_rate,
            )
            results.append(summary)
        except Exception as error:
            failures += 1
            results.append(
                {
                    "config": str(config_path),
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            if args.stop_on_error:
                break
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failures else 0


def _load_group(manifest_path: Path, group_name: str) -> list[Path]:
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("groups"), dict):
        raise ValueError("experiment manifest must contain a groups object.")
    group = value["groups"].get(group_name)
    if not isinstance(group, list) or not all(isinstance(item, str) for item in group):
        available_groups = ", ".join(sorted(value["groups"]))
        raise ValueError(f"unknown or invalid group {group_name!r}; available groups: {available_groups}")
    return [(manifest_path.parent / item).resolve() for item in group]


if __name__ == "__main__":
    raise SystemExit(main())
