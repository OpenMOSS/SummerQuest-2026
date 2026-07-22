"""List or sequentially execute local TOML-configured A1 experiment sweeps."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cs336_basics.training import load_config, train_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="TOML file containing a [sweep] table")
    parser.add_argument("--dry-run", "--list", action="store_true", dest="dry_run", help="List runs without training")
    parser.add_argument("--run", action="append", dest="selected_runs", help="Only execute a named run; repeatable")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a failed run")
    return parser.parse_args()


def _deep_merge(destination: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(destination.get(key), dict):
            _deep_merge(destination[key], value)
        else:
            destination[key] = copy.deepcopy(value)


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"invalid dotted override key: {key!r}")
    current = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise TypeError(f"cannot set {key!r}: {part!r} is not a table")
        current = child
    current[parts[-1]] = copy.deepcopy(value)


def _safe_name(value: Any) -> str:
    text = str(value).lower().replace("-", "m").replace(".", "p")
    return re.sub(r"[^a-z0-9_]+", "_", text).strip("_") or "run"


def _run_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    sweep = config.get("sweep")
    if not isinstance(sweep, Mapping):
        raise KeyError("configuration must contain a [sweep] table")
    explicit_runs = sweep.get("runs")
    if explicit_runs is not None:
        if not isinstance(explicit_runs, list) or not explicit_runs:
            raise ValueError("sweep.runs must be a non-empty array of tables")
        specs: list[dict[str, Any]] = []
        for index, item in enumerate(explicit_runs):
            if not isinstance(item, Mapping):
                raise TypeError("each sweep.runs item must be a table")
            run_item = dict(item)
            name = str(run_item.get("name", f"run_{index:02d}"))
            overrides = run_item.get("overrides", {})
            dotted = run_item.get("set", {})
            if not isinstance(overrides, Mapping) or not isinstance(dotted, Mapping):
                raise TypeError("run overrides and set fields must be tables")
            specs.append({"name": name, "overrides": dict(overrides), "set": dict(dotted)})
        return specs

    parameter = sweep.get("parameter")
    values = sweep.get("values")
    if not isinstance(parameter, str) or not isinstance(values, list) or not values:
        raise ValueError("[sweep] must define runs, or a parameter plus non-empty values")
    names = sweep.get("names")
    if names is not None and (not isinstance(names, list) or len(names) != len(values)):
        raise ValueError("sweep.names must have the same length as sweep.values")
    return [
        {
            "name": str(names[index]) if names is not None else f"{parameter.rsplit('.', 1)[-1]}_{_safe_name(value)}",
            "overrides": {},
            "set": {parameter: value},
        }
        for index, value in enumerate(values)
    ]


def materialize_runs(config: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Return ``(name, effective_config, public_overrides)`` for every run."""

    sweep = config["sweep"]
    output_root = Path(str(sweep.get("output_root", "runs/sweeps")))
    materialized: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for spec in _run_specs(config):
        name = spec["name"]
        if name in seen:
            raise ValueError(f"duplicate sweep run name: {name}")
        seen.add(name)
        effective = copy.deepcopy(config)
        effective.pop("sweep", None)
        _deep_merge(effective, spec["overrides"])
        for key, value in spec["set"].items():
            _set_dotted(effective, str(key), value)
        run_table = effective.setdefault("run", {})
        if not isinstance(run_table, dict):
            raise TypeError("[run] must be a table")
        nested_run_override = spec["overrides"].get("run", {})
        has_explicit_output = (
            isinstance(nested_run_override, Mapping) and "output_dir" in nested_run_override
        ) or "run.output_dir" in spec["set"]
        if not has_explicit_output:
            run_table["output_dir"] = str(output_root / name)
        public_overrides = {"nested": spec["overrides"], "dotted": spec["set"]}
        materialized.append((name, effective, public_overrides))
    return materialized


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runs = materialize_runs(config)
    if args.selected_runs:
        requested = set(args.selected_runs)
        known = {name for name, _, _ in runs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown sweep run(s): {', '.join(sorted(unknown))}")
        runs = [run for run in runs if run[0] in requested]

    plan = [
        {"name": name, "output_dir": effective["run"]["output_dir"], "overrides": overrides}
        for name, effective, overrides in runs
    ]
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        return

    results: list[dict[str, Any]] = []
    for name, effective, _ in runs:
        try:
            summary = train_from_config(effective)
            results.append({"name": name, "status": "complete", "summary": summary})
        except Exception as error:
            results.append({"name": name, "status": "failed", "error": f"{type(error).__name__}: {error}"})
            if not args.continue_on_error:
                print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False))
                raise
    print(json.dumps(results, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
