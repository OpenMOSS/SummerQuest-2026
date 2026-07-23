from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ABLATION_CONFIGS = {
    "no_rmsnorm": "configs/tinystories_no_rmsnorm.json",
    "post_norm": "configs/tinystories_post_norm.json",
    "nope": "configs/tinystories_nope.json",
    "silu": "configs/tinystories_silu.json",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run all required TinyStories architecture ablations.")
    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=tuple(ABLATION_CONFIGS),
        default=list(ABLATION_CONFIGS),
    )
    parser.add_argument("--max-learning-rate", type=float, default=1.2e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1.2e-4)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--logs-dir", default="logs/ablation")
    parser.add_argument("--runs-dir", default="runs/ablation")
    parser.add_argument("--configs-dir", default="configs/ablation")
    parser.add_argument("--force", action="store_true", help="Rerun experiments with existing summaries.")
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if not 0 <= args.min_learning_rate <= args.max_learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= max_learning_rate.")
    if len(set(args.ablations)) != len(args.ablations):
        raise ValueError("ablations must not contain duplicates.")


def _write_json_atomically(output_path: Path, value: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _build_run_config(
    *,
    ablation: str,
    source_config: dict[str, Any],
    max_learning_rate: float,
    min_learning_rate: float,
    seed: int,
    device: str,
    logs_dir: Path,
    runs_dir: Path,
) -> dict[str, Any]:
    run_name = f"train_tinystories_ablation_{ablation}"
    training = dict(source_config["training"])
    training.update(
        {
            "batch_size": 128,
            "context_length": 256,
            "max_steps": 10_000,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": 500,
            "cosine_cycle_steps": 10_000,
            "device": device,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "eps": 1e-8,
            "max_grad_norm": 1.0,
            "log_interval": 10,
            "eval_interval": 250,
            "eval_batches": 20,
            "checkpoint_interval": 1_000,
            "output_dir": os.fspath(runs_dir / ablation),
            "log_path": os.fspath(logs_dir / f"{run_name}.jsonl"),
            "summary_path": os.fspath(logs_dir / f"{run_name}.summary.json"),
        }
    )
    return {
        "run_name": run_name,
        "experiment": "architecture_ablation",
        "ablation": ablation,
        "seed": seed,
        "model": source_config["model"],
        "data": source_config["data"],
        "training": training,
    }


def _classify_failure(console_text: str) -> str:
    lowered = console_text.lower()
    if "out of memory" in lowered:
        return "oom"
    if "non-finite loss" in lowered or "non_finite_loss" in lowered:
        return "diverged"
    return "failed"


def main() -> None:
    args = _build_parser().parse_args()
    _validate_arguments(args)

    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / args.logs_dir
    runs_dir = repo_root / args.runs_dir
    configs_dir = repo_root / args.configs_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary_path = repo_root / "logs/lr_sweep/train_tinystories_lr_1p2e-3.summary.json"
    baseline_summary = _load_json(baseline_summary_path) if baseline_summary_path.exists() else None
    master_summary_path = logs_dir / "summary.json"
    master_summary: dict[str, object] = {
        "experiment_name": "tinystories_architecture_ablations",
        "ablations": args.ablations,
        "max_learning_rate": args.max_learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "batch_size": 128,
        "context_length": 256,
        "max_steps": 10_000,
        "total_tokens_per_run": 327_680_000,
        "seed": args.seed,
        "baseline": baseline_summary,
        "runs": [],
    }

    for ablation in args.ablations:
        source_path = repo_root / ABLATION_CONFIGS[ablation]
        source_config = _load_json(source_path)
        run_config = _build_run_config(
            ablation=ablation,
            source_config=source_config,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            seed=args.seed,
            device=args.device,
            logs_dir=Path(args.logs_dir),
            runs_dir=Path(args.runs_dir),
        )
        run_name = run_config["run_name"]
        assert isinstance(run_name, str)
        generated_config_path = configs_dir / f"tinystories_ablation_{ablation}.json"
        summary_path = logs_dir / f"{run_name}.summary.json"
        console_path = logs_dir / f"{run_name}.console.log"
        _write_json_atomically(generated_config_path, run_config)

        runs = master_summary["runs"]
        assert isinstance(runs, list)
        if summary_path.exists() and not args.force:
            existing = _load_json(summary_path)
            runs.append({"ablation": ablation, "status": "skipped_existing", **existing})
            _write_json_atomically(master_summary_path, master_summary)
            print(f"Skipping {ablation}; summary already exists.", flush=True)
            continue

        print(f"Starting ablation={ablation} with config {generated_config_path}", flush=True)
        start_time = time.perf_counter()
        with open(console_path, "w", encoding="utf-8") as console:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/train_lm.py",
                    "--config",
                    os.fspath(generated_config_path.relative_to(repo_root)),
                ],
                cwd=repo_root,
                stdout=console,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if completed.returncode == 0 and summary_path.exists():
            result = {"ablation": ablation, "status": "completed", **_load_json(summary_path)}
            print(f"Completed ablation={ablation}.", flush=True)
        else:
            console_text = console_path.read_text(encoding="utf-8", errors="replace")
            status = _classify_failure(console_text)
            result = {
                "run_name": run_name,
                "ablation": ablation,
                "status": status,
                "return_code": completed.returncode,
                "elapsed_seconds": time.perf_counter() - start_time,
                "final_validation_loss": None,
                "final_val_loss": None,
                "total_training_time_sec": time.perf_counter() - start_time,
                "seed": args.seed,
                "model": run_config["model"],
                "training": run_config["training"],
                "console_path": os.fspath(console_path.relative_to(repo_root)),
            }
            _write_json_atomically(summary_path, result)
            print(f"Ablation={ablation} ended with status={status}; continuing.", flush=True)

        runs.append(result)
        _write_json_atomically(master_summary_path, master_summary)

    print(f"Ablation summary: {master_summary_path.relative_to(repo_root)}", flush=True)


if __name__ == "__main__":
    main()
