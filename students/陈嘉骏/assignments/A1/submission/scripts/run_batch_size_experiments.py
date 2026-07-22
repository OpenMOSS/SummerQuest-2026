from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import TrainingConfig, load_token_dataset, train_language_model


BASELINE_BATCH_SIZE = 128
BASELINE_CONTEXT_LENGTH = 256
BASELINE_STEPS = 10_000
BASELINE_WARMUP_STEPS = 500
BASELINE_LOG_INTERVAL = 10
BASELINE_EVAL_INTERVAL = 250
BASELINE_EVAL_BATCHES = 20
BASELINE_CHECKPOINT_INTERVAL = 1_000
DEFAULT_TOTAL_TOKENS = BASELINE_BATCH_SIZE * BASELINE_CONTEXT_LENGTH * BASELINE_STEPS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fixed-token TinyStories runs across a batch-size sweep.")
    parser.add_argument("--config", default="configs/tinystories_lr6e4.json")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32, 64, 256])
    parser.add_argument("--total-tokens", type=int, default=DEFAULT_TOTAL_TOKENS)
    parser.add_argument("--max-learning-rate", type=float, default=1.2e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1.2e-4)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", help="Overrides training.device in the base config.")
    parser.add_argument("--logs-dir", default="logs/batch_size")
    parser.add_argument("--runs-dir", default="runs/batch_size")
    parser.add_argument("--configs-dir", default="configs/batch_size")
    parser.add_argument("--force", action="store_true", help="Rerun experiments with existing summaries.")
    parser.add_argument(
        "--stop-on-oom",
        action="store_true",
        help="Stop the sweep immediately after the first out-of-memory result.",
    )
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if not args.batch_sizes or any(batch_size <= 0 for batch_size in args.batch_sizes):
        raise ValueError("batch_sizes must contain positive integers.")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise ValueError("batch_sizes must not contain duplicates.")
    if args.total_tokens <= 0:
        raise ValueError("total_tokens must be positive.")
    if not 0 <= args.min_learning_rate <= args.max_learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= max_learning_rate.")


def _scaled_steps(base_steps: int, batch_size: int) -> int:
    numerator = base_steps * BASELINE_BATCH_SIZE
    return max(1, round(numerator / batch_size))


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


def _reset_random_state(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _build_training_config(
    *,
    base: TrainingConfig,
    batch_size: int,
    total_tokens: int,
    max_learning_rate: float,
    min_learning_rate: float,
    device: str,
    logs_dir: Path,
    runs_dir: Path,
) -> TrainingConfig:
    tokens_per_step = batch_size * base.context_length
    if total_tokens % tokens_per_step != 0:
        raise ValueError(
            f"total_tokens={total_tokens} is not divisible by batch_size * context_length={tokens_per_step}."
        )
    max_steps = total_tokens // tokens_per_step
    expected_steps = _scaled_steps(BASELINE_STEPS, batch_size)
    if total_tokens == DEFAULT_TOTAL_TOKENS and max_steps != expected_steps:
        raise RuntimeError("Internal fixed-token step calculation is inconsistent.")

    run_name = f"train_tinystories_bs_{batch_size}"
    return TrainingConfig(
        batch_size=batch_size,
        context_length=base.context_length,
        max_steps=max_steps,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_steps=_scaled_steps(BASELINE_WARMUP_STEPS, batch_size),
        cosine_cycle_steps=max_steps,
        device=device,
        weight_decay=base.weight_decay,
        beta1=base.beta1,
        beta2=base.beta2,
        eps=base.eps,
        max_grad_norm=base.max_grad_norm,
        log_interval=_scaled_steps(BASELINE_LOG_INTERVAL, batch_size),
        eval_interval=_scaled_steps(BASELINE_EVAL_INTERVAL, batch_size),
        eval_batches=_scaled_steps(BASELINE_EVAL_BATCHES, batch_size),
        checkpoint_interval=_scaled_steps(BASELINE_CHECKPOINT_INTERVAL, batch_size),
        output_dir=os.fspath(runs_dir / f"bs_{batch_size}"),
        log_path=os.fspath(logs_dir / f"{run_name}.jsonl"),
        summary_path=os.fspath(logs_dir / f"{run_name}.summary.json"),
    )


def _persist_generated_config(
    *,
    output_path: Path,
    run_name: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    training_config: TrainingConfig,
    seed: int,
) -> None:
    _write_json_atomically(
        output_path,
        {
            "run_name": run_name,
            "seed": seed,
            "model": model_config,
            "data": data_config,
            "training": asdict(training_config),
        },
    )


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def main() -> None:
    args = _build_parser().parse_args()
    _validate_arguments(args)

    with open(args.config, encoding="utf-8") as f:
        configuration = json.load(f)
    model_config: dict[str, Any] = configuration["model"]
    data_config: dict[str, Any] = configuration["data"]
    base_training_config = TrainingConfig(**configuration["training"])
    base_training_config.validate()
    if base_training_config.context_length != BASELINE_CONTEXT_LENGTH:
        raise ValueError(f"Expected context length {BASELINE_CONTEXT_LENGTH}.")

    device = torch.device(base_training_config.device if args.device is None else args.device)
    logs_dir = Path(args.logs_dir)
    runs_dir = Path(args.runs_dir)
    configs_dir = Path(args.configs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_token_dataset(data_config["train"])
    validation_path = data_config.get("validation")
    validation_dataset = load_token_dataset(validation_path) if validation_path is not None else None

    master_summary_path = logs_dir / "summary.json"
    master_summary: dict[str, object] = {
        "experiment_name": "tinystories_batch_size_sweep",
        "batch_sizes": args.batch_sizes,
        "total_tokens_per_run": args.total_tokens,
        "max_learning_rate": args.max_learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "seed": args.seed,
        "model": model_config,
        "runs": [],
    }

    for batch_size in args.batch_sizes:
        run_name = f"train_tinystories_bs_{batch_size}"
        summary_path = logs_dir / f"{run_name}.summary.json"
        training_config = _build_training_config(
            base=base_training_config,
            batch_size=batch_size,
            total_tokens=args.total_tokens,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            device=str(device),
            logs_dir=logs_dir,
            runs_dir=runs_dir,
        )
        _persist_generated_config(
            output_path=configs_dir / f"tinystories_bs_{batch_size}.json",
            run_name=run_name,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            seed=args.seed,
        )

        if summary_path.exists() and not args.force:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            runs = master_summary["runs"]
            assert isinstance(runs, list)
            runs.append({"batch_size": batch_size, "status": "skipped_existing", **existing_summary})
            _write_json_atomically(master_summary_path, master_summary)
            print(f"Skipping batch_size={batch_size}; summary already exists.", flush=True)
            continue

        print(
            f"Starting batch_size={batch_size}, steps={training_config.max_steps}, "
            f"tokens={args.total_tokens}, lr={args.max_learning_rate:.8g}",
            flush=True,
        )
        _reset_random_state(args.seed, device)
        model: TransformerLM | None = None
        start_time = time.perf_counter()
        try:
            model = TransformerLM(**model_config)
            summary = train_language_model(
                model,
                train_dataset,
                validation_dataset,
                training_config,
                run_metadata={
                    "run_name": run_name,
                    "experiment": "batch_size_sweep",
                    "seed": args.seed,
                    "model": model_config,
                    "data": data_config,
                    "total_tokens_budget": args.total_tokens,
                },
            )
            result: dict[str, object] = {
                "batch_size": batch_size,
                "status": "completed",
                **asdict(summary),
            }
        except (torch.OutOfMemoryError, RuntimeError) as error:
            if not _is_cuda_oom(error):
                raise
            result = {
                "batch_size": batch_size,
                "status": "oom",
                "error": str(error),
                "elapsed_seconds": time.perf_counter() - start_time,
                "planned_steps": training_config.max_steps,
                "total_tokens_budget": args.total_tokens,
            }
            _write_json_atomically(summary_path, result)
            print(f"batch_size={batch_size} ran out of memory; continuing.", flush=True)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        runs = master_summary["runs"]
        assert isinstance(runs, list)
        runs.append(result)
        _write_json_atomically(master_summary_path, master_summary)
        if result["status"] == "oom" and args.stop_on_oom:
            print(f"Stopping after first OOM at batch_size={batch_size}.", flush=True)
            break

    print(f"Batch-size sweep summary: {master_summary_path}", flush=True)


if __name__ == "__main__":
    main()
