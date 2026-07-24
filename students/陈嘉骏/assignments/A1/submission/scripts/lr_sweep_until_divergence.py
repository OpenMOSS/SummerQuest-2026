from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    TrainingConfig,
    clip_gradients,
    cross_entropy,
    evaluate_model,
    get_batch,
    get_cosine_learning_rate,
    load_token_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Increase learning rate until TinyStories training diverges.")
    parser.add_argument("--config", default="configs/tinystories_lr6e4.json")
    parser.add_argument("--start-lr", type=float, help="Defaults to training.max_learning_rate in the config.")
    parser.add_argument("--multiplier", type=float, default=2.0)
    parser.add_argument("--max-lr", type=float, default=0.1)
    parser.add_argument("--max-trials", type=int, default=10)
    parser.add_argument("--steps-per-trial", type=int, help="Defaults to training.max_steps in the config.")
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--device", help="Overrides training.device in the config.")
    parser.add_argument("--output-dir", default="logs/lr_sweep")
    parser.add_argument("--print-interval", type=int, default=100)
    parser.add_argument("--ema-alpha", type=float, default=0.05)
    parser.add_argument("--divergence-ratio", type=float, default=2.0)
    parser.add_argument("--divergence-patience", type=int, default=25)
    parser.add_argument("--absolute-loss-threshold", type=float, default=20.0)
    parser.add_argument("--absolute-loss-patience", type=int, default=3)
    return parser


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.start_lr is not None and args.start_lr <= 0:
        raise ValueError("start_lr must be positive.")
    if args.multiplier <= 1:
        raise ValueError("multiplier must be greater than 1.")
    if args.max_lr <= 0 or args.max_trials <= 0:
        raise ValueError("max_lr and max_trials must be positive.")
    if args.steps_per_trial is not None and args.steps_per_trial <= 0:
        raise ValueError("steps_per_trial must be positive.")
    if args.print_interval <= 0:
        raise ValueError("print_interval must be positive.")
    if not 0 < args.ema_alpha <= 1:
        raise ValueError("ema_alpha must be in (0, 1].")
    if args.divergence_ratio <= 1 or args.divergence_patience <= 0:
        raise ValueError("divergence_ratio must exceed 1 and divergence_patience must be positive.")
    if args.absolute_loss_threshold <= 0 or args.absolute_loss_patience <= 0:
        raise ValueError("absolute loss settings must be positive.")


def _reset_random_state(seed: int, device: torch.device) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _learning_rate_slug(learning_rate: float) -> str:
    mantissa, exponent = f"{learning_rate:.8e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"{mantissa}e{int(exponent)}"


def _append_json_line(output_path: Path, record: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(_json_safe(record), f, ensure_ascii=False, allow_nan=False)
        f.write("\n")


def _write_json_atomically(output_path: Path, value: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(value), f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _run_trial(
    *,
    model_config: dict[str, Any],
    training_config: TrainingConfig,
    train_dataset: Any,
    validation_dataset: Any,
    max_learning_rate: float,
    min_learning_rate_ratio: float,
    steps_per_trial: int,
    seed: int,
    device: torch.device,
    output_dir: Path,
    print_interval: int,
    ema_alpha: float,
    divergence_ratio: float,
    divergence_patience: int,
    absolute_loss_threshold: float,
    absolute_loss_patience: int,
) -> dict[str, object]:
    _reset_random_state(seed, device)
    model = TransformerLM(**model_config)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=max_learning_rate,
        betas=(training_config.beta1, training_config.beta2),
        eps=training_config.eps,
        weight_decay=training_config.weight_decay,
    )

    min_learning_rate = max_learning_rate * min_learning_rate_ratio
    learning_rate_name = _learning_rate_slug(max_learning_rate)
    run_name = f"train_tinystories_lr_{learning_rate_name}"
    log_path = output_dir / f"{run_name}.jsonl"
    summary_path = output_dir / f"{run_name}.summary.json"
    log_path.unlink(missing_ok=True)

    start_time = time.perf_counter()
    ema_loss: float | None = None
    best_ema_loss = math.inf
    relative_divergence_steps = 0
    absolute_divergence_steps = 0
    final_train_loss = math.nan
    final_validation_loss: float | None = None
    status = "running"
    divergence_reason: str | None = None
    completed_steps = 0
    detection_start_step = max(training_config.warmup_steps, 1)

    print(
        f"Starting lr={max_learning_rate:.8g}, min_lr={min_learning_rate:.8g}, steps={steps_per_trial}, seed={seed}",
        flush=True,
    )

    for iteration in range(steps_per_trial):
        learning_rate = get_cosine_learning_rate(
            iteration,
            max_learning_rate,
            min_learning_rate,
            training_config.warmup_steps,
            training_config.cosine_cycle_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = get_batch(
            train_dataset,
            training_config.batch_size,
            training_config.context_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss_value = float(loss.detach())
        completed_steps = iteration + 1
        final_train_loss = loss_value

        if not math.isfinite(loss_value):
            status = "diverged"
            divergence_reason = "non_finite_loss"
        else:
            loss.backward()
            clip_gradients(model.parameters(), training_config.max_grad_norm)
            optimizer.step()

            ema_loss = loss_value if ema_loss is None else ema_alpha * loss_value + (1.0 - ema_alpha) * ema_loss
            if completed_steps >= detection_start_step:
                best_ema_loss = min(best_ema_loss, ema_loss)
                if ema_loss > best_ema_loss * divergence_ratio:
                    relative_divergence_steps += 1
                else:
                    relative_divergence_steps = 0

                if loss_value > absolute_loss_threshold:
                    absolute_divergence_steps += 1
                else:
                    absolute_divergence_steps = 0

                if relative_divergence_steps >= divergence_patience:
                    status = "diverged"
                    divergence_reason = "sustained_loss_rebound"
                elif absolute_divergence_steps >= absolute_loss_patience:
                    status = "diverged"
                    divergence_reason = "absolute_loss_threshold"

        should_evaluate = (
            status != "diverged"
            and validation_dataset is not None
            and completed_steps % training_config.eval_interval == 0
        )
        if should_evaluate:
            final_validation_loss = evaluate_model(
                model,
                validation_dataset,
                training_config.batch_size,
                training_config.context_length,
                training_config.eval_batches,
                device,
            )

        should_log = (
            completed_steps == 1
            or completed_steps % training_config.log_interval == 0
            or should_evaluate
            or status == "diverged"
        )
        if should_log:
            record: dict[str, int | float | str] = {
                "step": completed_steps,
                "wall_clock_sec": time.perf_counter() - start_time,
                "train_loss": loss_value,
                "lr": learning_rate,
                "processed_tokens": completed_steps * training_config.batch_size * training_config.context_length,
                "status": status,
            }
            if ema_loss is not None:
                record["ema_train_loss"] = ema_loss
            if should_evaluate and final_validation_loss is not None:
                record["val_loss"] = final_validation_loss
            if divergence_reason is not None:
                record["divergence_reason"] = divergence_reason
            _append_json_line(log_path, record)

        if completed_steps % print_interval == 0 or should_evaluate or status == "diverged":
            validation_text = "" if final_validation_loss is None else f", val={final_validation_loss:.6f}"
            print(
                f"lr={max_learning_rate:.8g} step={completed_steps}/{steps_per_trial} "
                f"train={loss_value:.6f}{validation_text} status={status}",
                flush=True,
            )

        if status == "diverged":
            break

    if status == "running":
        status = "completed"

    if status == "completed" and validation_dataset is not None:
        final_validation_loss = evaluate_model(
            model,
            validation_dataset,
            training_config.batch_size,
            training_config.context_length,
            training_config.eval_batches,
            device,
        )

    elapsed_seconds = time.perf_counter() - start_time
    summary: dict[str, object] = {
        "run_name": run_name,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "status": status,
        "divergence_reason": divergence_reason,
        "final_iteration": completed_steps,
        "completed_steps": completed_steps,
        "planned_steps": steps_per_trial,
        "processed_tokens": completed_steps * training_config.batch_size * training_config.context_length,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "final_val_loss": final_validation_loss,
        "best_ema_train_loss": None if not math.isfinite(best_ema_loss) else best_ema_loss,
        "elapsed_seconds": elapsed_seconds,
        "total_training_time_sec": elapsed_seconds,
        "seed": seed,
        "model": model_config,
        "training": {
            "batch_size": training_config.batch_size,
            "context_length": training_config.context_length,
            "max_steps": steps_per_trial,
            "max_learning_rate": max_learning_rate,
            "min_learning_rate": min_learning_rate,
            "warmup_steps": training_config.warmup_steps,
            "cosine_cycle_steps": training_config.cosine_cycle_steps,
            "weight_decay": training_config.weight_decay,
            "beta1": training_config.beta1,
            "beta2": training_config.beta2,
            "max_grad_norm": training_config.max_grad_norm,
        },
        "log_path": os.fspath(log_path),
    }
    _write_json_atomically(summary_path, summary)

    del optimizer
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def main() -> None:
    args = _build_parser().parse_args()
    _validate_arguments(args)

    with open(args.config, encoding="utf-8") as f:
        configuration = json.load(f)
    model_config = configuration["model"]
    data_config = configuration["data"]
    training_config = TrainingConfig(**configuration["training"])
    training_config.validate()

    device = torch.device(training_config.device if args.device is None else args.device)
    steps_per_trial = training_config.max_steps if args.steps_per_trial is None else args.steps_per_trial
    start_learning_rate = training_config.max_learning_rate if args.start_lr is None else args.start_lr
    if start_learning_rate > args.max_lr:
        raise ValueError("start_lr cannot exceed max_lr.")

    min_learning_rate_ratio = training_config.min_learning_rate / training_config.max_learning_rate
    train_dataset = load_token_dataset(data_config["train"])
    validation_path = data_config.get("validation")
    validation_dataset = load_token_dataset(validation_path) if validation_path is not None else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_summary: dict[str, object] = {
        "config": args.config,
        "start_learning_rate": start_learning_rate,
        "multiplier": args.multiplier,
        "max_learning_rate": args.max_lr,
        "max_trials": args.max_trials,
        "steps_per_trial": steps_per_trial,
        "seed": args.seed,
        "trials": [],
        "first_divergent_learning_rate": None,
    }

    learning_rate = start_learning_rate
    for _ in range(args.max_trials):
        if learning_rate > args.max_lr:
            break
        trial_summary = _run_trial(
            model_config=model_config,
            training_config=training_config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            max_learning_rate=learning_rate,
            min_learning_rate_ratio=min_learning_rate_ratio,
            steps_per_trial=steps_per_trial,
            seed=args.seed,
            device=device,
            output_dir=output_dir,
            print_interval=args.print_interval,
            ema_alpha=args.ema_alpha,
            divergence_ratio=args.divergence_ratio,
            divergence_patience=args.divergence_patience,
            absolute_loss_threshold=args.absolute_loss_threshold,
            absolute_loss_patience=args.absolute_loss_patience,
        )
        trials = sweep_summary["trials"]
        assert isinstance(trials, list)
        trials.append(trial_summary)

        if trial_summary["status"] == "diverged":
            sweep_summary["first_divergent_learning_rate"] = learning_rate
            _write_json_atomically(output_dir / "summary.json", sweep_summary)
            print(f"Stopped at first divergent learning rate: {learning_rate:.8g}", flush=True)
            return

        _write_json_atomically(output_dir / "summary.json", sweep_summary)
        learning_rate *= args.multiplier

    _write_json_atomically(output_dir / "summary.json", sweep_summary)
    print("No divergent learning rate found within the configured sweep limits.", flush=True)


if __name__ == "__main__":
    main()
