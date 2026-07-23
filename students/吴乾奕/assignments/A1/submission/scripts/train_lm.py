#!/usr/bin/env python3
"""Train, validate, log, checkpoint, and resume a decoder-only Transformer LM."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.config import apply_overrides, load_json_config, project_root, resolve_project_path, write_json
from cs336_basics.experiment import (
    atomic_torch_save,
    build_transformer,
    canonical_model_config,
    checkpoint_model_config,
    extract_model_state,
    load_checkpoint_payload,
    parameter_count,
)
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.training import (
    ThroughputMeter,
    append_jsonl,
    autocast_context,
    estimate_loss,
    load_token_array,
    resolve_device,
    resolve_dtype,
    sample_batch,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted JSON key; may be repeated",
    )
    parser.add_argument("--resume", type=Path, help="checkpoint path; overrides training.resume_from")
    return parser.parse_args()


def optimizer_from_config(model: torch.nn.Module, config: dict[str, Any]) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=float(config["max_learning_rate"]),
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.95))),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.1)),
    )


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def assert_resume_compatible(payload: dict[str, Any], config: dict[str, Any]) -> None:
    """Reject silent changes that would make a resumed run internally inconsistent."""

    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        return
    saved_model = checkpoint_model_config(payload, config["model"])
    if canonical_model_config(saved_model) != canonical_model_config(config["model"]):
        raise ValueError("resume model configuration does not match the checkpoint")
    if saved_config.get("data") != config.get("data"):
        raise ValueError("resume data configuration does not match the checkpoint")
    for key in ("seed", "dtype", "parameter_dtype", "amp", "float32_matmul_precision"):
        if saved_config.get(key) != config.get(key):
            raise ValueError(f"resume {key} does not match the checkpoint")
    if saved_config.get("optimizer") != config.get("optimizer"):
        raise ValueError(
            "resume optimizer/schedule configuration does not match the checkpoint; "
            "start a new output directory for a forked experiment"
        )
    saved_training = saved_config.get("training", {})
    current_training = config.get("training", {})
    for key in ("batch_size", "gradient_accumulation_steps", "gradient_clip"):
        if saved_training.get(key, 1) != current_training.get(key, 1):
            raise ValueError(f"resume training.{key} does not match the checkpoint")


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    iteration: int,
    config: dict[str, Any],
    best_validation_loss: float,
    elapsed_seconds: float,
    processed_tokens: int,
    tokens_per_step: int,
    train_rng: np.random.Generator,
    validation_rng: np.random.Generator,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "cs336_a1_training_checkpoint_v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "best_validation_loss": best_validation_loss,
        "elapsed_seconds": float(elapsed_seconds),
        "processed_tokens": int(processed_tokens),
        "tokens_per_step": int(tokens_per_step),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "train_rng_state": train_rng.bit_generator.state,
        "validation_rng_state": validation_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    if scaler.is_enabled():
        payload["grad_scaler"] = scaler.state_dict()
    return payload


def restore_training_state(
    payload: dict[str, Any],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    train_rng: np.random.Generator,
    validation_rng: np.random.Generator,
) -> tuple[int, float]:
    model.load_state_dict(extract_model_state(payload))
    optimizer_state = payload.get("optimizer", payload.get("optimizer_state_dict"))
    if optimizer_state is None:
        raise KeyError("resume checkpoint does not contain optimizer state")
    optimizer.load_state_dict(optimizer_state)
    if scaler.is_enabled() and "grad_scaler" in payload:
        scaler.load_state_dict(payload["grad_scaler"])
    if "python_random_state" in payload:
        random.setstate(payload["python_random_state"])
    if "numpy_random_state" in payload:
        np.random.set_state(payload["numpy_random_state"])
    if "train_rng_state" in payload:
        train_rng.bit_generator.state = payload["train_rng_state"]
    if "validation_rng_state" in payload:
        validation_rng.bit_generator.state = payload["validation_rng_state"]
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if device.type == "cuda" and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state(payload["cuda_rng_state"], device=device)
    elif torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        # Backward compatibility with early checkpoints that stored every
        # visible device. Restore only states for devices visible now.
        for device_index, rng_state in enumerate(payload["cuda_rng_state_all"][: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(rng_state, device=device_index)
    return int(payload.get("iteration", 0)), float(payload.get("best_validation_loss", math.inf))


def main() -> None:
    args = parse_args()
    root = project_root()
    config = apply_overrides(load_json_config(args.config), args.overrides)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(config.get("device", "auto")))
    compute_dtype = resolve_dtype(str(config.get("dtype", "float32")), device)
    parameter_dtype = resolve_dtype(str(config.get("parameter_dtype", "float32")), device)
    amp_enabled = bool(config.get("amp", device.type in {"cuda", "mps"})) and compute_dtype != torch.float32
    if parameter_dtype != torch.float32 and amp_enabled:
        print(
            "warning: parameter_dtype is already low precision; AMP normally uses float32 parameters", file=sys.stderr
        )

    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    optimizer_config = config["optimizer"]

    train_path = resolve_project_path(data_config["train_tokens"], root=root)
    validation_path = resolve_project_path(data_config["validation_tokens"], root=root)
    assert train_path is not None and validation_path is not None
    train_data = load_token_array(train_path)
    validation_data = load_token_array(validation_path)

    output_dir = resolve_project_path(training_config["output_dir"], root=root)
    assert output_dir is not None
    metrics_path = output_dir / "metrics.jsonl"
    resume_path = args.resume or training_config.get("resume_from")
    resume_path = resolve_project_path(resume_path, root=root)
    if metrics_path.exists() and resume_path is None:
        raise FileExistsError(
            f"{metrics_path} already exists; choose a new output_dir or resume the existing checkpoint"
        )
    resume_payload: dict[str, Any] | None = None
    if resume_path is not None:
        resume_payload = load_checkpoint_payload(resume_path, map_location="cpu")
        assert_resume_compatible(resume_payload, config)

    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    if resume_payload is None:
        write_json(output_dir / "resolved_config.json", public_config)
        shutil.copy2(args.config, output_dir / "source_config.json")

    raw_model = build_transformer(model_config, device=device, dtype=parameter_dtype)
    optimizer = optimizer_from_config(raw_model, optimizer_config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and compute_dtype == torch.float16 and amp_enabled
    )

    train_rng = np.random.default_rng(seed)
    validation_rng = np.random.default_rng(seed + 1)
    starting_iteration = 0
    best_validation_loss = math.inf
    elapsed_offset = 0.0
    if resume_payload is not None:
        starting_iteration, best_validation_loss = restore_training_state(
            resume_payload,
            model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            train_rng=train_rng,
            validation_rng=validation_rng,
        )
        elapsed_offset = float(resume_payload.get("elapsed_seconds", 0.0))
        resume_suffix = f"resume_from_step_{starting_iteration:07d}"
        write_json(output_dir / f"resolved_config_{resume_suffix}.json", public_config)
        shutil.copy2(args.config, output_dir / f"source_config_{resume_suffix}.json")

    training_model: torch.nn.Module = raw_model
    compile_model = bool(config.get("compile", False))
    if compile_model:
        compile_backend = config.get("compile_backend")
        compile_kwargs = {} if compile_backend in (None, "") else {"backend": str(compile_backend)}
        training_model = torch.compile(raw_model, **compile_kwargs)

    batch_size = int(training_config["batch_size"])
    gradient_accumulation_steps = int(training_config.get("gradient_accumulation_steps", 1))
    max_steps = int(training_config["max_steps"])
    context_length = int(model_config["context_length"])
    validation_interval = int(training_config.get("validation_interval", 100))
    validation_batches = int(training_config.get("validation_batches", 20))
    validation_batch_size = int(training_config.get("validation_batch_size", batch_size))
    log_interval = int(training_config.get("log_interval", 10))
    checkpoint_interval = int(training_config.get("checkpoint_interval", 1000))
    max_grad_norm = float(training_config.get("gradient_clip", 1.0))
    effective_batch_size = batch_size * gradient_accumulation_steps
    tokens_per_step = effective_batch_size * context_length
    processed_token_offset = starting_iteration * tokens_per_step
    if resume_payload is not None:
        saved_tokens_per_step = int(resume_payload.get("tokens_per_step", tokens_per_step))
        if saved_tokens_per_step != tokens_per_step:
            raise ValueError("resume token budget per step does not match the checkpoint")
        processed_token_offset = int(resume_payload.get("processed_tokens", processed_token_offset))

    if (
        min(
            batch_size,
            gradient_accumulation_steps,
            max_steps,
            validation_interval,
            validation_batches,
            validation_batch_size,
            log_interval,
            checkpoint_interval,
        )
        <= 0
    ):
        raise ValueError("batch, step, validation, log, and checkpoint settings must all be positive")
    if starting_iteration > max_steps:
        raise ValueError(f"checkpoint iteration {starting_iteration} exceeds configured max_steps {max_steps}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision(str(config.get("float32_matmul_precision", "high")))
        torch.cuda.reset_peak_memory_stats(device)

    run_name = str(config.get("run_name", output_dir.name))
    start_record = {
        "event": "start",
        "run_name": run_name,
        "timestamp": time.time(),
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "parameter_dtype": str(parameter_dtype),
        "parameter_count": parameter_count(raw_model),
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "context_length": context_length,
        "starting_iteration": starting_iteration,
        "starting_processed_tokens": processed_token_offset,
        "max_steps": max_steps,
    }
    append_jsonl(metrics_path, start_record)
    print(json.dumps(start_record, sort_keys=True))

    if compile_model and bool(training_config.get("compile_warmup", True)):
        compile_started = time.perf_counter()
        try:
            window = np.asarray(train_data[: context_length + 1])
            warmup_batch = np.repeat(window[None, :], batch_size, axis=0)
            warmup_tensor = torch.as_tensor(warmup_batch, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, compute_dtype, amp_enabled):
                warmup_logits = training_model(warmup_tensor[:, :-1])
                warmup_loss = cross_entropy(warmup_logits, warmup_tensor[:, 1:])
            scaler.scale(warmup_loss).backward()
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            del warmup_logits, warmup_loss, warmup_tensor, warmup_batch, window
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as error:
            is_out_of_memory = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            if not is_out_of_memory:
                record = {
                    "event": "failed",
                    "run_name": run_name,
                    "step": starting_iteration,
                    "phase": "compile_warmup",
                    "error_type": type(error).__name__,
                    "error": str(error).splitlines()[0][:1000],
                }
                append_jsonl(metrics_path, record)
                print(json.dumps(record, sort_keys=True), file=sys.stderr)
                raise SystemExit(74) from error
            record = {
                "event": "oom",
                "run_name": run_name,
                "step": starting_iteration,
                "phase": "compile_warmup",
                "error": str(error).splitlines()[0][:1000],
            }
            if device.type == "cuda":
                record["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
            append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), file=sys.stderr)
            raise SystemExit(75) from error
        compile_warmup_seconds = time.perf_counter() - compile_started
        elapsed_offset += compile_warmup_seconds
        compile_record = {
            "event": "compile_warmup",
            "run_name": run_name,
            "compile_warmup_seconds": compile_warmup_seconds,
        }
        append_jsonl(metrics_path, compile_record)
        print(json.dumps(compile_record, sort_keys=True))

    meter = ThroughputMeter(elapsed_offset=elapsed_offset)
    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        print(f"received signal {signum}; a checkpoint will be written after this step", file=sys.stderr)
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    completed_iteration = starting_iteration
    last_training_loss = math.nan
    try:
        for step in range(starting_iteration, max_steps):
            next_iteration = step + 1
            measure_step_time = next_iteration % log_interval == 0 or next_iteration == 1
            if device.type == "cuda" and measure_step_time:
                torch.cuda.synchronize(device)
            learning_rate = get_lr_cosine_schedule(
                step,
                float(optimizer_config["max_learning_rate"]),
                float(optimizer_config["min_learning_rate"]),
                int(optimizer_config["warmup_iters"]),
                int(optimizer_config.get("cosine_cycle_iters", max_steps)),
            )
            set_optimizer_lr(optimizer, learning_rate)
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), device=device, dtype=torch.float32)
            step_started = time.perf_counter()

            for _ in range(gradient_accumulation_steps):
                inputs, targets = sample_batch(train_data, batch_size, context_length, device, train_rng)
                with autocast_context(device, compute_dtype, amp_enabled):
                    logits = training_model(inputs)
                    micro_loss = cross_entropy(logits, targets)
                    scaled_loss = micro_loss / gradient_accumulation_steps
                accumulated_loss.add_(micro_loss.detach().float())
                scaler.scale(scaled_loss).backward()

            last_training_loss = float((accumulated_loss / gradient_accumulation_steps).cpu())
            if not math.isfinite(last_training_loss):
                raise FloatingPointError(f"non-finite training loss at step {step}: {last_training_loss}")
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_norm = gradient_clipping(raw_model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            del inputs, targets, logits, micro_loss, scaled_loss

            completed_iteration = next_iteration
            processed_tokens = processed_token_offset + (completed_iteration - starting_iteration) * tokens_per_step
            if device.type == "cuda" and measure_step_time:
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_started

            if completed_iteration % log_interval == 0 or completed_iteration == 1:
                record = {
                    "event": "train",
                    "run_name": run_name,
                    "step": completed_iteration,
                    "processed_tokens": processed_tokens,
                    "train_loss": last_training_loss,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm if math.isfinite(gradient_norm) else None,
                    "step_seconds": step_seconds,
                    "step_tokens_per_second": tokens_per_step / step_seconds,
                    "elapsed_seconds": meter.elapsed(),
                    "tokens_per_second": meter.tokens_per_second(
                        (completed_iteration - starting_iteration) * tokens_per_step
                    ),
                }
                if device.type == "cuda":
                    record["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
                append_jsonl(metrics_path, record)
                print(json.dumps(record, sort_keys=True))

            if completed_iteration % validation_interval == 0 or completed_iteration == max_steps:
                validation_loss = estimate_loss(
                    training_model,
                    validation_data,
                    batch_size=validation_batch_size,
                    context_length=context_length,
                    num_batches=validation_batches,
                    device=device,
                    rng=validation_rng,
                    amp_dtype=compute_dtype,
                    amp_enabled=amp_enabled,
                )
                if not math.isfinite(validation_loss):
                    raise FloatingPointError(
                        f"non-finite validation loss at step {completed_iteration}: {validation_loss}"
                    )
                validation_record = {
                    "event": "validation",
                    "run_name": run_name,
                    "step": completed_iteration,
                    "processed_tokens": processed_tokens,
                    "train_loss": last_training_loss,
                    "validation_loss": validation_loss,
                    "perplexity": math.exp(validation_loss) if validation_loss < 80 else None,
                    "elapsed_seconds": meter.elapsed(),
                }
                append_jsonl(metrics_path, validation_record)
                print(json.dumps(validation_record, sort_keys=True))

                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    atomic_torch_save(
                        checkpoint_payload(
                            model=raw_model,
                            optimizer=optimizer,
                            scaler=scaler,
                            device=device,
                            iteration=completed_iteration,
                            config=config,
                            best_validation_loss=best_validation_loss,
                            elapsed_seconds=meter.elapsed(),
                            processed_tokens=processed_tokens,
                            tokens_per_step=tokens_per_step,
                            train_rng=train_rng,
                            validation_rng=validation_rng,
                        ),
                        output_dir / "best.pt",
                    )

            if completed_iteration % checkpoint_interval == 0 or completed_iteration == max_steps or stop_requested:
                payload = checkpoint_payload(
                    model=raw_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    iteration=completed_iteration,
                    config=config,
                    best_validation_loss=best_validation_loss,
                    elapsed_seconds=meter.elapsed(),
                    processed_tokens=processed_tokens,
                    tokens_per_step=tokens_per_step,
                    train_rng=train_rng,
                    validation_rng=validation_rng,
                )
                numbered_path = output_dir / f"checkpoint_step_{completed_iteration:07d}.pt"
                atomic_torch_save(payload, numbered_path)
                atomic_torch_save(payload, output_dir / "latest.pt")

            if stop_requested:
                interrupted_record = {
                    "event": "interrupted",
                    "run_name": run_name,
                    "step": completed_iteration,
                    "processed_tokens": processed_tokens,
                    "elapsed_seconds": meter.elapsed(),
                }
                append_jsonl(metrics_path, interrupted_record)
                print(json.dumps(interrupted_record, sort_keys=True))
                raise SystemExit(130)

    except FloatingPointError as error:
        record = {
            "event": "divergent",
            "run_name": run_name,
            "step": completed_iteration,
            "error": str(error),
            "elapsed_seconds": meter.elapsed(),
        }
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        raise SystemExit(76) from error
    except RuntimeError as error:
        error_text = str(error)
        is_out_of_memory = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in error_text.lower()
        if not is_out_of_memory:
            record = {
                "event": "failed",
                "run_name": run_name,
                "step": completed_iteration,
                "error_type": type(error).__name__,
                "error": error_text.splitlines()[0][:1000],
                "elapsed_seconds": meter.elapsed(),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), file=sys.stderr)
            raise SystemExit(74) from error
        record = {
            "event": "oom",
            "run_name": run_name,
            "step": completed_iteration,
            "error": error_text.splitlines()[0][:1000],
            "elapsed_seconds": meter.elapsed(),
        }
        if device.type == "cuda":
            record["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise SystemExit(75) from error
    except Exception as error:
        record = {
            "event": "failed",
            "run_name": run_name,
            "step": completed_iteration,
            "error_type": type(error).__name__,
            "error": str(error).splitlines()[0][:1000],
            "elapsed_seconds": meter.elapsed(),
        }
        append_jsonl(metrics_path, record)
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        raise SystemExit(74) from error

    completed_record = {
        "event": "completed",
        "run_name": run_name,
        "step": completed_iteration,
        "processed_tokens": processed_token_offset + (completed_iteration - starting_iteration) * tokens_per_step,
        "best_validation_loss": best_validation_loss if math.isfinite(best_validation_loss) else None,
        "elapsed_seconds": meter.elapsed(),
    }
    append_jsonl(metrics_path, completed_record)
    print(json.dumps(completed_record, sort_keys=True))


if __name__ == "__main__":
    main()
