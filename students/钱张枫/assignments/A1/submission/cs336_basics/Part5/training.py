from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import warnings
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from cs336_basics.Part3.transformer_lm import TransformerLM
from cs336_basics.Part4.adamw import AdamW
from cs336_basics.Part4.cross_entropy import cross_entropy
from cs336_basics.Part4.gradient_clipping import clip_gradient_norm_
from cs336_basics.Part4.learning_rate_schedule import get_lr_cosine_schedule
from cs336_basics.Part5.checkpointing import load_training_checkpoint, save_checkpoint
from cs336_basics.Part5.configuration import ExperimentConfig, ModelConfig, load_experiment_config
from cs336_basics.Part5.data_loading import TokenArray, sample_batch
from cs336_basics.Part5.experiment_logging import JsonlMetricLogger, read_last_metric


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def resolve_device(requested_device: str) -> torch.device:
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    dtype_by_name = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return dtype_by_name[dtype_name]
    except KeyError as error:
        raise ValueError(f"unsupported model dtype: {dtype_name}") from error


def resolve_amp_dtype(dtype_name: str | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported autocast dtype: {dtype_name}")


def build_model(model_config: ModelConfig, *, device: torch.device, dtype: torch.dtype) -> TransformerLM:
    return TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta,
        device=device,
        dtype=dtype,
        norm_mode=model_config.norm_mode,
        use_rope=model_config.use_rope,
        ffn_type=model_config.ffn_type,
    )


def build_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=config.optimizer.max_learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def load_token_dataset(path: str | Path) -> TokenArray:
    dataset_path = resolve_project_path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"token dataset does not exist: {dataset_path}")
    dataset = np.load(dataset_path, mmap_mode="r")
    if not isinstance(dataset, np.ndarray):
        raise TypeError(f"token dataset must be a NumPy array: {dataset_path}")
    if dataset.ndim != 1:
        raise ValueError(f"token dataset must be one-dimensional, got {dataset.shape}: {dataset_path}")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"token dataset must contain integer IDs, got {dataset.dtype}: {dataset_path}")
    return cast(TokenArray, dataset)


@torch.no_grad()
def evaluate_model(
    model: TransformerLM,
    validation_dataset: TokenArray,
    *,
    batch_size: int,
    context_length: int,
    eval_batches: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    seed: int,
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    losses: list[float] = []
    try:
        for _ in range(eval_batches):
            inputs, targets = sample_batch(
                validation_dataset,
                batch_size,
                context_length,
                device,
                generator,
            )
            loss = _forward_loss(model, inputs, targets, device=device, amp_dtype=amp_dtype)
            losses.append(float(loss.detach().to(dtype=torch.float32).item()))
    finally:
        model.train(was_training)
    return math.fsum(losses) / len(losses)


def train_experiment(
    config: ExperimentConfig,
    *,
    device_override: str | None = None,
    resume_override: str | None = None,
    max_steps_override: int | None = None,
    batch_size_override: int | None = None,
    max_learning_rate_override: float | None = None,
    min_learning_rate_override: float | None = None,
    warmup_steps_override: int | None = None,
    cosine_cycle_steps_override: int | None = None,
    experiment_name_override: str | None = None,
    log_dir_override: str | None = None,
    checkpoint_dir_override: str | None = None,
) -> dict[str, Any]:
    """Train one configured experiment and return the persisted summary."""

    training_config = config.training
    if device_override is not None:
        training_config = replace(training_config, device=device_override)
    if resume_override is not None:
        training_config = replace(training_config, resume_from=resume_override)
    if max_steps_override is not None:
        training_config = replace(training_config, max_steps=max_steps_override)
    if batch_size_override is not None:
        training_config = replace(training_config, batch_size=batch_size_override)
    if log_dir_override is not None:
        training_config = replace(training_config, log_dir=log_dir_override)
    if checkpoint_dir_override is not None:
        training_config = replace(training_config, checkpoint_dir=checkpoint_dir_override)
    optimizer_config = config.optimizer
    if max_learning_rate_override is not None:
        optimizer_config = replace(optimizer_config, max_learning_rate=max_learning_rate_override)
    if min_learning_rate_override is not None:
        optimizer_config = replace(optimizer_config, min_learning_rate=min_learning_rate_override)
    if warmup_steps_override is not None:
        optimizer_config = replace(optimizer_config, warmup_steps=warmup_steps_override)
    if cosine_cycle_steps_override is not None:
        optimizer_config = replace(optimizer_config, cosine_cycle_steps=cosine_cycle_steps_override)
    config = replace(
        config,
        experiment_name=experiment_name_override or config.experiment_name,
        training=training_config,
        optimizer=optimizer_config,
    )

    _seed_everything(config.training.seed)
    device = resolve_device(config.training.device)
    dtype = resolve_dtype(config.training.dtype)
    amp_dtype = resolve_amp_dtype(config.training.amp_dtype)
    _validate_device_dtype(device, dtype, amp_dtype)
    _configure_device_math(device)

    train_dataset = load_token_dataset(config.data.train_path)
    validation_dataset = load_token_dataset(config.data.validation_path)
    model = build_model(config.model, device=device, dtype=dtype)
    optimizer = build_optimizer(model, config)

    tokens_per_step = (
        config.training.batch_size
        * config.model.context_length
        * config.training.gradient_accumulation_steps
    )
    config_signature = _training_config_signature(config)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(config.training.seed + 1)
    start_iteration = 0
    processed_tokens = 0
    wall_clock_offset = 0.0
    has_exact_training_state = False
    if config.training.resume_from is not None:
        start_iteration, training_state = load_training_checkpoint(
            resolve_project_path(config.training.resume_from),
            model,
            optimizer,
        )
        if training_state is None:
            warnings.warn(
                "checkpoint has no training_state metadata; resume cannot reproduce the exact batch sequence.",
                stacklevel=2,
            )
            train_generator.manual_seed(config.training.seed + start_iteration + 1)
            processed_tokens = start_iteration * tokens_per_step
        else:
            has_exact_training_state = True
            processed_tokens, wall_clock_offset = _restore_training_state(
                training_state,
                expected_signature=config_signature,
                expected_tokens_per_step=tokens_per_step,
                train_generator=train_generator,
            )
    if start_iteration > config.training.max_steps:
        raise ValueError(
            f"checkpoint iteration {start_iteration} exceeds configured max_steps {config.training.max_steps}."
        )

    log_dir = resolve_project_path(config.training.log_dir)
    previous_metric = read_last_metric(log_dir / "metrics.jsonl") if start_iteration > 0 else None
    if has_exact_training_state:
        _validate_resume_metric(
            previous_metric,
            expected_step=start_iteration,
            expected_processed_tokens=processed_tokens,
            expected_wall_clock_sec=wall_clock_offset,
        )
    elif previous_metric is not None:
        processed_tokens, wall_clock_offset = _restore_legacy_log_progress(
            previous_metric,
            expected_step=start_iteration,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start_time = time.perf_counter()
    final_train_loss: float | None = None
    final_validation_loss: float | None = None
    final_checkpoint: str | None = None
    status = "completed"
    failure_reason: str | None = None
    completed_iteration = start_iteration
    last_learning_rate = config.optimizer.min_learning_rate
    last_evaluated_iteration: int | None = None

    with JsonlMetricLogger(log_dir, append=start_iteration > 0) as logger:
        logger.write_config(config.to_dict())
        try:
            for iteration in range(start_iteration, config.training.max_steps):
                learning_rate = get_lr_cosine_schedule(
                    iteration,
                    config.optimizer.max_learning_rate,
                    config.optimizer.min_learning_rate,
                    config.optimizer.warmup_steps,
                    config.optimizer.cosine_cycle_steps,
                )
                last_learning_rate = learning_rate
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = learning_rate

                optimizer.zero_grad(set_to_none=True)
                micro_batch_losses: list[float] = []
                for _ in range(config.training.gradient_accumulation_steps):
                    inputs, targets = sample_batch(
                        train_dataset,
                        config.training.batch_size,
                        config.model.context_length,
                        device,
                        train_generator,
                    )
                    loss = _forward_loss(model, inputs, targets, device=device, amp_dtype=amp_dtype)
                    scalar_loss = float(loss.detach().to(dtype=torch.float32).item())
                    if not math.isfinite(scalar_loss):
                        status = "diverged"
                        failure_reason = "non_finite_train_loss"
                        break
                    (loss / config.training.gradient_accumulation_steps).backward()
                    micro_batch_losses.append(scalar_loss)

                if status == "diverged":
                    final_train_loss = None
                    logger.write_metric(
                        {
                            "step": completed_iteration,
                            "wall_clock_sec": wall_clock_offset + (time.perf_counter() - start_time),
                            "train_loss": None,
                            "val_loss": None,
                            "lr": learning_rate,
                            "processed_tokens": processed_tokens,
                            "status": status,
                            "failure_reason": failure_reason,
                        }
                    )
                    break

                clip_gradient_norm_(model.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()

                completed_iteration = iteration + 1
                processed_tokens += tokens_per_step
                final_train_loss = math.fsum(micro_batch_losses) / len(micro_batch_losses)
                current_wall_clock_sec = wall_clock_offset + (time.perf_counter() - start_time)
                if (
                    config.training.max_wall_clock_sec is not None
                    and current_wall_clock_sec >= config.training.max_wall_clock_sec
                ):
                    status = "time_limit_reached"

                should_evaluate = status == "completed" and (
                    completed_iteration % config.training.eval_interval == 0
                    or completed_iteration == config.training.max_steps
                )
                if should_evaluate:
                    final_validation_loss = evaluate_model(
                        model,
                        validation_dataset,
                        batch_size=config.training.eval_batch_size,
                        context_length=config.model.context_length,
                        eval_batches=config.training.eval_batches,
                        device=device,
                        amp_dtype=amp_dtype,
                        seed=config.training.seed + 10_000,
                    )
                    last_evaluated_iteration = completed_iteration
                    if not math.isfinite(final_validation_loss):
                        status = "diverged"
                        failure_reason = "non_finite_validation_loss"
                        final_validation_loss = None

                current_wall_clock_sec = wall_clock_offset + (time.perf_counter() - start_time)
                if (
                    status == "completed"
                    and config.training.max_wall_clock_sec is not None
                    and current_wall_clock_sec >= config.training.max_wall_clock_sec
                ):
                    status = "time_limit_reached"

                should_checkpoint = (
                    status == "completed"
                    and config.training.checkpoint_interval > 0
                    and completed_iteration % config.training.checkpoint_interval == 0
                )
                should_log = status != "time_limit_reached" and (
                    completed_iteration % config.training.log_interval == 0
                    or should_evaluate
                    or should_checkpoint
                    or status == "diverged"
                )
                if should_log:
                    metric = {
                        "step": completed_iteration,
                        "wall_clock_sec": current_wall_clock_sec,
                        "train_loss": final_train_loss,
                        "val_loss": final_validation_loss if should_evaluate else None,
                        "lr": learning_rate,
                        "processed_tokens": processed_tokens,
                        "status": status,
                        "failure_reason": failure_reason,
                    }
                    logger.write_metric(metric)
                    print(
                        _format_progress(
                            completed_iteration,
                            config.training.max_steps,
                            final_train_loss,
                            final_validation_loss if should_evaluate else None,
                            learning_rate,
                            current_wall_clock_sec,
                        ),
                        flush=True,
                    )

                if status != "completed":
                    break

                if should_checkpoint:
                    final_checkpoint = _save_training_checkpoint(
                        model,
                        optimizer,
                        completed_iteration,
                        config.training.checkpoint_dir,
                        config_signature=config_signature,
                        train_generator=train_generator,
                        processed_tokens=processed_tokens,
                        elapsed_wall_clock_sec=current_wall_clock_sec,
                        tokens_per_step=tokens_per_step,
                    )

            if status == "time_limit_reached":
                if last_evaluated_iteration != completed_iteration:
                    final_validation_loss = evaluate_model(
                        model,
                        validation_dataset,
                        batch_size=config.training.eval_batch_size,
                        context_length=config.model.context_length,
                        eval_batches=config.training.eval_batches,
                        device=device,
                        amp_dtype=amp_dtype,
                        seed=config.training.seed + 10_000,
                    )
                    if not math.isfinite(final_validation_loss):
                        status = "diverged"
                        failure_reason = "non_finite_terminal_validation_loss"
                        final_validation_loss = None
                terminal_wall_clock_sec = wall_clock_offset + (time.perf_counter() - start_time)
                logger.write_metric(
                    {
                        "step": completed_iteration,
                        "wall_clock_sec": terminal_wall_clock_sec,
                        "train_loss": final_train_loss,
                        "val_loss": final_validation_loss,
                        "lr": last_learning_rate,
                        "processed_tokens": processed_tokens,
                        "status": status,
                        "failure_reason": failure_reason,
                        "terminal_evaluation": True,
                    }
                )

            if (
                status in ("completed", "time_limit_reached")
                and config.training.checkpoint_interval > 0
                and completed_iteration > 0
                and (
                    completed_iteration % config.training.checkpoint_interval != 0
                    or final_checkpoint is None
                )
            ):
                final_checkpoint = _save_training_checkpoint(
                    model,
                    optimizer,
                    completed_iteration,
                    config.training.checkpoint_dir,
                    config_signature=config_signature,
                    train_generator=train_generator,
                    processed_tokens=processed_tokens,
                    elapsed_wall_clock_sec=wall_clock_offset + (time.perf_counter() - start_time),
                    tokens_per_step=tokens_per_step,
                )
        except Exception as error:
            status = "failed"
            failure_reason = f"{type(error).__name__}: {error}"
            raise
        except KeyboardInterrupt:
            status = "interrupted"
            failure_reason = "KeyboardInterrupt"
            if config.training.checkpoint_interval > 0 and completed_iteration > 0:
                final_checkpoint = _save_training_checkpoint(
                    model,
                    optimizer,
                    completed_iteration,
                    config.training.checkpoint_dir,
                    config_signature=config_signature,
                    train_generator=train_generator,
                    processed_tokens=processed_tokens,
                    elapsed_wall_clock_sec=wall_clock_offset + (time.perf_counter() - start_time),
                    tokens_per_step=tokens_per_step,
                )
            raise
        finally:
            total_wall_clock_sec = wall_clock_offset + (time.perf_counter() - start_time)
            summary = _build_summary(
                config=config,
                device=device,
                dtype=dtype,
                amp_dtype=amp_dtype,
                parameter_count=parameter_count,
                ffn_hidden_dim=model.ffn_hidden_dim,
                train_dataset_tokens=len(train_dataset),
                validation_dataset_tokens=len(validation_dataset),
                start_iteration=start_iteration,
                completed_iteration=completed_iteration,
                processed_tokens=processed_tokens,
                total_wall_clock_sec=total_wall_clock_sec,
                final_train_loss=final_train_loss,
                final_validation_loss=final_validation_loss,
                final_checkpoint=final_checkpoint,
                status=status,
                failure_reason=failure_reason,
            )
            logger.write_summary(summary)

    return summary


def train_from_config(
    config_path: str | Path,
    **overrides: str | int | float | None,
) -> dict[str, Any]:
    config = load_experiment_config(config_path)
    return train_experiment(
        config,
        device_override=cast(str | None, overrides.get("device_override")),
        resume_override=cast(str | None, overrides.get("resume_override")),
        max_steps_override=cast(int | None, overrides.get("max_steps_override")),
        batch_size_override=cast(int | None, overrides.get("batch_size_override")),
        max_learning_rate_override=cast(float | None, overrides.get("max_learning_rate_override")),
        min_learning_rate_override=cast(float | None, overrides.get("min_learning_rate_override")),
        warmup_steps_override=cast(int | None, overrides.get("warmup_steps_override")),
        cosine_cycle_steps_override=cast(int | None, overrides.get("cosine_cycle_steps_override")),
        experiment_name_override=cast(str | None, overrides.get("experiment_name_override")),
        log_dir_override=cast(str | None, overrides.get("log_dir_override")),
        checkpoint_dir_override=cast(str | None, overrides.get("checkpoint_dir_override")),
    )


def load_model_for_inference(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    *,
    device_override: str | None = None,
) -> tuple[TransformerLM, torch.device]:
    device = resolve_device(device_override or config.training.device)
    dtype = resolve_dtype(config.training.dtype)
    _validate_device_dtype(device, dtype, None)
    model = build_model(config.model, device=device, dtype=dtype)
    raw_checkpoint = torch.load(resolve_project_path(checkpoint_path), map_location=device, weights_only=True)
    if not isinstance(raw_checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping.")
    model_state = raw_checkpoint.get("model_state_dict")
    if not isinstance(model_state, Mapping):
        raise ValueError("checkpoint is missing a model_state_dict mapping.")
    model.load_state_dict(cast(dict[str, Any], model_state))
    model.eval()
    return model, device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_device_dtype(
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype | None,
) -> None:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested but is not available.")
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 training is not supported on CPU; use float32 or bfloat16.")
    if amp_dtype is not None:
        if dtype != torch.float32:
            raise ValueError("autocast requires float32 model parameters and optimizer state.")
        if amp_dtype != torch.bfloat16:
            raise ValueError("only bfloat16 autocast is supported.")
        if device.type not in ("cpu", "cuda"):
            raise ValueError("bfloat16 autocast is supported only on CPU or CUDA in this training entry point.")
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16 autocast.")


def _configure_device_math(device: torch.device) -> None:
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True


def _forward_loss(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> torch.Tensor:
    if amp_dtype is None:
        return cross_entropy(model(inputs), targets)
    with torch.autocast(device_type=device.type, dtype=amp_dtype):
        logits = model(inputs)
    return cross_entropy(logits, targets)


def _save_training_checkpoint(
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    checkpoint_dir: str | Path,
    *,
    config_signature: str,
    train_generator: torch.Generator,
    processed_tokens: int,
    elapsed_wall_clock_sec: float,
    tokens_per_step: int,
) -> str:
    directory = resolve_project_path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / f"step_{iteration:08d}.pt"
    temporary_checkpoint_path = checkpoint_path.with_suffix(".pt.tmp")
    save_checkpoint(
        model,
        optimizer,
        iteration,
        temporary_checkpoint_path,
        training_state={
            "version": 1,
            "config_signature": config_signature,
            "train_generator_state": train_generator.get_state(),
            "processed_tokens": processed_tokens,
            "elapsed_wall_clock_sec": elapsed_wall_clock_sec,
            "tokens_per_step": tokens_per_step,
        },
    )
    os.replace(temporary_checkpoint_path, checkpoint_path)
    latest_path = directory / "latest_checkpoint.txt"
    temporary_latest_path = directory / ".latest_checkpoint.txt.tmp"
    temporary_latest_path.write_text(checkpoint_path.name + "\n", encoding="utf-8")
    os.replace(temporary_latest_path, latest_path)
    try:
        return str(checkpoint_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(checkpoint_path)


def _training_config_signature(config: ExperimentConfig) -> str:
    payload = {
        "data": config.to_dict()["data"],
        "model": config.to_dict()["model"],
        "optimizer": config.to_dict()["optimizer"],
        "training": {
            "batch_size": config.training.batch_size,
            "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
            "eval_batch_size": config.training.eval_batch_size,
            "seed": config.training.seed,
            "dtype": config.training.dtype,
            "amp_dtype": config.training.amp_dtype,
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _restore_training_state(
    training_state: Mapping[str, Any],
    *,
    expected_signature: str,
    expected_tokens_per_step: int,
    train_generator: torch.Generator,
) -> tuple[int, float]:
    if training_state.get("version") != 1:
        raise ValueError("unsupported training_state version in checkpoint.")
    if training_state.get("config_signature") != expected_signature:
        raise ValueError("checkpoint training configuration does not match the current experiment config.")
    if training_state.get("tokens_per_step") != expected_tokens_per_step:
        raise ValueError("checkpoint tokens_per_step does not match the current batch configuration.")

    generator_state = training_state.get("train_generator_state")
    if not isinstance(generator_state, torch.Tensor):
        raise ValueError("checkpoint training_state is missing a tensor train_generator_state.")
    train_generator.set_state(generator_state.to(device="cpu"))

    processed_tokens = training_state.get("processed_tokens")
    if isinstance(processed_tokens, bool) or not isinstance(processed_tokens, int) or processed_tokens < 0:
        raise ValueError("checkpoint processed_tokens must be a non-negative integer.")
    elapsed = training_state.get("elapsed_wall_clock_sec")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError("checkpoint elapsed_wall_clock_sec must be a number.")
    elapsed_value = float(elapsed)
    if not math.isfinite(elapsed_value) or elapsed_value < 0:
        raise ValueError("checkpoint elapsed_wall_clock_sec must be finite and non-negative.")
    return processed_tokens, elapsed_value


def _validate_resume_metric(
    metric: Mapping[str, Any] | None,
    *,
    expected_step: int,
    expected_processed_tokens: int,
    expected_wall_clock_sec: float,
) -> None:
    if metric is None:
        return
    step = metric.get("step")
    processed_tokens = metric.get("processed_tokens")
    wall_clock_sec = metric.get("wall_clock_sec")
    if step != expected_step:
        raise ValueError("the last log step does not match the checkpoint iteration.")
    if processed_tokens != expected_processed_tokens:
        raise ValueError("the last log processed_tokens value does not match the checkpoint.")
    if isinstance(wall_clock_sec, bool) or not isinstance(wall_clock_sec, (int, float)):
        raise ValueError("the last log wall_clock_sec value is invalid.")
    tolerance = max(1.0, expected_wall_clock_sec * 0.01)
    if abs(float(wall_clock_sec) - expected_wall_clock_sec) > tolerance:
        raise ValueError("the last log wall_clock_sec value does not match the checkpoint.")


def _restore_legacy_log_progress(
    metric: Mapping[str, Any],
    *,
    expected_step: int,
) -> tuple[int, float]:
    if metric.get("step") != expected_step:
        raise ValueError("the last log step does not match the legacy checkpoint iteration.")
    processed_tokens = metric.get("processed_tokens")
    wall_clock_sec = metric.get("wall_clock_sec")
    if isinstance(processed_tokens, bool) or not isinstance(processed_tokens, int) or processed_tokens < 0:
        raise ValueError("the last log processed_tokens value is invalid.")
    if isinstance(wall_clock_sec, bool) or not isinstance(wall_clock_sec, (int, float)):
        raise ValueError("the last log wall_clock_sec value is invalid.")
    wall_clock_value = float(wall_clock_sec)
    if not math.isfinite(wall_clock_value) or wall_clock_value < 0:
        raise ValueError("the last log wall_clock_sec value must be finite and non-negative.")
    return processed_tokens, wall_clock_value


def _format_progress(
    step: int,
    max_steps: int,
    train_loss: float,
    validation_loss: float | None,
    learning_rate: float,
    wall_clock_sec: float,
) -> str:
    validation_text = "-" if validation_loss is None else f"{validation_loss:.6f}"
    return (
        f"step={step}/{max_steps} train_loss={train_loss:.6f} val_loss={validation_text} "
        f"lr={learning_rate:.6g} wall_clock_sec={wall_clock_sec:.2f}"
    )


def _build_summary(
    *,
    config: ExperimentConfig,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype | None,
    parameter_count: int,
    ffn_hidden_dim: int,
    train_dataset_tokens: int,
    validation_dataset_tokens: int,
    start_iteration: int,
    completed_iteration: int,
    processed_tokens: int,
    total_wall_clock_sec: float,
    final_train_loss: float | None,
    final_validation_loss: float | None,
    final_checkpoint: str | None,
    status: str,
    failure_reason: str | None,
) -> dict[str, Any]:
    return {
        "experiment_name": config.experiment_name,
        "status": status,
        "failure_reason": failure_reason,
        "start_step": start_iteration,
        "final_step": completed_iteration,
        "configured_max_steps": config.training.max_steps,
        "configured_max_wall_clock_sec": config.training.max_wall_clock_sec,
        "processed_tokens": processed_tokens,
        "tokens_per_step": (
            config.training.batch_size
            * config.model.context_length
            * config.training.gradient_accumulation_steps
        ),
        "batch_size": config.training.batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "effective_batch_size": config.training.batch_size * config.training.gradient_accumulation_steps,
        "eval_batch_size": config.training.eval_batch_size,
        "context_length": config.model.context_length,
        "d_model": config.model.d_model,
        "d_ff": config.model.d_ff,
        "ffn_hidden_dim": ffn_hidden_dim,
        "num_layers": config.model.num_layers,
        "num_heads": config.model.num_heads,
        "norm_mode": config.model.norm_mode,
        "use_rope": config.model.use_rope,
        "ffn_type": config.model.ffn_type,
        "parameter_count": parameter_count,
        "train_dataset_tokens": train_dataset_tokens,
        "validation_dataset_tokens": validation_dataset_tokens,
        "max_learning_rate": config.optimizer.max_learning_rate,
        "min_learning_rate": config.optimizer.min_learning_rate,
        "warmup_steps": config.optimizer.warmup_steps,
        "weight_decay": config.optimizer.weight_decay,
        "max_grad_norm": config.optimizer.max_grad_norm,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
        "total_wall_clock_sec": total_wall_clock_sec,
        "final_checkpoint": final_checkpoint,
        "device": str(device),
        "dtype": str(dtype).removeprefix("torch."),
        "amp_dtype": None if amp_dtype is None else str(amp_dtype).removeprefix("torch."),
        "seed": config.training.seed,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "platform": platform.platform(),
    }
