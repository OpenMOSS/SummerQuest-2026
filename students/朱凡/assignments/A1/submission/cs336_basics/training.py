"""Reusable training loop for the assignment's Transformer language model."""

from __future__ import annotations

import json
import os
import socket
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor

from .checkpointing import load_checkpoint, save_checkpoint
from .data import get_batch
from .model import TransformerLM
from .nn_utils import clip_gradients, cross_entropy
from .optimizer import AdamW, cosine_learning_rate


@dataclass(slots=True)
class TrainingConfig:
    train_data: str
    validation_data: str
    output_dir: str
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1_344
    rope_theta: float = 10_000.0
    norm_mode: Literal["pre", "post", "none"] = "pre"
    use_rope: bool = True
    ffn_type: Literal["swiglu", "silu"] = "swiglu"
    batch_size: int = 32
    max_iters: int = 5_000
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_iters: int = 100
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 500
    data_dtype: str = "uint16"
    device: str = "cpu"
    seed: int = 1337
    resume_from: str | None = None
    compile_model: bool = False
    autocast_dtype: str | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None
    tied_embeddings: bool = False
    max_train_seconds: float | None = None

    def validate(self) -> None:
        positive_fields = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "batch_size": self.batch_size,
            "max_iters": self.max_iters,
            "eval_interval": self.eval_interval,
            "eval_iters": self.eval_iters,
            "log_interval": self.log_interval,
            "checkpoint_interval": self.checkpoint_interval,
        }
        for name, value in positive_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.warmup_iters < 0 or self.warmup_iters > self.max_iters:
            raise ValueError("warmup_iters must be between 0 and max_iters")
        if self.autocast_dtype not in (None, "float16", "bfloat16"):
            raise ValueError("autocast_dtype must be float16, bfloat16, or null")
        if self.norm_mode not in ("pre", "post", "none"):
            raise ValueError("norm_mode must be pre, post, or none")
        if self.ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type must be swiglu or silu")
        if self.max_train_seconds is not None and self.max_train_seconds <= 0:
            raise ValueError("max_train_seconds must be positive or null")


def load_token_data(path: str | Path, dtype: str | np.dtype = "uint16") -> np.memmap:
    """Open a flat binary token file without loading it into memory."""

    data = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if data.ndim != 1:
        raise ValueError("token data must be flat")
    return data


def _autocast_context(config: TrainingConfig):
    if config.autocast_dtype is None:
        return nullcontext()
    device_type = torch.device(config.device).type
    dtype = getattr(torch, config.autocast_dtype)
    return torch.autocast(device_type=device_type, dtype=dtype)


@torch.inference_mode()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    config: TrainingConfig,
) -> float:
    """Estimate mean per-token loss from independently sampled batches."""

    was_training = model.training
    model.eval()
    losses: list[Tensor] = []
    try:
        for _ in range(config.eval_iters):
            inputs, targets = get_batch(dataset, config.batch_size, config.context_length, config.device)
            with _autocast_context(config):
                logits = model(inputs)
                loss = cross_entropy(logits, targets)
            losses.append(loss.detach().float().cpu())
    finally:
        model.train(was_training)
    return float(torch.stack(losses).mean())


def _append_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, sort_keys=True) + "\n")


def _read_metric_offsets(path: Path) -> tuple[float, float]:
    """Return the largest persisted training/process elapsed values."""

    training_elapsed = 0.0
    process_elapsed = 0.0
    if not path.exists():
        return training_elapsed, process_elapsed
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            training_elapsed = max(
                training_elapsed,
                float(record.get("training_elapsed_seconds", record.get("elapsed_seconds", 0.0))),
            )
            process_elapsed = max(
                process_elapsed,
                float(record.get("process_elapsed_seconds", record.get("elapsed_seconds", 0.0))),
            )
    return training_elapsed, process_elapsed


def train(config: TrainingConfig) -> TransformerLM:
    """Train a language model, periodically evaluate it, and save checkpoints."""

    process_started_at = time.perf_counter()
    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    metrics_path = output_dir / "metrics.jsonl"
    if config.resume_from is None and metrics_path.exists() and metrics_path.stat().st_size:
        raise FileExistsError(f"refusing to append a new run to existing metrics: {metrics_path}")
    config_path.write_text(json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8")
    training_elapsed_offset, process_elapsed_offset = _read_metric_offsets(metrics_path)

    train_data = load_token_data(config.train_data, config.data_dtype)
    validation_data = load_token_data(config.validation_data, config.data_dtype)
    if train_data.size and int(train_data.max()) >= config.vocab_size:
        raise ValueError("training data contains a token ID outside vocab_size")
    if validation_data.size and int(validation_data.max()) >= config.vocab_size:
        raise ValueError("validation data contains a token ID outside vocab_size")

    model = TransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
        norm_mode=config.norm_mode,
        use_rope=config.use_rope,
        ffn_type=config.ffn_type,
        tied_embeddings=config.tied_embeddings,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    completed_iterations = 0
    if config.resume_from is not None:
        completed_iterations = load_checkpoint(
            config.resume_from,
            model,
            optimizer,
            map_location=config.device,
        )
    if completed_iterations > config.max_iters:
        raise ValueError("checkpoint iteration is greater than max_iters")

    training_model: torch.nn.Module = model
    if config.compile_model:
        if torch.device(config.device).type == "mps":
            training_model = cast(torch.nn.Module, torch.compile(model, backend="aot_eager"))
        else:
            training_model = cast(torch.nn.Module, torch.compile(model))

    wandb_run = None
    if config.wandb_project:
        import wandb

        wandb_id_path = output_dir / "wandb_run_id.txt"
        wandb_run_id = wandb_id_path.read_text(encoding="utf-8").strip() if wandb_id_path.exists() else None
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=asdict(config),
            id=wandb_run_id,
            resume="allow",
        )
        if wandb_run is not None and not wandb_run_id:
            wandb_id_path.write_text(wandb_run.id + "\n", encoding="utf-8")

    training_model.train()
    started_at = time.perf_counter()
    run_id = output_dir.name
    common_metrics = {
        "run_id": run_id,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resume_from": config.resume_from,
    }

    def elapsed_metrics() -> dict[str, float]:
        training_elapsed = training_elapsed_offset + (time.perf_counter() - started_at)
        process_elapsed = process_elapsed_offset + (time.perf_counter() - process_started_at)
        return {
            "elapsed_seconds": training_elapsed,
            "training_elapsed_seconds": training_elapsed,
            "process_elapsed_seconds": process_elapsed,
        }

    try:
        for iteration in range(completed_iterations, config.max_iters):
            learning_rate = cosine_learning_rate(
                iteration,
                config.max_learning_rate,
                config.min_learning_rate,
                config.warmup_iters,
                max(config.max_iters - 1, config.warmup_iters),
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            inputs, targets = get_batch(train_data, config.batch_size, config.context_length, config.device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(config):
                logits = training_model(inputs)
                loss = cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                completed = iteration + 1
                metrics = {
                    "iteration": completed,
                    "train_loss": float(loss.detach()),
                    "learning_rate": learning_rate,
                    "tokens_seen": completed * config.batch_size * config.context_length,
                    "stop_reason": "non_finite_loss",
                    **elapsed_metrics(),
                    **common_metrics,
                }
                _append_metrics(metrics_path, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed)
                raise FloatingPointError(f"non-finite loss at iteration {completed}")
            loss.backward()
            if any(
                parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            ):
                completed = iteration + 1
                metrics = {
                    "iteration": completed,
                    "train_loss": float(loss.detach()),
                    "learning_rate": learning_rate,
                    "tokens_seen": completed * config.batch_size * config.context_length,
                    "stop_reason": "non_finite_gradient",
                    **elapsed_metrics(),
                    **common_metrics,
                }
                _append_metrics(metrics_path, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed)
                raise FloatingPointError(f"non-finite gradient at iteration {completed}")
            clip_gradients(model.parameters(), config.max_grad_norm)
            optimizer.step()

            completed = iteration + 1
            elapsed_values = elapsed_metrics()
            tokens_seen = completed * config.batch_size * config.context_length

            if completed % config.log_interval == 0 or completed == 1:
                metrics = {
                    "iteration": completed,
                    "train_loss": float(loss.detach()),
                    "learning_rate": learning_rate,
                    "tokens_seen": tokens_seen,
                    **elapsed_values,
                    **common_metrics,
                }
                _append_metrics(metrics_path, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed)
                print(
                    f"iter {completed:6d} | train loss {metrics['train_loss']:.4f} | "
                    f"lr {learning_rate:.3e} | {tokens_seen / max(elapsed_values['training_elapsed_seconds'], 1e-9):,.0f} tok/s",
                    flush=True,
                )

            time_limit_reached = (
                config.max_train_seconds is not None
                and elapsed_values["process_elapsed_seconds"] >= config.max_train_seconds
            )
            should_evaluate = completed % config.eval_interval == 0 or completed == config.max_iters or time_limit_reached
            if should_evaluate:
                validation_loss = estimate_loss(training_model, validation_data, config)
                metrics = {
                    "iteration": completed,
                    "validation_loss": validation_loss,
                    "tokens_seen": tokens_seen,
                    **elapsed_metrics(),
                    **common_metrics,
                }
                if time_limit_reached:
                    metrics["stop_reason"] = "max_train_seconds"
                _append_metrics(metrics_path, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=completed)
                print(f"iter {completed:6d} | validation loss {validation_loss:.4f}", flush=True)

            if completed % config.checkpoint_interval == 0 or completed == config.max_iters or time_limit_reached:
                save_checkpoint(model, optimizer, completed, output_dir / f"checkpoint_{completed:07d}.pt")
                save_checkpoint(model, optimizer, completed, output_dir / "checkpoint_latest.pt")
            if time_limit_reached:
                break
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    return model
