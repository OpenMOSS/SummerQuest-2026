from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import IO, Any, BinaryIO, cast, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer

from cs336_basics.tokenizer_experiments import load_encoded_dataset


def get_batch(
    dataset: npt.NDArray[Any],
    batch_size: int,
    context_length: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Sample random next-token prediction windows from a one-dimensional corpus."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional.")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive.")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more than context_length tokens.")

    num_start_positions = len(dataset) - context_length
    start_positions = np.random.randint(0, num_start_positions, size=batch_size)
    offsets = np.arange(context_length + 1)
    token_windows = np.asarray(dataset[start_positions[:, None] + offsets], dtype=np.int64)
    token_windows_tensor = torch.tensor(token_windows, dtype=torch.long, device=device)
    return token_windows_tensor[:, :-1], token_windows_tensor[:, 1:]


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy computed directly from logits with log-sum-exp stabilization."""
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions.")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets shape must equal logits shape without the vocabulary dimension.")
    if targets.numel() == 0:
        raise ValueError("cross_entropy requires at least one target.")

    computation_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    stable_logits = logits.to(computation_dtype)
    maximum = torch.amax(stable_logits, dim=-1, keepdim=True)
    log_normalizer = maximum.squeeze(-1) + torch.log(torch.sum(torch.exp(stable_logits - maximum), dim=-1))
    target_logits = torch.gather(stable_logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_normalizer - target_logits)


def clip_gradients(parameters: Iterable[Parameter], max_l2_norm: float) -> None:
    """Clip all available gradients using one global L2 norm."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive.")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    squared_norm = sum(float(torch.sum(gradient.detach().to(torch.float32) ** 2)) for gradient in gradients)
    total_norm = math.sqrt(squared_norm)
    clip_coefficient = min(1.0, max_l2_norm / (total_norm + 1e-6))
    if clip_coefficient < 1.0:
        with torch.no_grad():
            for gradient in gradients:
                gradient.mul_(clip_coefficient)


class AdamW(Optimizer):
    """Adam with bias correction and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative.")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1).")
        if eps < 0:
            raise ValueError("eps must be non-negative.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exponential_average = state["exp_avg"]
                squared_exponential_average = state["exp_avg_sq"]

                parameter.mul_(1.0 - learning_rate * weight_decay)
                exponential_average.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                squared_exponential_average.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denominator = squared_exponential_average.sqrt().div_(math.sqrt(bias_correction2)).add_(epsilon)
                parameter.addcdiv_(
                    exponential_average,
                    denominator,
                    value=-learning_rate / bias_correction1,
                )
        return loss


def get_cosine_learning_rate(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by cosine decay and a constant minimum tail."""
    if iteration < 0:
        raise ValueError("iteration must be non-negative.")
    if warmup_iters < 0 or cosine_cycle_iters <= warmup_iters:
        raise ValueError("Require 0 <= warmup_iters < cosine_cycle_iters.")
    if min_learning_rate < 0 or max_learning_rate < min_learning_rate:
        raise ValueError("Require 0 <= min_learning_rate <= max_learning_rate.")

    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters if warmup_iters else max_learning_rate
    if iteration > cosine_cycle_iters:
        return min_learning_rate

    cosine_ratio = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_coefficient = 0.5 * (1.0 + math.cos(math.pi * cosine_ratio))
    return min_learning_rate + cosine_coefficient * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    iteration: int,
    output: str | os.PathLike[str] | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if isinstance(output, (str, os.PathLike)):
        output_path = Path(cast(str | os.PathLike[str], output))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(output_path.name + ".tmp")
        try:
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    else:
        torch.save(checkpoint, output)


def load_checkpoint(
    source: str | os.PathLike[str] | BinaryIO | IO[bytes],
    model: Module,
    optimizer: Optimizer,
) -> int:
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    checkpoint = torch.load(source, map_location=model_device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])


def load_token_dataset(input_path: str | os.PathLike[str]) -> npt.NDArray[Any]:
    """Memory-map either a .npy array or the raw format produced by the tokenizer pipeline."""
    path = Path(input_path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return load_encoded_dataset(path)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    context_length: int
    max_steps: int
    max_learning_rate: float
    min_learning_rate: float
    warmup_steps: int
    cosine_cycle_steps: int
    device: str = "cpu"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 100
    eval_batches: int = 10
    checkpoint_interval: int = 1000
    output_dir: str = "runs/default"
    log_path: str | None = None
    summary_path: str | None = None

    def validate(self) -> None:
        if self.batch_size <= 0 or self.context_length <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size, context_length, and max_steps must be positive.")
        if self.log_interval <= 0 or self.eval_interval <= 0 or self.eval_batches <= 0:
            raise ValueError("logging and evaluation intervals must be positive.")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive.")
        if self.log_path == "" or self.summary_path == "":
            raise ValueError("log_path and summary_path must be non-empty when provided.")
        get_cosine_learning_rate(
            0,
            self.max_learning_rate,
            self.min_learning_rate,
            self.warmup_steps,
            self.cosine_cycle_steps,
        )


@dataclass(frozen=True)
class TrainingSummary:
    final_iteration: int
    final_train_loss: float
    final_validation_loss: float | None
    elapsed_seconds: float
    processed_tokens: int
    checkpoint_path: str


@torch.no_grad()
def evaluate_model(
    model: Module,
    dataset: npt.NDArray[Any],
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: torch.device | str,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    try:
        for _ in range(num_batches):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            logits = model(inputs)
            losses.append(float(cross_entropy(logits, targets)))
    finally:
        model.train(was_training)
    return sum(losses) / len(losses)


def train_language_model(
    model: Module,
    train_dataset: npt.NDArray[Any],
    validation_dataset: npt.NDArray[Any] | None,
    config: TrainingConfig,
    resume_from: str | os.PathLike[str] | None = None,
    run_metadata: dict[str, object] | None = None,
) -> TrainingSummary:
    """Run a configurable training loop with validation, JSONL logs, and checkpoints."""
    config.validate()
    device = torch.device(config.device)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.max_learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )

    start_iteration = 0
    if resume_from is not None:
        start_iteration = load_checkpoint(resume_from, model, optimizer)
    if start_iteration > config.max_steps:
        raise ValueError("Checkpoint iteration exceeds configured max_steps.")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(config.log_path) if config.log_path is not None else output_dir / "train.jsonl"
    summary_path = Path(config.summary_path) if config.summary_path is not None else output_dir / "summary.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if resume_from is None:
        log_path.unlink(missing_ok=True)
        elapsed_offset = 0.0
    else:
        elapsed_offset = _last_logged_wall_clock(log_path)
    start_time = time.perf_counter()
    final_train_loss = float("nan")
    final_validation_loss: float | None = None

    for iteration in range(start_iteration, config.max_steps):
        learning_rate = get_cosine_learning_rate(
            iteration,
            config.max_learning_rate,
            config.min_learning_rate,
            config.warmup_steps,
            config.cosine_cycle_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        inputs, targets = get_batch(
            train_dataset,
            config.batch_size,
            config.context_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        completed_iteration = iteration + 1
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            _append_json_line(
                log_path,
                {
                    "step": completed_iteration,
                    "wall_clock_sec": elapsed_offset + time.perf_counter() - start_time,
                    "train_loss": None,
                    "lr": learning_rate,
                    "processed_tokens": completed_iteration * config.batch_size * config.context_length,
                    "status": "diverged",
                    "divergence_reason": "non_finite_loss",
                },
            )
            raise FloatingPointError(f"Non-finite loss at step {completed_iteration}.")
        loss.backward()
        clip_gradients(model.parameters(), config.max_grad_norm)
        optimizer.step()

        final_train_loss = loss_value
        should_log = completed_iteration % config.log_interval == 0 or completed_iteration == 1
        should_evaluate = validation_dataset is not None and completed_iteration % config.eval_interval == 0
        if should_evaluate:
            assert validation_dataset is not None
            final_validation_loss = evaluate_model(
                model,
                validation_dataset,
                config.batch_size,
                config.context_length,
                config.eval_batches,
                device,
            )

        if should_log or should_evaluate:
            record: dict[str, int | float] = {
                "step": completed_iteration,
                "wall_clock_sec": elapsed_offset + time.perf_counter() - start_time,
                "train_loss": final_train_loss,
                "lr": learning_rate,
                "processed_tokens": completed_iteration * config.batch_size * config.context_length,
            }
            if should_evaluate and final_validation_loss is not None:
                record["val_loss"] = final_validation_loss
            _append_json_line(log_path, record)

        if completed_iteration % config.checkpoint_interval == 0:
            save_checkpoint(
                model,
                optimizer,
                completed_iteration,
                output_dir / f"checkpoint_{completed_iteration:08d}.pt",
            )

    if validation_dataset is not None:
        final_validation_loss = evaluate_model(
            model,
            validation_dataset,
            config.batch_size,
            config.context_length,
            config.eval_batches,
            device,
        )

    final_checkpoint = output_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, config.max_steps, final_checkpoint)
    summary = TrainingSummary(
        final_iteration=config.max_steps,
        final_train_loss=final_train_loss,
        final_validation_loss=final_validation_loss,
        elapsed_seconds=elapsed_offset + time.perf_counter() - start_time,
        processed_tokens=config.max_steps * config.batch_size * config.context_length,
        checkpoint_path=os.fspath(final_checkpoint),
    )
    summary_payload: dict[str, object] = {
        **({} if run_metadata is None else run_metadata),
        "training": asdict(config),
        **asdict(summary),
        "final_val_loss": summary.final_validation_loss,
        "total_training_time_sec": summary.elapsed_seconds,
    }
    _write_json_atomically(summary_path, summary_payload)
    return summary


def _append_json_line(output_path: Path, record: object) -> None:
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def _last_logged_wall_clock(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0

    last_wall_clock = 0.0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            wall_clock = record.get("wall_clock_sec")
            if isinstance(wall_clock, (int, float)):
                last_wall_clock = float(wall_clock)
    return last_wall_clock


def _write_json_atomically(output_path: Path, value: object) -> None:
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
