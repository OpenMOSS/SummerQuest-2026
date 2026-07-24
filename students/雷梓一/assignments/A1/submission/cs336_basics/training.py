from __future__ import annotations

import math
import os
from contextlib import nullcontext
from collections.abc import Iterable
from typing import BinaryIO, IO

import numpy as np
import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    logits = logits.to(torch.float32)
    normalized_logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_partition = torch.log(torch.sum(torch.exp(normalized_logits), dim=-1))
    target_logits = normalized_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_partition - target_logits)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    num_starts = len(dataset) - context_length
    if num_starts <= 0:
        raise ValueError("dataset must contain more than context_length tokens")
    starts = np.random.randint(0, num_starts, size=(batch_size, 1))
    offsets = np.arange(context_length + 1, dtype=np.int64)[None, :]
    sequences = np.asarray(dataset[starts + offsets], dtype=np.int64)
    tensor = torch.from_numpy(sequences).to(device=device, dtype=torch.long, non_blocking=True)
    x = tensor[:, :-1]
    y = tensor[:, 1:]
    return x, y


def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    total_squared = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        gradient_float = gradient.detach().to(torch.float32)
        total_squared += torch.sum(gradient_float * gradient_float)
    total_norm = torch.sqrt(total_squared)
    scale = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0 or weight_decay < 0:
            raise ValueError("eps and weight_decay must be non-negative")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                first_moment = state["exp_avg"]
                second_moment = state["exp_avg_sq"]
                first_moment.mul_(beta1).add_(gradient, alpha=1 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                parameter.mul_(1 - lr * weight_decay)
                adjusted_lr = lr * math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                parameter.addcdiv_(first_moment, torch.sqrt(second_moment) + eps, value=-adjusted_lr)
        return loss


def cosine_learning_rate(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        if warmup_iters == 0:
            return max_learning_rate
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    batches: int,
    amp_dtype: torch.dtype | None = None,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        device_type = torch.device(device).type
        amp_context = (
            torch.autocast(device_type=device_type, dtype=amp_dtype)
            if amp_dtype is not None
            else nullcontext()
        )
        with amp_context:
            logits = model(inputs)
            losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)))
    if was_training:
        model.train()
    return float(torch.stack(losses).mean().item())
