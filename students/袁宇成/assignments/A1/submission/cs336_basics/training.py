from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    maximum = logits.max(dim=-1, keepdim=True).values
    log_normalizer = maximum.squeeze(-1) + (logits - maximum).exp().sum(dim=-1).log()
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more than context_length tokens")
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    offsets = np.arange(context_length, dtype=np.int64)
    indices = starts.numpy()[:, None] + offsets[None, :]
    x = torch.from_numpy(np.asarray(dataset[indices], dtype=np.int64))
    y = torch.from_numpy(np.asarray(dataset[indices + 1], dtype=np.int64))
    return x.to(device), y.to(device)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return
    total = torch.sqrt(sum(parameter.grad.detach().float().square().sum() for parameter in parameters))
    scale = min(1.0, max_l2_norm / (total.item() + 1e-6))
    if scale < 1.0:
        for parameter in parameters:
            parameter.grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0 or eps < 0 or weight_decay < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("invalid optimizer hyperparameter")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
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
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias2) / bias1
                parameter.addcdiv_(state["exp_avg"], state["exp_avg_sq"].sqrt().add_(group["eps"]), value=-step_size)
        return loss


def cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
