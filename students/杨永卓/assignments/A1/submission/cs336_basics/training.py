"""Optimization, batching, and checkpoint helpers for language-model training."""

from __future__ import annotations

import math
import os
from collections.abc import Iterable
from typing import BinaryIO, IO

import numpy as np
import torch
from torch import Tensor, nn


def softmax(inputs: Tensor, dim: int) -> Tensor:
    # softmax 对整体平移不变；每个切片减最大值可防止 exp(inputs) 溢出。
    shifted = inputs - inputs.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    # 输入是 logits 而不是概率。先 shift，再计算 log(sum(exp(logits)))，等价于稳定版 logsumexp。
    max_logits = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_logits
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    # gather 沿 vocab 维取出每个位置正确 token 的 logit：(..., 1) 再 squeeze 为 (...,)。
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more than context_length tokens")
    # 每个样本随机选择起点；y 相对 x 向右平移一位，形成 next-token prediction 目标。
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[start : start + context_length] for start in starts])
    y = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return (
        torch.as_tensor(x, dtype=torch.long, device=device),
        torch.as_tensor(y, dtype=torch.long, device=device),
    )


def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    parameters_with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters_with_grad:
        return
    # 先跨所有参数求一个全局 L2 norm，再按同一比例缩放，保留梯度方向。
    total_squared_norm = torch.zeros((), device=parameters_with_grad[0].grad.device)
    for parameter in parameters_with_grad:
        total_squared_norm += parameter.grad.detach().float().square().sum()
    total_norm = torch.sqrt(total_squared_norm)
    scale = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)
    for parameter in parameters_with_grad:
        parameter.grad.mul_(scale.to(parameter.grad.dtype))


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must be in [0, 1)")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        # closure 是 PyTorch Optimizer API 的兼容入口；通常训练循环已在 step 前完成 backward。
        loss = None if closure is None else closure()
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
                    # 每个参数维护自己的 step、一阶矩 m 和二阶矩 v，第一次更新时初始化。
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                # Adam 的一阶、二阶矩指数滑动平均。
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1
                # Decoupled weight decay 直接收缩参数，不把 L2 项混入梯度。
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)
        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    # 先线性 warmup，随后余弦衰减；超过一个周期时固定在最小学习率。
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    # 同时保存模型、优化器和 iteration，恢复训练时三者必须保持一致。
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration},
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    # 先映射到 CPU，调用方再决定模型应放回哪张设备。
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


# Names kept for compatibility with the user's earlier implementation.
run_softmax = softmax
run_cross_entropy = cross_entropy
run_gradient_clipping = clip_gradients
run_get_lr_cosine_schedule = get_lr_cosine_schedule
run_save_checkpoint = save_checkpoint
run_load_checkpoint = load_checkpoint


def get_adamw_cls():
    return AdamW
