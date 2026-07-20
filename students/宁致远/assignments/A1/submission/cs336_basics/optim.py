"""AdamW, LR schedule, gradient clipping, batch sampling, checkpoint utils."""

from __future__ import annotations

import math

import numpy as np
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        if lr < 0:
            raise ValueError("lr must be >= 0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                bias1 = 1 - b1**t
                bias2 = 1 - b2**t
                lr_t = lr * math.sqrt(bias2) / bias1
                p.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
        return loss


def cosine_lr(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_iters: int,
) -> float:
    if it < warmup_iters:
        return max_lr * it / max(1, warmup_iters)
    if it > cosine_iters:
        return min_lr
    r = (it - warmup_iters) / max(1, (cosine_iters - warmup_iters))
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)


def clip_grad_l2(parameters, max_norm: float, eps: float = 1e-6) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total = torch.sqrt(sum((g.detach().float() ** 2).sum() for g in grads))
    coef = max_norm / (total + eps)
    if coef < 1:
        for g in grads:
            g.mul_(coef)


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset) - context_length
    ix = np.random.randint(0, n, size=batch_size)
    x = np.stack([dataset[i : i + context_length] for i in ix])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in ix])
    return (
        torch.from_numpy(x).long().to(device),
        torch.from_numpy(y).long().to(device),
    )


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "iter": iteration,
        },
        out,
    )


def load_checkpoint(src, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt["iter"])
