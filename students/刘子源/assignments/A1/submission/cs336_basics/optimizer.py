from __future__ import annotations

import math

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        beta1, beta2 = betas
        bad_args = lr < 0 or eps < 0 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or weight_decay < 0
        if bad_args:
            raise ValueError("invalid AdamW hyperparameter")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                stuff = self.state[param]
                if not stuff:
                    stuff["step"] = 0
                    stuff["exp_avg"] = torch.zeros_like(param)
                    stuff["exp_avg_sq"] = torch.zeros_like(param)
                stuff["step"] += 1

                step_now = stuff["step"]
                avg = stuff["exp_avg"]
                avg_sq = stuff["exp_avg_sq"]
                if decay:
                    param.mul_(1 - lr * decay)
                avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                avg_fix = avg / (1 - beta1**step_now)
                avg_sq_fix = avg_sq / (1 - beta2**step_now)
                param.addcdiv_(avg_fix, avg_sq_fix.sqrt().add_(eps), value=-lr)
        return loss


def get_lr_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if not 0 <= warmup_iters <= cosine_cycle_iters:
        raise ValueError("require 0 <= warmup_iters <= cosine_cycle_iters")
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters if warmup_iters else max_learning_rate
    if iteration <= cosine_cycle_iters:
        left_steps = cosine_cycle_iters - warmup_iters
        went = (iteration - warmup_iters) / left_steps if left_steps else 1.0
        cosine = 0.5 * (1 + math.cos(math.pi * went))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    return min_learning_rate
