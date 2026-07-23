import math
import os
from collections.abc import Iterable
from typing import BinaryIO, IO

import numpy as np
import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if inputs.dtype in {torch.float16, torch.bfloat16}:
        inputs = inputs.float()
    shifted = inputs - inputs.max(dim=-1, keepdim=True).values
    logsumexp = shifted.exp().sum(dim=-1).log()
    target_logits = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str):
    starts = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(np.asarray(dataset[i : i + context_length]).astype(np.int64)) for i in starts.tolist()])
    y = torch.stack([torch.from_numpy(np.asarray(dataset[i + 1 : i + context_length + 1]).astype(np.int64)) for i in starts.tolist()])
    return x.to(device), y.to(device)


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> None:
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    norm = torch.sqrt(sum(torch.sum(g.detach().float() ** 2) for g in grads))
    scale = min(1.0, max_norm / (norm.item() + 1e-6))
    if scale < 1:
        for grad in grads:
            grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                g = p.grad
                state["exp_avg"].mul_(b1).add_(g, alpha=1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(g, g, value=1 - b2)
                step_size = group["lr"] * math.sqrt(1 - b2**t) / (1 - b1**t)
                p.addcdiv_(state["exp_avg"], state["exp_avg_sq"].sqrt().add_(group["eps"]), value=-step_size)
                p.mul_(1 - group["lr"] * group["weight_decay"])
        return loss


def cosine_schedule(it, max_lr, min_lr, warmup_iters, cycle_iters):
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if cycle_iters <= warmup_iters:
        return min_lr
    if it > cycle_iters:
        return min_lr
    ratio = (it - warmup_iters) / (cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


def save_checkpoint(model, optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src, model, optimizer) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
