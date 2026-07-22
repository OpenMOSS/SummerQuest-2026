from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, weight_decay=0.02, lr=1e-3, betas=(0.9, 0.95), eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay rate: {weight_decay}")

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "betas": betas,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state assocßiated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                m = state.get("m_hat", torch.zeros_like(p.data))  # init_m
                v = state.get("v_hat", torch.zeros_like(p.data))  # init_v

                grad = p.grad.data  # Get the gradient of loss with respect to p.ß

                # Compute adjusted lr for iteration t
                n_lr = (
                    lr
                    * math.sqrt(1.0 - math.pow(beta2, t))
                    / (1.0 - math.pow(beta1, t))
                )

                p.data -= lr * weight_decay * p.data  # Apply weight decay

                m = beta1 * m + (1.0 - beta1) * grad  # Update the first moment estimate
                v = beta2 * v + (1.0 - beta2) * torch.pow(
                    grad, 2
                )  # Update the second moment estimate
                p.data -= (
                    n_lr * m / (torch.sqrt(v) + eps)
                )  # Apply moment-adjusted weight updates

                state["t"] = t + 1  # Increment iteration number.
                state["v_hat"] = v  # update v
                state["m_hat"] = m  # update m

        return loss
