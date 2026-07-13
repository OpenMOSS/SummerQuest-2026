"""Custom optimizers for assignment 1."""

import math

import torch
from torch.optim import Optimizer


def _group_tensors(tensors):
    """Group tensors by (device, dtype) for safe foreach operations."""
    groups = {}
    for t in tensors:
        key = (str(t.device), t.dtype)
        groups.setdefault(key, []).append(t)
    return list(groups.values())


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Matches the standard PyTorch AdamW implementation.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            steps = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p.data)
                grads.append(p.grad.data)
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state["step"] += 1
                steps.append(state["step"])

            if not params_with_grad:
                continue

            # Decoupled weight decay.
            if weight_decay != 0:
                for subgroup in _group_tensors(params_with_grad):
                    torch._foreach_mul_(subgroup, 1 - lr * weight_decay)

            # Build aligned groups of (grad, exp_avg, exp_avg_sq) for foreach ops.
            grouped_indices = {}
            for i, (g, m, v) in enumerate(zip(grads, exp_avgs, exp_avg_sqs)):
                key = (str(g.device), g.dtype)
                grouped_indices.setdefault(key, []).append(i)

            for indices in grouped_indices.values():
                g_subgroup = [grads[i] for i in indices]
                m_subgroup = [exp_avgs[i] for i in indices]
                v_subgroup = [exp_avg_sqs[i] for i in indices]

                torch._foreach_mul_(m_subgroup, beta1)
                torch._foreach_add_(m_subgroup, g_subgroup, alpha=1 - beta1)

                torch._foreach_mul_(v_subgroup, beta2)
                torch._foreach_addcmul_(v_subgroup, g_subgroup, g_subgroup, value=1 - beta2)

            # Bias correction and parameter update.
            for param_data, grad, exp_avg, exp_avg_sq, step in zip(
                params_with_grad, grads, exp_avgs, exp_avg_sqs, steps
            ):
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                param_data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
