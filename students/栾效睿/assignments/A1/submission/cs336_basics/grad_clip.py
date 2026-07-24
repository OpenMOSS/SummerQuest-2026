from collections.abc import Iterable
import torch


def grad_clip(parameters: Iterable[torch.nn.Parameter], M: float, eps=1e-6) -> None:
    if M < 0:
        raise ValueError("M must be non-negative")

    params = list(parameters)
    grads = [parameter.grad for parameter in params if parameter.grad is not None]
    if not grads:
        return None

    total_L2_square = torch.zeros((), device=grads[0].device, dtype=grads[0].dtype)
    for g in grads:
        g_data = g.detach()
        total_L2_square += (g_data**2).sum()

    total_L2 = torch.sqrt(total_L2_square)
    if total_L2 >= M:
        scale = M / (total_L2 + eps)
        for g in grads:
            g.mul_(scale)
    return None
