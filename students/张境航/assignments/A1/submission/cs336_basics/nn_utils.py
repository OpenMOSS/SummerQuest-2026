import torch
from torch import Tensor
from collections.abc import Iterable
from torch.nn import Parameter

def softmax(x:Tensor, dim: int) -> Tensor:
    max_value=torch.max(x, dim=dim, keepdim=True).values
    shifted= x-max_value
    numerator=torch.exp(shifted)
    denominator=torch.sum(numerator, dim=dim, keepdim=True)

    return numerator/denominator
def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    max_values = torch.max(
        inputs,
        dim=-1,
        keepdim=True,
    ).values

    shifted_inputs = inputs - max_values

    log_sum_exp = (
        max_values.squeeze(-1)
        + torch.log(
            torch.sum(
                torch.exp(shifted_inputs),
                dim=-1,
            )
        )
    )

    correct_logits = inputs.gather(
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    losses = log_sum_exp - correct_logits

    return losses.mean()

def gradient_clipping(
    parameters: Iterable[Parameter],
    max_l2_norm: float,
) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    parameters = list(parameters)

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]

    if not gradients:
        return

    total_squared_norm = torch.zeros(
        (),
        device=gradients[0].device,
        dtype=torch.float32,
    )

    for gradient in gradients:
        total_squared_norm += torch.sum(
            gradient.detach().to(torch.float32) ** 2
        )

    total_norm = torch.sqrt(total_squared_norm)

    clip_coefficient = max_l2_norm / (total_norm + 1e-6)

    if clip_coefficient < 1:
        for gradient in gradients:
            gradient.mul_(
                clip_coefficient.to(
                    device=gradient.device,
                    dtype=gradient.dtype,
                )
            )