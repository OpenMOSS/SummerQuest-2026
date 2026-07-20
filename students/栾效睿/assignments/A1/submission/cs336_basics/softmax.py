import torch
import torch.nn as nn


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_values = x.max(dim=dim, keepdim=True).values
    exp_values = torch.exp(x - max_values)
    sum_exp = torch.sum(exp_values, dim=dim, keepdim=True)
    return exp_values / sum_exp
