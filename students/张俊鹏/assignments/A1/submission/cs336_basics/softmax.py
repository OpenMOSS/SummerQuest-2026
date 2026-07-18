import torch

"""
细节保证数值稳定性
"""

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    x_max = torch.max(x, dim=dim, keepdim=True).values

    x_shifted = x - x_max
    x_exp = torch.exp(x_shifted)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)

    probs = x_exp / x_exp_sum
    
    return probs