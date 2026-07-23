import torch


def silu(x):
    return x * torch.sigmoid(x)