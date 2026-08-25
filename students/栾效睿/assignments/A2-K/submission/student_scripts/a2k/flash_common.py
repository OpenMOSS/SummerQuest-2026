from __future__ import annotations

import torch


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    scores = q.float() @ k.float().transpose(-2, -1) * (q.shape[-1] ** -0.5)
    if is_causal:
        scores = scores.masked_fill(torch.arange(q.shape[1], device=q.device)[:, None] < torch.arange(k.shape[1], device=k.device)[None, :], -1_000_000.0)
    probabilities = scores.softmax(dim=-1)
    return (probabilities @ v.float()).to(q.dtype), scores.logsumexp(dim=-1)


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    return {"max_abs": float(difference.max()), "max_rel": float((difference / expected.float().abs().clamp_min(1e-6)).max())}


def make_attention_inputs(seed: int, sequence_length: int, head_dim: int, dtype: torch.dtype, device: str = "cuda") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    shape = (1, sequence_length, head_dim)
    return (*(torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for _ in range(3)), torch.randn(shape, device=device, dtype=dtype))


def causal_mask(sequence_length: int, device: str = "cuda") -> torch.Tensor:
    return torch.ones((sequence_length, sequence_length), device=device, dtype=torch.bool).tril_()
