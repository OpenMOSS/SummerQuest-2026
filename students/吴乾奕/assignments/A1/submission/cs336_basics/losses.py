"""Numerically stable loss and normalization functions."""

from __future__ import annotations

import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """Apply a numerically stable softmax along ``dim``.

    Subtracting the maximum does not change the softmax, but prevents large
    logits from overflowing when exponentiated.
    """

    output_dtype = in_features.dtype
    working = in_features.float() if output_dtype in {torch.float16, torch.bfloat16} else in_features
    maxima = working.max(dim=dim, keepdim=True).values
    shifted = torch.where(torch.isfinite(maxima), working - maxima, working)
    exponentials = torch.exp(shifted)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    result = exponentials / torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return result.to(output_dtype)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy from unnormalized class logits.

    ``inputs`` may have any number of leading dimensions; the final dimension
    is interpreted as the class dimension and ``targets`` must match the
    leading shape.  The implementation uses the log-sum-exp identity rather
    than first forming probabilities, which remains stable for large logits.
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if inputs.shape[:-1] != targets.shape:
        raise ValueError(
            "targets must have the same shape as inputs excluding the class dimension; "
            f"got inputs.shape={tuple(inputs.shape)} and targets.shape={tuple(targets.shape)}"
        )
    if inputs.shape[-1] == 0:
        raise ValueError("the class dimension must be non-empty")

    # Loss reduction is kept in fp32 for fp16/bf16 training. This avoids a
    # visibly quantized validation metric on CPU and does not depend on a
    # backend's autocast allow-list.
    target_indices = targets.to(dtype=torch.long).unsqueeze(-1)
    if inputs.dtype in {torch.float16, torch.bfloat16} and inputs.shape[-1] > 4096:
        # Avoid materializing an fp32 copy of the entire (B, T, vocab) logits
        # tensor. Each chunk participates in autograd, while the reduction and
        # final metric remain fp32.
        max_logits = inputs.max(dim=-1, keepdim=True).values.float()
        exponential_sum = torch.zeros_like(max_logits.squeeze(-1))
        for chunk in inputs.split(4096, dim=-1):
            exponential_sum = exponential_sum + torch.exp(chunk.float() - max_logits).sum(dim=-1)
        log_normalizer = torch.log(exponential_sum)
        target_logits = inputs.gather(dim=-1, index=target_indices).squeeze(-1).float() - max_logits.squeeze(-1)
    else:
        working_inputs = inputs.float() if inputs.dtype in {torch.float16, torch.bfloat16} else inputs
        max_logits = working_inputs.max(dim=-1, keepdim=True).values
        shifted_logits = working_inputs - max_logits
        log_normalizer = torch.log(torch.exp(shifted_logits).sum(dim=-1))
        target_logits = shifted_logits.gather(dim=-1, index=target_indices).squeeze(-1)
    return (log_normalizer - target_logits).mean()
