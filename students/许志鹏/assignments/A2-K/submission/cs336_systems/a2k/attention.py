from __future__ import annotations

import math
from collections.abc import Callable

import torch


def _validate_attention_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim < 3 or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("Q, K, and V must have matching rank >= 3")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("Q, K, and V must have matching batch dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K must have the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K and V must have the same sequence length")
    if q.shape[-1] != v.shape[-1]:
        raise ValueError("A2-K expects Q, K, and V to use the same head dimension")


def causal_mask(n_queries: int, n_keys: int, device: torch.device) -> torch.Tensor:
    query_indices = torch.arange(n_queries, device=device)
    key_indices = torch.arange(n_keys, device=device)
    return query_indices[:, None] >= key_indices[None, :]


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Explicit QK^T -> mask -> softmax -> PV attention baseline."""
    _validate_attention_inputs(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        scores = scores.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def explicit_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_attention_inputs(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        scores = scores.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), -1e6)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v), torch.logsumexp(scores, dim=-1)


def tiled_flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    query_tile_size: int = 64,
    key_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch tiled FlashAttention-2 forward with online softmax."""
    _validate_attention_inputs(q, k, v)
    if query_tile_size < 16 or key_tile_size < 16:
        raise ValueError("A2-K tile sizes must be at least 16")

    leading_shape = q.shape[:-2]
    n_queries, head_dim = q.shape[-2:]
    n_keys = k.shape[-2]
    flat_batch = math.prod(leading_shape)
    q_flat = q.reshape(flat_batch, n_queries, head_dim)
    k_flat = k.reshape(flat_batch, n_keys, head_dim)
    v_flat = v.reshape(flat_batch, n_keys, head_dim)
    output = torch.empty_like(q_flat)
    lse = torch.empty((flat_batch, n_queries), device=q.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_dim)

    for query_start in range(0, n_queries, query_tile_size):
        query_end = min(query_start + query_tile_size, n_queries)
        q_tile = q_flat[:, query_start:query_end].float()
        rows = query_end - query_start
        running_max = torch.full((flat_batch, rows), float("-inf"), device=q.device, dtype=torch.float32)
        running_sum = torch.zeros((flat_batch, rows), device=q.device, dtype=torch.float32)
        accumulator = torch.zeros((flat_batch, rows, head_dim), device=q.device, dtype=torch.float32)

        for key_start in range(0, n_keys, key_tile_size):
            key_end = min(key_start + key_tile_size, n_keys)
            k_tile = k_flat[:, key_start:key_end].float()
            v_tile = v_flat[:, key_start:key_end].float()
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale
            if is_causal:
                query_indices = torch.arange(query_start, query_end, device=q.device)
                key_indices = torch.arange(key_start, key_end, device=q.device)
                scores = scores.masked_fill(query_indices[:, None] < key_indices[None, :], -1e6)

            tile_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, tile_max)
            correction = torch.exp(running_max - new_max)
            unnormalized = torch.exp(scores - new_max.unsqueeze(-1))
            running_sum = correction * running_sum + unnormalized.sum(dim=-1)
            accumulator = correction.unsqueeze(-1) * accumulator + torch.matmul(unnormalized, v_tile)
            running_max = new_max

        output[:, query_start:query_end] = (accumulator / running_sum.unsqueeze(-1)).to(q.dtype)
        lse[:, query_start:query_end] = running_max + torch.log(running_sum)

    return output.reshape(*leading_shape, n_queries, head_dim), lse.reshape(*leading_shape, n_queries)


def flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations 13-19: recompute probabilities from Q, K, and saved LSE."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    grad_output_float = grad_output.float()

    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    probabilities = torch.exp(scores - lse.float().unsqueeze(-1))
    if is_causal:
        probabilities = probabilities.masked_fill(~causal_mask(q.shape[-2], k.shape[-2], q.device), 0.0)

    d_vector = (output.float() * grad_output_float).sum(dim=-1)
    grad_probabilities = torch.matmul(grad_output_float, v_float.transpose(-2, -1))
    grad_scores = probabilities * (grad_probabilities - d_vector.unsqueeze(-1))
    grad_q = torch.matmul(grad_scores, k_float) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q_float) * scale
    grad_v = torch.matmul(probabilities.transpose(-2, -1), grad_output_float)
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


_compiled_backward: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None


def compiled_flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global _compiled_backward
    if not q.is_cuda:
        return flash_attention_backward(q, k, v, output, grad_output, lse, is_causal)
    if _compiled_backward is None:
        _compiled_backward = torch.compile(flash_attention_backward, fullgraph=True)
    return _compiled_backward(q, k, v, output, grad_output, lse, is_causal)


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = tiled_flash_attention_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = compiled_flash_attention_backward(
            q,
            k,
            v,
            output,
            grad_output,
            lse,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None
