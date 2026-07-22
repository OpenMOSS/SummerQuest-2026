"""Explicit and tiled attention implementations used by A2-K.

The reference implementation intentionally keeps the algorithm visible.  The
Triton forward path fuses the score, online softmax, and value reduction; its
backward path uses the assignment's recomputation equations in PyTorch.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

triton: Any = None
tl: Any = None
try:  # Triton is optional on CPU-only development environments.
    import triton as _triton
    import triton.language as _tl

    triton = _triton
    tl = _tl
except ImportError:  # pragma: no cover - exercised only without Triton.
    triton = None
    tl = None


def _causal_scores(
    scores: Tensor,
    query_start: int,
    key_start: int,
    is_causal: bool,
) -> Tensor:
    if not is_causal:
        return scores
    q_idx = torch.arange(
        query_start, query_start + scores.shape[-2], device=scores.device
    )
    k_idx = torch.arange(
        key_start, key_start + scores.shape[-1], device=scores.device
    )
    mask = q_idx[:, None] >= k_idx[None, :]
    return scores.masked_fill(~mask, -1e6)


def explicit_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
) -> Tensor:
    """Materialized PyTorch attention baseline.

    The function deliberately spells out ``QK^T -> mask -> softmax -> PV`` and
    never dispatches to ``scaled_dot_product_attention``.
    """

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    scores = _causal_scores(scores, 0, 0, is_causal)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def triton_launch_config(
    n_queries: int,
    n_keys: int,
    head_dim: int,
) -> dict[str, int]:
    """Return the launch parameters shared by the kernel and benchmarks."""

    tile_size = 16 if head_dim >= 128 else 64

    def bounded_tile(length: int) -> int:
        if length >= tile_size:
            return tile_size
        return max(16, 1 << max(0, length - 1).bit_length())

    return {
        "q_tile_size": bounded_tile(n_queries),
        "k_tile_size": bounded_tile(n_keys),
        "num_warps": 2 if head_dim >= 128 else 4,
        "num_stages": 2,
    }


class ExplicitAttention(torch.nn.Module):
    """``nn.Module`` wrapper useful for eager/compiled benchmarks."""

    def __init__(self, is_causal: bool = True):
        super().__init__()
        self.is_causal = is_causal

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return explicit_attention(q, k, v, self.is_causal)


def _online_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    q_tile_size: int = 64,
    k_tile_size: int = 64,
) -> tuple[Tensor, Tensor]:
    """Pure PyTorch tiled FlashAttention forward.

    Returns the output and the row-wise log-sum-exp.  All running reductions
    use FP32, while the output follows the input dtype.
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [batch, sequence, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("incompatible batch/sequence shapes")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")

    batch, n_queries, dim = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(dim)
    output = torch.zeros_like(q)
    lse = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)

    # Keeping the outer query loop explicit makes the memory bound clear:
    # no [batch, queries, keys] tensor survives a tile iteration.
    for q_start in range(0, n_queries, q_tile_size):
        q_stop = min(q_start + q_tile_size, n_queries)
        q_tile = q[:, q_start:q_stop].float()
        rows = q_stop - q_start
        acc = torch.zeros((batch, rows, dim), device=q.device, dtype=torch.float32)
        running_max = torch.full(
            (batch, rows), -torch.inf, device=q.device, dtype=torch.float32
        )
        running_sum = torch.zeros(
            (batch, rows), device=q.device, dtype=torch.float32
        )

        for k_start in range(0, n_keys, k_tile_size):
            k_stop = min(k_start + k_tile_size, n_keys)
            k_tile = k[:, k_start:k_stop].float()
            v_tile = v[:, k_start:k_stop].float()
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            scores = _causal_scores(scores, q_start, k_start, is_causal)

            tile_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, tile_max)
            old_scale = torch.exp(running_max - new_max)
            numerators = torch.exp(scores - new_max.unsqueeze(-1))
            new_sum = old_scale * running_sum + numerators.sum(dim=-1)
            acc = (
                acc * old_scale.unsqueeze(-1)
                + torch.matmul(numerators, v_tile)
            )
            running_max = new_max
            running_sum = new_sum

        output[:, q_start:q_stop] = (acc / running_sum.unsqueeze(-1)).to(q.dtype)
        lse[:, q_start:q_stop] = running_max + torch.log(running_sum)

    return output, lse


def _recompute_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    output: Tensor,
    d_output: Tensor,
    lse: Tensor,
    is_causal: bool,
    q_tile_size: int = 256,
    k_tile_size: int = 256,
) -> tuple[Tensor, Tensor, Tensor]:
    """Recompute tiled probabilities and apply Equations 13--19."""

    batch, n_queries, dim = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(dim)
    qf, kf, vf = q.float(), k.float(), v.float()
    dof, of = d_output.float(), output.float()
    lf = lse.float()
    d = (of * dof).sum(dim=-1)
    dq = torch.zeros_like(qf)
    dk = torch.zeros_like(kf)
    dv = torch.zeros_like(vf)

    for q_start in range(0, n_queries, q_tile_size):
        q_stop = min(q_start + q_tile_size, n_queries)
        q_tile = qf[:, q_start:q_stop]
        do_tile = dof[:, q_start:q_stop]
        l_tile = lf[:, q_start:q_stop]
        d_tile = d[:, q_start:q_stop]

        for k_start in range(0, n_keys, k_tile_size):
            k_stop = min(k_start + k_tile_size, n_keys)
            k_tile = kf[:, k_start:k_stop]
            v_tile = vf[:, k_start:k_stop]
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            scores = _causal_scores(scores, q_start, k_start, is_causal)
            probabilities = torch.exp(scores - l_tile.unsqueeze(-1))

            d_v_tile = torch.matmul(probabilities.transpose(-1, -2), do_tile)
            d_p = torch.matmul(do_tile, v_tile.transpose(-1, -2))
            d_s = probabilities * (d_p - d_tile.unsqueeze(-1))
            d_q_tile = torch.matmul(d_s, k_tile) * scale
            d_k_tile = torch.matmul(d_s.transpose(-1, -2), q_tile) * scale

            dq[:, q_start:q_stop] += d_q_tile
            dk[:, k_start:k_stop] += d_k_tile
            dv[:, k_start:k_stop] += d_v_tile

    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttentionPytorch(torch.autograd.Function):
    """Autograd wrapper around the tiled PyTorch implementation."""

    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        output, lse = _online_attention(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: Any,
        d_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _recompute_backward(
            q, k, v, output, d_output, lse, ctx.is_causal
        )
        return dq, dk, dv, None


if triton is not None:

    @triton.jit  # type: ignore[union-attr]
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        n_queries,
        n_keys,
        scale,
        d: tl.constexpr,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """One program owns one query tile and loops over key tiles."""

        query_tile = tl.program_id(0)
        batch_index = tl.program_id(1)
        q_offsets = query_tile * q_tile_size + tl.arange(0, q_tile_size)
        d_offsets = tl.arange(0, d)
        q_mask = q_offsets < n_queries
        q_ptrs = (
            q_ptr
            + batch_index * stride_qb
            + q_offsets[:, None] * stride_qq
            + d_offsets[None, :] * stride_qd
        )
        q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0)
        acc = tl.zeros((q_tile_size, d), dtype=tl.float32)
        running_max = tl.full((q_tile_size,), -float("inf"), dtype=tl.float32)
        running_sum = tl.zeros((q_tile_size,), dtype=tl.float32)

        for key_start in tl.range(0, n_keys, k_tile_size):
            k_offsets = key_start + tl.arange(0, k_tile_size)
            k_mask = k_offsets < n_keys
            k_ptrs = (
                k_ptr
                + batch_index * stride_kb
                + k_offsets[:, None] * stride_kk
                + d_offsets[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + batch_index * stride_vb
                + k_offsets[:, None] * stride_vk
                + d_offsets[None, :] * stride_vd
            )
            k_tile = tl.load(k_ptrs, mask=k_mask[:, None], other=0.0)
            v_tile = tl.load(v_ptrs, mask=k_mask[:, None], other=0.0)
            scores = tl.dot(
                q_tile,
                tl.trans(k_tile),
                input_precision="ieee",
                out_dtype=tl.float32,
            ) * scale
            valid = q_mask[:, None] & k_mask[None, :]
            if is_causal:
                valid = valid & (q_offsets[:, None] >= k_offsets[None, :])
            scores = tl.where(valid, scores, -1.0e6)

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            old_scale = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            new_sum = old_scale * running_sum + tl.sum(probabilities, axis=1)
            acc = old_scale[:, None] * acc + tl.dot(
                probabilities.to(v_tile.dtype),
                v_tile,
                acc=tl.zeros((q_tile_size, d), dtype=tl.float32),
                out_dtype=tl.float32,
            )
            running_max = new_max
            running_sum = new_sum

        output = acc / running_sum[:, None]
        logsumexp = running_max + tl.log(running_sum)
        o_ptrs = (
            o_ptr
            + batch_index * stride_ob
            + q_offsets[:, None] * stride_oq
            + d_offsets[None, :] * stride_od
        )
        l_ptrs = l_ptr + batch_index * stride_lb + q_offsets * stride_lq
        tl.store(o_ptrs, output, mask=q_mask[:, None])
        tl.store(l_ptrs, logsumexp, mask=q_mask)


def _triton_forward(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> tuple[Tensor, Tensor]:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not q.is_cuda:
        return _online_attention(q, k, v, is_causal)
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    batch, n_queries, dim = q.shape
    n_keys = k.shape[1]
    launch = triton_launch_config(n_queries, n_keys, dim)
    q_tile_size = launch["q_tile_size"]
    k_tile_size = launch["k_tile_size"]
    output = torch.empty_like(q)
    lse = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(n_queries, q_tile_size), batch)
    _flash_fwd_kernel[grid](  # ty: ignore[unknown-argument, invalid-argument-type]
        q,
        k,
        v,
        output,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        n_queries,
        n_keys,
        1.0 / math.sqrt(dim),
        d=dim,
        q_tile_size=q_tile_size,
        k_tile_size=k_tile_size,
        is_causal=is_causal,
        num_warps=launch["num_warps"],
        num_stages=launch["num_stages"],
    )
    return output, lse


class FlashAttentionTriton(torch.autograd.Function):
    """Triton-forward/PyTorch-recompute-backward FlashAttention path."""

    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        is_causal: bool = False,
    ) -> Tensor:
        output, lse = _triton_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(  # ty: ignore[invalid-method-override]
        ctx: Any,
        d_output: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _recompute_backward(
            q, k, v, output, d_output, lse, ctx.is_causal
        )
        return dq, dk, dv, None


def flash_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    implementation: str = "pytorch",
) -> Tensor:
    """Convenience dispatcher used by the student benchmark scripts."""

    if implementation == "pytorch":
        return FlashAttentionPytorch.apply(q, k, v, is_causal)
    if implementation == "triton":
        return FlashAttentionTriton.apply(q, k, v, is_causal)
    raise ValueError(f"unknown attention implementation: {implementation}")


__all__ = [
    "ExplicitAttention",
    "FlashAttentionPytorch",
    "FlashAttentionTriton",
    "explicit_attention",
    "flash_attention",
    "triton_launch_config",
]
