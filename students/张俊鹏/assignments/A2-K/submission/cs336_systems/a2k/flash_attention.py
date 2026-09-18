from __future__ import annotations

import math

import torch

import triton
import triton.language as tl


def _apply_causal_mask(
    scores: torch.Tensor,
    query_start: int,
    key_start: int,
) -> torch.Tensor:
    """Mask keys that are to the right of each query position."""
    n_queries = scores.shape[-2]
    n_keys = scores.shape[-1]
    query_positions = torch.arange(
        query_start,
        query_start + n_queries,
        device=scores.device,
    )
    key_positions = torch.arange(
        key_start,
        key_start + n_keys,
        device=scores.device,
    )
    allowed = query_positions[:, None] >= key_positions[None, :]
    return scores.masked_fill(~allowed, float("-inf"))


def _tiled_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    query_block_size: int = 64,
    key_block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference FlashAttention-style forward using only PyTorch tensor ops."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [batch, sequence, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v have incompatible batch/sequence shapes")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")

    batch_size, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    accumulator_dtype = torch.float32

    query_outputs: list[torch.Tensor] = []
    query_lse: list[torch.Tensor] = []

    q_acc = q.to(accumulator_dtype)
    k_acc = k.to(accumulator_dtype)
    v_acc = v.to(accumulator_dtype)

    for query_start in range(0, n_queries, query_block_size):
        query_end = min(query_start + query_block_size, n_queries)
        q_block = q_acc[:, query_start:query_end, :]
        block_queries = query_end - query_start

        running_max = torch.full(
            (batch_size, block_queries),
            float("-inf"),
            dtype=accumulator_dtype,
            device=q.device,
        )
        running_sum = torch.zeros_like(running_max)
        running_output = torch.zeros(
            (batch_size, block_queries, head_dim),
            dtype=accumulator_dtype,
            device=q.device,
        )

        for key_start in range(0, n_keys, key_block_size):
            key_end = min(key_start + key_block_size, n_keys)
            k_block = k_acc[:, key_start:key_end, :]
            v_block = v_acc[:, key_start:key_end, :]

            scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
            if is_causal:
                scores = _apply_causal_mask(scores, query_start, key_start)

            block_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            probability = torch.exp(scores - new_max.unsqueeze(-1))

            running_sum = old_scale * running_sum + probability.sum(dim=-1)
            running_output = (
                old_scale.unsqueeze(-1) * running_output
                + torch.matmul(probability, v_block)
            )
            running_max = new_max

        output_block = running_output / running_sum.unsqueeze(-1)
        lse_block = running_max + torch.log(running_sum)
        query_outputs.append(output_block)
        query_lse.append(lse_block)

    output = torch.cat(query_outputs, dim=1).to(dtype=q.dtype)
    lse = torch.cat(query_lse, dim=1)
    return output, lse


class FlashAttentionPytorch(torch.autograd.Function):
    """Tiled FlashAttention-style autograd function implemented in PyTorch."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = _tiled_attention_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, _output, _lse = ctx.saved_tensors

        # Recompute the tiled forward under autograd, then use the incoming
        # output gradient to obtain dQ, dK and dV. This keeps the reference
        # path explicit and avoids calling a fused attention primitive.
        with torch.enable_grad():
            q_recompute = q.detach().requires_grad_(True)
            k_recompute = k.detach().requires_grad_(True)
            v_recompute = v.detach().requires_grad_(True)
            output_recompute, _ = _tiled_attention_forward(
                q_recompute,
                k_recompute,
                v_recompute,
                ctx.is_causal,
            )
            dq, dk, dv = torch.autograd.grad(
                output_recompute,
                (q_recompute, k_recompute, v_recompute),
                do,
                allow_unused=False,
            )

        return dq, dk, dv, None


import triton
import triton.language as tl


@triton.jit
def _flash_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    stride_qb,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_om,
    stride_od,
    stride_lb,
    stride_lm,
    n_queries,
    n_keys,
    head_dim,
    scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < n_queries) & (offs_d[None, :] < head_dim)
    q_ptrs = (
        q_ptr
        + pid_b * stride_qb
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    running_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_output = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for key_start in range(0, n_keys, BLOCK_N):
        k_mask = (key_start + offs_n[:, None] < n_keys) & (
            offs_d[None, :] < head_dim
        )
        k_ptrs = (
            k_ptr
            + pid_b * stride_kb
            + (key_start + offs_n[:, None]) * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        v_mask = (key_start + offs_n[:, None] < n_keys) & (
            offs_d[None, :] < head_dim
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_vb
            + (key_start + offs_n[:, None]) * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        valid_keys = key_start + offs_n < n_keys
        scores = tl.where(valid_keys[None, :], scores, -float("inf"))
        if IS_CAUSAL:
            causal = offs_m[:, None] >= (key_start + offs_n[None, :])
            scores = tl.where(causal, scores, -float("inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probability = tl.exp(scores - new_max[:, None])

        running_sum = old_scale * running_sum + tl.sum(probability, axis=1)
        running_output = (
            old_scale[:, None] * running_output
            + tl.dot(probability, v)
        )
        running_max = new_max

    output = running_output / running_sum[:, None]
    lse = running_max + tl.log(running_sum)

    o_mask = (offs_m[:, None] < n_queries) & (offs_d[None, :] < head_dim)
    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, output, mask=o_mask)

    lse_ptrs = lse_ptr + pid_b * stride_lb + offs_m * stride_lm
    tl.store(lse_ptrs, lse, mask=offs_m < n_queries)


def _triton_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Triton FlashAttention requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [batch, sequence, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v have incompatible batch/sequence shapes")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("Triton path supports float16, bfloat16, and float32")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch_size, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        raise ValueError("head_dim must be <= 256")

    output = torch.empty_like(q)
    lse = torch.empty(
        (batch_size, n_queries),
        dtype=torch.float32,
        device=q.device,
    )
    grid = (triton.cdiv(n_queries, 64), batch_size)
    _flash_forward_kernel[grid](
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
        head_dim,
        1.0 / math.sqrt(head_dim),
        IS_CAUSAL=bool(is_causal),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return output, lse


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention forward in Triton with temporary PyTorch recompute backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = _triton_attention_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, _output, _lse = ctx.saved_tensors
        with torch.enable_grad():
            q_recompute = q.detach().requires_grad_(True)
            k_recompute = k.detach().requires_grad_(True)
            v_recompute = v.detach().requires_grad_(True)
            output_recompute, _ = _tiled_attention_forward(
                q_recompute,
                k_recompute,
                v_recompute,
                ctx.is_causal,
            )
            dq, dk, dv = torch.autograd.grad(
                output_recompute,
                (q_recompute, k_recompute, v_recompute),
                do,
                allow_unused=False,
            )
        return dq, dk, dv, None
