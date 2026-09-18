from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attention_forward_kernel(
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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
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

    running_max = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
    running_output = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for key_start in range(0, n_keys, BLOCK_N):
        key_offsets = key_start + offs_n
        k_mask = (key_offsets[:, None] < n_keys) & (
            offs_d[None, :] < head_dim
        )
        k_ptrs = (
            k_ptr
            + pid_b * stride_kb
            + key_offsets[:, None] * stride_kn
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        v_mask = (key_offsets[:, None] < n_keys) & (
            offs_d[None, :] < head_dim
        )
        v_ptrs = (
            v_ptr
            + pid_b * stride_vb
            + key_offsets[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(key_offsets[None, :] < n_keys, scores, float("-inf"))
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= key_offsets[None, :]
            scores = tl.where(causal_mask, scores, float("-inf"))

        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - new_max)
        probability = tl.exp(scores - new_max[:, None])

        running_sum = (
            old_scale * running_sum
            + tl.sum(probability, axis=1)
        )
        running_output = (
            old_scale[:, None] * running_output
            + tl.dot(probability, v)
        )
        running_max = new_max

    output = running_output / running_sum[:, None]
    lse = running_max + tl.log(running_sum)

    o_ptrs = (
        o_ptr
        + pid_b * stride_ob
        + offs_m[:, None] * stride_om
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, output, mask=q_mask)

    lse_ptrs = lse_ptr + pid_b * stride_lb + offs_m * stride_lm
    tl.store(lse_ptrs, lse, mask=offs_m < n_queries)


@triton.jit
def _flash_attention_backward_dq_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    do_ptr,
    dq_ptr,
    n_queries,
    n_keys,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    valid_m = offs_m < n_queries

    q_ptrs = q_ptr + (pid_b * n_queries + offs_m[:, None]) * head_dim + offs_d[None, :]
    do_ptrs = do_ptr + (pid_b * n_queries + offs_m[:, None]) * head_dim + offs_d[None, :]
    o_ptrs = o_ptr + (pid_b * n_queries + offs_m[:, None]) * head_dim + offs_d[None, :]
    row_mask = valid_m[:, None] & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    o = tl.load(o_ptrs, mask=row_mask, other=0.0).to(tl.float32)
    lse = tl.load(lse_ptr + pid_b * n_queries + offs_m, mask=valid_m, other=0.0)
    delta = tl.sum(do * o, axis=1)
    dq = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for key_start in range(0, n_keys, BLOCK_N):
        key_offsets = key_start + offs_n
        valid_n = key_offsets < n_keys
        k_ptrs = k_ptr + (pid_b * n_keys + key_offsets[:, None]) * head_dim + offs_d[None, :]
        v_ptrs = v_ptr + (pid_b * n_keys + key_offsets[:, None]) * head_dim + offs_d[None, :]
        k = tl.load(
            k_ptrs,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptrs,
            mask=valid_n[:, None] & (offs_d[None, :] < head_dim),
            other=0.0,
        ).to(tl.float32)

        scores = tl.dot(q, tl.trans(k)) * scale
        valid_scores = valid_m[:, None] & valid_n[None, :]
        scores = tl.where(valid_scores, scores, float("-inf"))
        if IS_CAUSAL:
            scores = tl.where(offs_m[:, None] >= key_offsets[None, :], scores, float("-inf"))

        p = tl.exp(scores - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds, k) * scale

    dq_ptrs = dq_ptr + (pid_b * n_queries + offs_m[:, None]) * head_dim + offs_d[None, :]
    tl.store(dq_ptrs, dq, mask=row_mask)


@triton.jit
def _flash_attention_backward_dkdv_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    lse_ptr,
    do_ptr,
    dk_ptr,
    dv_ptr,
    n_queries,
    n_keys,
    head_dim,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    valid_n = offs_n < n_keys

    k_ptrs = k_ptr + (pid_b * n_keys + offs_n[:, None]) * head_dim + offs_d[None, :]
    v_ptrs = v_ptr + (pid_b * n_keys + offs_n[:, None]) * head_dim + offs_d[None, :]
    key_mask = valid_n[:, None] & (offs_d[None, :] < head_dim)
    k = tl.load(k_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=key_mask, other=0.0).to(tl.float32)
    dk = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for query_start in range(0, n_queries, BLOCK_M):
        query_offsets = query_start + offs_m
        valid_m = query_offsets < n_queries
        q_ptrs = q_ptr + (pid_b * n_queries + query_offsets[:, None]) * head_dim + offs_d[None, :]
        do_ptrs = do_ptr + (pid_b * n_queries + query_offsets[:, None]) * head_dim + offs_d[None, :]
        o_ptrs = o_ptr + (pid_b * n_queries + query_offsets[:, None]) * head_dim + offs_d[None, :]
        row_mask = valid_m[:, None] & (offs_d[None, :] < head_dim)
        q = tl.load(q_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        o = tl.load(o_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        lse = tl.load(lse_ptr + pid_b * n_queries + query_offsets, mask=valid_m, other=0.0)
        delta = tl.sum(do * o, axis=1)

        scores = tl.dot(q, tl.trans(k)) * scale
        valid_scores = valid_m[:, None] & valid_n[None, :]
        scores = tl.where(valid_scores, scores, float("-inf"))
        if IS_CAUSAL:
            scores = tl.where(query_offsets[:, None] >= offs_n[None, :], scores, float("-inf"))

        p = tl.exp(scores - lse[:, None])
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds), q) * scale
        dv += tl.dot(tl.trans(p), do)

    dk_ptrs = dk_ptr + (pid_b * n_keys + offs_n[:, None]) * head_dim + offs_d[None, :]
    dv_ptrs = dv_ptr + (pid_b * n_keys + offs_n[:, None]) * head_dim + offs_d[None, :]
    tl.store(dk_ptrs, dk, mask=key_mask)
    tl.store(dv_ptrs, dv, mask=key_mask)


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 style forward backed by a student Triton kernel."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("FlashAttentionTriton requires CUDA tensors")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("q, k, and v must have shape [batch, sequence, head_dim]")
        if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
            raise ValueError("q, k, and v have incompatible batch/sequence shapes")
        if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
            raise ValueError("q, k, and v must have the same head dimension")

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch_size, n_queries, head_dim = q.shape
        n_keys = k.shape[1]
        output = torch.empty_like(q)
        lse = torch.empty(
            (batch_size, n_queries),
            dtype=torch.float32,
            device=q.device,
        )

        block_m = 32 if head_dim > 64 else 64
        block_n = 32 if head_dim > 64 else 64
        block_d = triton.next_power_of_2(head_dim)
        num_stages = 1 if head_dim > 64 else 2
        grid = (triton.cdiv(n_queries, block_m), batch_size)
        _flash_attention_forward_kernel[grid](
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            IS_CAUSAL=bool(is_causal),
            num_warps=4,
            num_stages=num_stages,
        )

        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        do = do.contiguous()
        batch_size, n_queries, head_dim = q.shape
        n_keys = k.shape[1]

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # A 32x32 backward tile is robust for all tested head dimensions.
        # In particular, it avoids the incorrect transpose-dot layout that
        # Triton 3.2 generated for the 64x64 dK/dV tile with BF16 operands.
        block_m = 32
        block_n = 32
        block_d = triton.next_power_of_2(head_dim)
        # Backward uses two matrix products per tile; one pipeline stage keeps
        # its shared-memory footprint below the RTX 4090 launch limit.
        num_stages = 1
        scale = 1.0 / math.sqrt(head_dim)

        dq_grid = (triton.cdiv(n_queries, block_m), batch_size)
        _flash_attention_backward_dq_kernel[dq_grid](
            q,
            k,
            v,
            output,
            lse,
            do,
            dq,
            n_queries,
            n_keys,
            head_dim,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            IS_CAUSAL=ctx.is_causal,
            num_warps=4,
            num_stages=num_stages,
        )

        dkdv_grid = (triton.cdiv(n_keys, block_n), batch_size)
        _flash_attention_backward_dkdv_kernel[dkdv_grid](
            q,
            k,
            v,
            output,
            lse,
            do,
            dk,
            dv,
            n_queries,
            n_keys,
            head_dim,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            IS_CAUSAL=ctx.is_causal,
            num_warps=4,
            num_stages=num_stages,
        )
        return dq, dk, dv, None
