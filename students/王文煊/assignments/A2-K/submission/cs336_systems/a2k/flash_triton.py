"""Student-written FlashAttention-2 forward Triton kernel + recompute backward.

The @triton.jit forward kernel assigns one program instance per query tile,
loops over key/value tiles inside the kernel, keeps the accumulator and the
online-softmax state (running max m_i, running sum l_i) in FP32, applies the
causal mask when requested, and writes O plus the log-sum-exp L.

Backward recomputes dQ, dK, dV in plain PyTorch from the saved Q, K, V, O, L
(allowed by the assignment), and is shared with the pure-PyTorch path.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from .flash_pytorch import flash_backward_recompute


@triton.jit
def _flash_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    stride_lb, stride_lm,
    N_QUERIES, N_KEYS,
    SCALE,
    IS_CAUSAL: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # query tile index
    pid_b = tl.program_id(1)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    m_mask = offs_m < N_QUERIES

    q_ptrs = q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                # running sum
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)             # output accumulator

    if IS_CAUSAL:
        hi = tl.minimum((pid_m + 1) * BLOCK_M, N_KEYS)
    else:
        hi = N_KEYS

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N_KEYS
        k_ptrs = k_ptr + pid_b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * SCALE  # [BLOCK_M, BLOCK_N], fp32
        s = tl.where(n_mask[None, :], s, float("-inf"))
        if IS_CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # Guard against -inf running max (fully masked first tile in causal mode).
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + pid_b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    o = acc / l_safe[:, None]

    o_ptrs = o_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, o.to(o_ptr.dtype.element_ty), mask=m_mask[:, None])

    lse = m_i + tl.log(l_safe)
    l_ptrs = l_ptr + pid_b * stride_lb + offs_m * stride_lm
    tl.store(l_ptrs, lse, mask=m_mask)


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 with a student-written Triton forward kernel."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        batch_shape = q.shape[:-2]
        nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
        assert d in (16, 32, 64, 128), f"head dim must be a power of two in [16,128], got {d}"
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        batch = 1
        for s in batch_shape:
            batch *= s
        q2 = q.reshape(batch, nq, d)
        k2 = k.reshape(batch, nk, d)
        v2 = v.reshape(batch, nk, d)

        o = torch.empty_like(q2)
        lse = torch.empty(batch, nq, device=q.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        grid = (triton.cdiv(nq, BLOCK_M), batch)
        _flash_fwd_kernel[grid](
            q2, k2, v2, o, lse,
            q2.stride(0), q2.stride(1), q2.stride(2),
            k2.stride(0), k2.stride(1), k2.stride(2),
            v2.stride(0), v2.stride(1), v2.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1),
            nq, nk,
            1.0 / math.sqrt(d),
            IS_CAUSAL=is_causal,
            D=d,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        o = o.reshape(batch_shape + (nq, d))
        lse = lse.reshape(batch_shape + (nq,))
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_backward_recompute(q, k, v, o, lse, do, ctx.is_causal)
        return dq, dk, dv, None
