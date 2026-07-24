from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    scores = torch.einsum("...qd,...kd->...qk", q, k) / math.sqrt(d)
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        scores = torch.where(mask, scores, torch.tensor(float("-inf"), device=q.device, dtype=scores.dtype))
    lse = torch.logsumexp(scores.float(), dim=-1)
    probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    out = torch.einsum("...qk,...kd->...qd", probs, v)
    return out, lse


def _recompute_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, grad_out: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out, _ = explicit_attention(q_, k_, v_, is_causal)
        dq, dk, dv = torch.autograd.grad(out, (q_, k_, v_), grad_out, retain_graph=False, create_graph=False)
    return dq, dk, dv


class FlashAttentionPytorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        out, lse = explicit_attention(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, _out, _lse = ctx.saved_tensors
        dq, dk, dv = _recompute_backward(q, k, v, grad_out, ctx.is_causal)
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lq: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        d_head: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        batch = tl.program_id(0)
        q_block = tl.program_id(1)
        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + batch * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_head),
            other=0.0,
        ).to(tl.float32)

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, n_keys, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                k_ptr + batch * stride_kb + offs_n[:, None] * stride_kk + offs_d[None, :] * stride_kd,
                mask=(offs_n[:, None] < n_keys) & (offs_d[None, :] < d_head),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr + batch * stride_vb + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=(offs_n[:, None] < n_keys) & (offs_d[None, :] < d_head),
                other=0.0,
            ).to(tl.float32)

            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(offs_n[None, :] < n_keys, scores, -float("inf"))
            if is_causal:
                scores = tl.where(offs_m[:, None] >= offs_n[None, :], scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_ij = tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + l_ij
            m_i = m_ij

        out = acc / l_i[:, None]
        tl.store(
            o_ptr + batch * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            out,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < d_head),
        )
        tl.store(
            lse_ptr + batch * stride_lb + offs_m * stride_lq,
            m_i + tl.log(l_i),
            mask=offs_m < n_queries,
        )


def _triton_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("triton is not installed")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        return explicit_attention(q, k, v, is_causal)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected q, k, v with shape [batch, sequence, d_head]")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, n_queries, d_head = q.shape
    _, n_keys, d_value = v.shape
    if d_head != d_value:
        raise ValueError("this implementation expects d_head == d_value")
    block_d = triton.next_power_of_2(d_head)
    if block_d > 128:
        raise ValueError("d_head > 128 is not supported")
    out = torch.empty_like(q)
    lse = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
    block_m = 8 if d_head >= 128 else 16
    block_n = 32 if d_head >= 128 else 64
    grid = (batch, triton.cdiv(n_queries, block_m))
    _flash_forward_kernel[grid](
        q,
        k,
        v,
        out,
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
        out.stride(0),
        out.stride(1),
        out.stride(2),
        lse.stride(0),
        lse.stride(1),
        n_queries,
        n_keys,
        d_head,
        1.0 / math.sqrt(d_head),
        bool(is_causal),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=3,
    )
    return out, lse


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        out, lse = _triton_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, _out, _lse = ctx.saved_tensors
        dq, dk, dv = _recompute_backward(q, k, v, grad_out, ctx.is_causal)
        return dq, dk, dv, None
