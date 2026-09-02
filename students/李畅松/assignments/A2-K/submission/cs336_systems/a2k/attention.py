"""Tiled PyTorch reference and Triton FlashAttention implementation."""

from __future__ import annotations

import math

import torch


def _causal_mask(scores: torch.Tensor, q0: int, k0: int, causal: bool) -> torch.Tensor:
    if not causal:
        return scores
    qi = torch.arange(q0, q0 + scores.shape[-2], device=scores.device)[:, None]
    ki = torch.arange(k0, k0 + scores.shape[-1], device=scores.device)[None, :]
    return scores.masked_fill(ki > qi, float("-inf"))


def tiled_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool, block: int = 64):
    scale = q.shape[-1] ** -0.5
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    for q0 in range(0, q.shape[-2], block):
        q1 = min(q0 + block, q.shape[-2])
        qb = q[..., q0:q1, :]
        m = torch.full(qb.shape[:-1], -math.inf, dtype=torch.float32, device=q.device)
        z = torch.zeros_like(m)
        acc = torch.zeros_like(qb, dtype=torch.float32)
        for k0 in range(0, k.shape[-2], block):
            k1 = min(k0 + block, k.shape[-2])
            scores = qb.float() @ k[..., k0:k1, :].float().transpose(-1, -2)
            scores = _causal_mask(scores * scale, q0, k0, causal)
            new_m = torch.maximum(m, scores.amax(-1))
            alpha = torch.exp(m - new_m)
            probs = torch.exp(scores - new_m.unsqueeze(-1))
            z = alpha * z + probs.sum(-1)
            acc = alpha.unsqueeze(-1) * acc + probs @ v[..., k0:k1, :].float()
            m = new_m
        out[..., q0:q1, :] = (acc / z.unsqueeze(-1)).to(q.dtype)
        lse[..., q0:q1] = m + torch.log(z)
    return out, lse


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        out, lse = tiled_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        scale = q.shape[-1] ** -0.5
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        delta = (grad_out.float() * out.float()).sum(-1)
        for q0 in range(0, q.shape[-2], 64):
            q1 = min(q0 + 64, q.shape[-2])
            qb, dob = q[..., q0:q1, :], grad_out[..., q0:q1, :]
            for k0 in range(0, k.shape[-2], 64):
                k1 = min(k0 + 64, k.shape[-2])
                kb, vb = k[..., k0:k1, :], v[..., k0:k1, :]
                scores = _causal_mask(qb.float() @ kb.float().transpose(-1, -2) * scale, q0, k0, ctx.is_causal)
                p = torch.exp(scores - lse[..., q0:q1].unsqueeze(-1))
                dv[..., k0:k1, :] += (p.transpose(-1, -2) @ dob.float()).to(dv.dtype)
                dp = dob.float() @ vb.float().transpose(-1, -2)
                ds = p * (dp - delta[..., q0:q1].unsqueeze(-1))
                dq[..., q0:q1, :] += (ds @ kb.float() * scale).to(dq.dtype)
                dk[..., k0:k1, :] += (ds.transpose(-1, -2) @ qb.float() * scale).to(dk.dtype)
        return dq, dk, dv, None


try:
    import triton
    import triton.language as tl
except ImportError:  # Importing the CPU reference must not require Triton.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _forward_kernel(q, k, v, out, lse, stride_b, n: tl.constexpr, d: tl.constexpr, causal: tl.constexpr, block_n: tl.constexpr):
        row = tl.program_id(0)
        batch = row // n
        qi = row - batch * n
        offs_d = tl.arange(0, d)
        qv = tl.load(q + batch * stride_b + qi * d + offs_d).to(tl.float32)
        m = -float("inf")
        z = 0.0
        acc = tl.zeros((d,), tl.float32)
        for k0 in range(0, n, block_n):
            offs_n = k0 + tl.arange(0, block_n)
            mask_n = offs_n < n
            matrix_offsets = offs_n[:, None] * d + offs_d[None, :]
            matrix_mask = mask_n[:, None]
            kval = tl.load(k + batch * stride_b + matrix_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
            scores = tl.sum(kval * qv[None, :], axis=1)
            scores *= 1.0 / tl.sqrt(float(d))
            scores = tl.where(mask_n & ((not causal) | (offs_n <= qi)), scores, -float("inf"))
            new_m = tl.maximum(m, tl.max(scores, axis=0))
            alpha = tl.exp(m - new_m)
            p = tl.exp(scores - new_m)
            new_z = alpha * z + tl.sum(p, axis=0)
            vv = tl.load(v + batch * stride_b + matrix_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
            acc = alpha * acc + tl.sum(p[:, None] * vv, axis=0)
            m, z = new_m, new_z
        tl.store(out + batch * stride_b + qi * d + offs_d, acc / z)
        tl.store(lse + batch * n + qi, m + tl.log(z))

    @triton.jit
    def _backward_kernel(q, k, v, out, dout, lse, dq, dk, dv, stride_b, n: tl.constexpr, d: tl.constexpr, causal: tl.constexpr, block_n: tl.constexpr):
        row = tl.program_id(0)
        batch = row // n
        qi = row - batch * n
        offs_d = tl.arange(0, d)
        qv = tl.load(q + batch * stride_b + qi * d + offs_d).to(tl.float32)
        ov = tl.load(out + batch * stride_b + qi * d + offs_d).to(tl.float32)
        dov = tl.load(dout + batch * stride_b + qi * d + offs_d).to(tl.float32)
        l = tl.load(lse + batch * n + qi)
        delta = tl.sum(ov * dov, axis=0)
        dq_acc = tl.zeros((d,), tl.float32)
        for k0 in range(0, n, block_n):
            offs_n = k0 + tl.arange(0, block_n)
            mask_n = offs_n < n
            matrix_offsets = offs_n[:, None] * d + offs_d[None, :]
            matrix_mask = mask_n[:, None]
            kval = tl.load(k + batch * stride_b + matrix_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
            vval = tl.load(v + batch * stride_b + matrix_offsets, mask=matrix_mask, other=0.0).to(tl.float32)
            score = tl.sum(kval * qv[None, :], axis=1)
            dp = tl.sum(vval * dov[None, :], axis=1)
            score *= 1.0 / tl.sqrt(float(d))
            valid = mask_n & ((not causal) | (offs_n <= qi))
            p = tl.where(valid, tl.exp(score - l), 0.0)
            ds = p * (dp - delta) / tl.sqrt(float(d))
            dq_acc += tl.sum(ds[:, None] * kval, axis=0)
            tl.atomic_add(dk + batch * stride_b + matrix_offsets, ds[:, None] * qv[None, :], mask=valid[:, None])
            tl.atomic_add(dv + batch * stride_b + matrix_offsets, p[:, None] * dov[None, :], mask=valid[:, None])
        tl.store(dq + batch * stride_b + qi * d + offs_d, dq_acc)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise ValueError("FlashAttentionTriton requires CUDA tensors")
        if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
            raise ValueError("expected q, k, v with identical [batch, sequence, head_dim] shapes")
        if q.shape[-1] not in (32, 64, 128):
            raise ValueError("supported head dimensions are 32, 64, and 128")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        b, n, d = q.shape
        out = torch.empty_like(q)
        lse = torch.empty((b, n), device=q.device, dtype=torch.float32)
        _forward_kernel[(b * n,)](q, k, v, out, lse, q.stride(0), n=n, d=d, causal=bool(is_causal), block_n=64, num_warps=4)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        b, n, d = q.shape
        grad_out = grad_out.contiguous()
        # FP32 accumulation is required because many query programs update dK/dV.
        dq32 = torch.empty_like(q, dtype=torch.float32)
        dk32 = torch.zeros_like(k, dtype=torch.float32)
        dv32 = torch.zeros_like(v, dtype=torch.float32)
        _backward_kernel[(b * n,)](q, k, v, out, grad_out, lse, dq32, dk32, dv32, q.stride(0), n=n, d=d, causal=ctx.is_causal, block_n=64, num_warps=4)
        return dq32.to(q.dtype), dk32.to(k.dtype), dv32.to(v.dtype), None
