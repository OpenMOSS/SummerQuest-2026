from __future__ import annotations

import math
import torch


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
    """Reference (unfused) attention returning output and log-sum-exp."""
    d = q.shape[-1]
    s = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(d))
    if is_causal:
        nq, nk = q.shape[-2], k.shape[-2]
        iq = torch.arange(nq, device=q.device)[:, None]
        ik = torch.arange(nk, device=q.device)[None, :]
        s = s.masked_fill(iq < ik, float("-inf"))
    lse = torch.logsumexp(s.float(), dim=-1).to(q.dtype)
    p = torch.softmax(s.float(), dim=-1).to(q.dtype)
    return torch.matmul(p, v), lse


def _tiled_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool,
                     block_q: int = 64, block_k: int = 64):
    """Pure PyTorch online-softmax tiled attention (no materialized score matrix)."""
    b, nq, d = q.shape
    nk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    out = torch.empty_like(q)
    lse = torch.empty((b, nq), device=q.device, dtype=q.dtype)
    for qs in range(0, nq, block_q):
        qe = min(qs + block_q, nq)
        qt = q[:, qs:qe]
        m = torch.full((b, qe - qs), -float("inf"), device=q.device, dtype=torch.float32)
        l = torch.zeros_like(m)
        acc = torch.zeros((b, qe - qs, d), device=q.device, dtype=torch.float32)
        for ks in range(0, nk, block_k):
            ke = min(ks + block_k, nk)
            if is_causal and ks >= qe:
                break
            scores = torch.matmul(qt.float(), k[:, ks:ke].float().transpose(-1, -2)) * scale
            if is_causal:
                iq = torch.arange(qs, qe, device=q.device)[:, None]
                ik = torch.arange(ks, ke, device=q.device)[None, :]
                scores = scores.masked_fill(iq < ik, -float("inf"))
            mb = scores.amax(dim=-1)
            m_new = torch.maximum(m, mb)
            alpha = torch.exp(m - m_new)
            p = torch.exp(scores - m_new.unsqueeze(-1))
            l = alpha * l + p.sum(dim=-1)
            acc = alpha.unsqueeze(-1) * acc + torch.matmul(p, v[:, ks:ke].float())
            m = m_new
        out[:, qs:qe] = (acc / l.clamp_min(1e-20).unsqueeze(-1)).to(q.dtype)
        lse[:, qs:qe] = (m + torch.log(l.clamp_min(1e-20))).to(q.dtype)
    return out, lse


def _tiled_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    o: torch.Tensor, lse: torch.Tensor, do: torch.Tensor,
                    is_causal: bool, block_q: int = 256, block_k: int = 1024):
    """Recompute attention probabilities tile-wise and accumulate gradients.

    This is the required recomputation-style backward, implemented without
    constructing a large autograd graph for every score tile.  It keeps only
    one score/probability tile resident and therefore preserves the memory
    behavior of the tiled forward while avoiding Python/autograd graph churn.
    """
    b, nq, d = q.shape
    nk = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    dq = torch.zeros_like(q, dtype=torch.float32)
    dk = torch.zeros_like(k, dtype=torch.float32)
    dv = torch.zeros_like(v, dtype=torch.float32)
    for qs in range(0, nq, block_q):
        qe = min(qs + block_q, nq)
        qf = q[:, qs:qe].float()
        dof = do[:, qs:qe].float()
        of = o[:, qs:qe].float()
        lf = lse[:, qs:qe].float()
        # D_i = sum_j dO_ij * O_ij, used by the softmax backward identity.
        delta = (dof * of).sum(dim=-1)
        dqt = torch.zeros((b, qe - qs, d), device=q.device, dtype=torch.float32)
        for ks in range(0, nk, block_k):
            ke = min(ks + block_k, nk)
            kf = k[:, ks:ke].float()
            vf = v[:, ks:ke].float()
            scores = torch.matmul(qf, kf.transpose(-1, -2)) * scale
            if is_causal:
                iq = torch.arange(qs, qe, device=q.device)[:, None]
                ik = torch.arange(ks, ke, device=q.device)[None, :]
                valid = iq >= ik
                scores = scores.masked_fill(~valid, -float("inf"))
            else:
                valid = None
            p = torch.exp(scores - lf.unsqueeze(-1))
            if valid is not None:
                p = torch.where(valid, p, torch.zeros_like(p))
            dp = torch.matmul(dof, vf.transpose(-1, -2))
            ds = p * (dp - delta.unsqueeze(-1))
            dqt = dqt + torch.matmul(ds, kf) * scale
            dk[:, ks:ke] += torch.matmul(ds.transpose(-1, -2), qf) * scale
            dv[:, ks:ke] += torch.matmul(p.transpose(-1, -2), dof)
        dq[:, qs:qe] = dqt
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttentionPytorch(torch.autograd.Function):
    """Tiled FlashAttention-style forward with recomputation backward."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        o, l = _tiled_attention(q, k, v, bool(is_causal))
        ctx.is_causal = bool(is_causal)
        # Requirement: save Q/K/V/O and exactly one [B, NQ] LSE tensor.
        ctx.save_for_backward(q, k, v, o, l)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l = ctx.saved_tensors
        return (*_tiled_backward(q, k, v, o, l, do, ctx.is_causal), None)


# Triton implementation.  Each program owns a query tile and loops over K/V
# tiles, maintaining the online-softmax state in FP32.
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _flash_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, lse_ptr,
                          NQ, NK, D, stride_qb, stride_qm, stride_qd,
                          stride_kb, stride_kn, stride_kd,
                          stride_vb, stride_vn, stride_vd,
                          stride_ob, stride_om, stride_od,
                          stride_lb, stride_lm,
                          BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr,
                          SCALE: tl.constexpr, CAUSAL: tl.constexpr,
                          INPUT_FP32: tl.constexpr):
        pid = tl.program_id(0)
        num_q_blocks = tl.cdiv(NQ, BLOCK_M)
        b = pid // num_q_blocks
        q_block = pid % num_q_blocks
        offs_m = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < NQ
        q = tl.load(q_ptr + b * stride_qb + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        m = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
        for ks in tl.range(0, NK, BLOCK_N):
            offs_n = ks + tl.arange(0, BLOCK_N)
            mask_n = offs_n < NK
            k = tl.load(k_ptr + b * stride_kb + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                        mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
            if INPUT_FP32:
                # FP32 correctness uses IEEE multiplication rather than TF32.
                scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
            else:
                # BF16 performance uses tensor cores with FP32 accumulation.
                scores = tl.dot(q, tl.trans(k)) * SCALE
            if CAUSAL:
                valid = mask_m[:, None] & mask_n[None, :] & (offs_n[None, :] <= offs_m[:, None])
            else:
                valid = mask_m[:, None] & mask_n[None, :]
            scores = tl.where(valid, scores, -float("inf"))
            mb = tl.max(scores, axis=1)
            m_new = tl.maximum(m, mb)
            safe_m = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - safe_m))
            p = tl.where(valid, tl.exp(scores - safe_m[:, None]), 0.0)
            vv = tl.load(v_ptr + b * stride_vb + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                         mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
            l = alpha * l + tl.sum(p, axis=1)
            if INPUT_FP32:
                acc = alpha[:, None] * acc + tl.dot(p, vv, input_precision="ieee")
            else:
                acc = alpha[:, None] * acc + tl.dot(p.to(tl.bfloat16), vv)
            m = m_new
        o = acc / tl.maximum(l, 1e-20)[:, None]
        tl.store(o_ptr + b * stride_ob + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 o, mask=mask_m[:, None] & (offs_d[None, :] < D))
        tl.store(lse_ptr + b * stride_lb + offs_m * stride_lm,
                 m + tl.log(tl.maximum(l, 1e-20)), mask=mask_m)

except Exception:  # Triton is optional on CPU-only development machines.
    triton = None


class FlashAttentionTriton(FlashAttentionPytorch):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        is_causal = bool(is_causal)
        if triton is None or not q.is_cuda:
            return FlashAttentionPytorch.forward(ctx, q, k, v, is_causal)
        b, nq, d = q.shape
        nk = k.shape[-2]
        o = torch.empty_like(q)
        l = torch.empty((b, nq), device=q.device, dtype=q.dtype)
        block_d = 1
        while block_d < d:
            block_d *= 2
        # 32-row query tiles keep the D=128 accumulator and operand tiles
        # below the RTX 4090 shared-memory limit while retaining tiled work.
        block_m = 32
        _flash_fwd_kernel[(b * math.ceil(nq / block_m),)](
            q, k, v, o, l, nq, nk, d,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            l.stride(0), l.stride(1),
            BLOCK_D=block_d, BLOCK_M=block_m, BLOCK_N=64,
            SCALE=1.0 / math.sqrt(d), CAUSAL=is_causal,
            INPUT_FP32=q.dtype == torch.float32,
            num_warps=4, num_stages=1,
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, o, l)
        return o
