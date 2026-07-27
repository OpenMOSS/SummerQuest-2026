"""Pure-PyTorch tiled FlashAttention-2 as a torch.autograd.Function.

Forward tiles over queries and keys/values with an online softmax and never
materializes the full N x N score matrix. It saves Q, K, V, O and exactly one
tensor of shape [batch, n_queries]: the log-sum-exp L.

Backward recomputes S and P from the saved Q, K, V and L and applies the
FlashAttention-2 backward formulas (allowed to be plain PyTorch).
"""

from __future__ import annotations

import math

import torch


def flash_backward_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute-based FlashAttention-2 backward (plain PyTorch).

    Given saved Q, K, V, O, L and upstream dO, compute dQ, dK, dV:
        D   = rowsum(dO * O)
        S   = Q K^T * scale  (causal-masked)
        P   = exp(S - L)
        dV  = P^T dO
        dP  = dO V^T
        dS  = P * (dP - D)
        dQ  = dS K * scale
        dK  = dS^T Q * scale
    Shapes: q/o/do (B, Nq, d); k/v (B, Nk, d); lse (B, Nq).
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    qf, kf, vf = q.float(), k.float(), v.float()
    dof, of_ = do.float(), o.float()
    lsef = lse.float()

    s = torch.matmul(qf, kf.transpose(-2, -1)) * scale
    if is_causal:
        nq, nk = s.shape[-2], s.shape[-1]
        q_idx = torch.arange(nq, device=s.device)[:, None]
        k_idx = torch.arange(nk, device=s.device)[None, :]
        s = s.masked_fill(k_idx > q_idx, float("-inf"))
    p = torch.exp(s - lsef.unsqueeze(-1))
    row_d = (dof * of_).sum(dim=-1, keepdim=True)  # (B, Nq, 1)
    dv = torch.matmul(p.transpose(-2, -1), dof)
    dp = torch.matmul(dof, vf.transpose(-2, -1))
    ds = p * (dp - row_d)
    dq = torch.matmul(ds, kf) * scale
    dk = torch.matmul(ds.transpose(-2, -1), qf) * scale
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


class FlashAttentionPyTorch(torch.autograd.Function):
    """Tiled FlashAttention-2 forward + recompute backward, pure PyTorch."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, tile_q=64, tile_k=64):
        batch_shape = q.shape[:-2]
        nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
        scale = 1.0 / math.sqrt(d)

        o = torch.zeros_like(q, dtype=torch.float32)
        m = torch.full(batch_shape + (nq,), float("-inf"), device=q.device, dtype=torch.float32)
        ell = torch.zeros(batch_shape + (nq,), device=q.device, dtype=torch.float32)

        for qs in range(0, nq, tile_q):
            qe = min(qs + tile_q, nq)
            q_tile = q[..., qs:qe, :].float()
            m_t = m[..., qs:qe]
            l_t = ell[..., qs:qe]
            o_t = o[..., qs:qe, :]
            for ks in range(0, nk, tile_k):
                ke = min(ks + tile_k, nk)
                if is_causal and ks >= qe:
                    break  # all remaining keys are masked for this query tile
                k_tile = k[..., ks:ke, :].float()
                v_tile = v[..., ks:ke, :].float()
                s = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale
                if is_causal:
                    q_idx = torch.arange(qs, qe, device=q.device)[:, None]
                    k_idx = torch.arange(ks, ke, device=q.device)[None, :]
                    s = s.masked_fill(k_idx > q_idx, float("-inf"))
                m_new = torch.maximum(m_t, s.amax(dim=-1))
                alpha = torch.exp(m_t - m_new)
                p = torch.exp(s - m_new.unsqueeze(-1))
                l_t = l_t * alpha + p.sum(dim=-1)
                o_t = o_t * alpha.unsqueeze(-1) + torch.matmul(p, v_tile)
                m_t = m_new
            o[..., qs:qe, :] = o_t / l_t.unsqueeze(-1)
            m[..., qs:qe] = m_t
            ell[..., qs:qe] = l_t

        lse = m + torch.log(ell)  # (..., n_queries)
        o_out = o.to(q.dtype)
        ctx.save_for_backward(q, k, v, o_out, lse)
        ctx.is_causal = is_causal
        return o_out

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = flash_backward_recompute(q, k, v, o, lse, do, ctx.is_causal)
        return dq, dk, dv, None, None, None
