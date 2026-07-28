from __future__ import annotations

import math
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only environments can still run the PyTorch path.
    triton = None
    tl = None


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit, unfused attention baseline returning output and log-sum-exp."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        query_pos = torch.arange(n_queries, device=q.device)[:, None]
        key_pos = torch.arange(n_keys, device=q.device)[None, :]
        scores = scores.masked_fill(query_pos < key_pos, -1e6)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v), torch.logsumexp(scores, dim=-1)


def _flash_backward_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        query_pos = torch.arange(n_queries, device=q.device)[:, None]
        key_pos = torch.arange(n_keys, device=q.device)[None, :]
        scores = scores.masked_fill(query_pos < key_pos, -1e6)

    probs = torch.exp(scores - lse.unsqueeze(-1))
    d_vec = torch.sum(o * grad_o, dim=-1)
    grad_v = torch.matmul(probs.transpose(-1, -2), grad_o)
    grad_p = torch.matmul(grad_o, v.transpose(-1, -2))
    grad_s = probs * (grad_p - d_vec.unsqueeze(-1))
    grad_q = torch.matmul(grad_s, k) * scale
    grad_k = torch.matmul(grad_s.transpose(-1, -2), q) * scale
    return grad_q, grad_k, grad_v


_compiled_flash_backward_recompute = None


def _flash_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    grad_o: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use the required torch.compile recompute path on CUDA, with CPU fallback for tests."""
    global _compiled_flash_backward_recompute
    if q.is_cuda:
        if _compiled_flash_backward_recompute is None:
            _compiled_flash_backward_recompute = torch.compile(_flash_backward_recompute, fullgraph=True)
        return _compiled_flash_backward_recompute(q, k, v, o, grad_o, lse, is_causal)
    return _flash_backward_recompute(q, k, v, o, grad_o, lse, is_causal)


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        qf, kf, vf = q.float(), k.float(), v.float()
        batch_shape = q.shape[:-2]
        n_queries, d = q.shape[-2], q.shape[-1]
        n_keys = k.shape[-2]
        q_flat = qf.reshape(-1, n_queries, d)
        k_flat = kf.reshape(-1, n_keys, d)
        v_flat = vf.reshape(-1, n_keys, d)
        o_flat = torch.empty_like(q_flat)
        lse_flat = torch.empty(q_flat.shape[0], n_queries, device=q.device, dtype=torch.float32)

        scale = 1.0 / math.sqrt(d)
        q_tile_size = 64
        k_tile_size = 64
        for batch_idx in range(q_flat.shape[0]):
            for q_start in range(0, n_queries, q_tile_size):
                q_end = min(q_start + q_tile_size, n_queries)
                q_tile = q_flat[batch_idx, q_start:q_end]
                m = torch.full((q_end - q_start,), -torch.inf, device=q.device, dtype=torch.float32)
                l = torch.zeros((q_end - q_start,), device=q.device, dtype=torch.float32)
                acc = torch.zeros((q_end - q_start, d), device=q.device, dtype=torch.float32)
                query_pos = torch.arange(q_start, q_end, device=q.device)[:, None]
                for k_start in range(0, n_keys, k_tile_size):
                    k_end = min(k_start + k_tile_size, n_keys)
                    k_tile = k_flat[batch_idx, k_start:k_end]
                    v_tile = v_flat[batch_idx, k_start:k_end]
                    scores = torch.matmul(q_tile, k_tile.transpose(0, 1)) * scale
                    if is_causal:
                        key_pos = torch.arange(k_start, k_end, device=q.device)[None, :]
                        scores = scores.masked_fill(query_pos < key_pos, -1e6)
                    m_next = torch.maximum(m, torch.max(scores, dim=1).values)
                    p = torch.exp(scores - m_next[:, None])
                    alpha = torch.exp(m - m_next)
                    acc = acc * alpha[:, None] + torch.matmul(p, v_tile)
                    l = l * alpha + torch.sum(p, dim=1)
                    m = m_next
                o_flat[batch_idx, q_start:q_end] = acc / l[:, None]
                lse_flat[batch_idx, q_start:q_end] = m + torch.log(l)

        o = o_flat.reshape(*batch_shape, n_queries, d).to(dtype=q.dtype)
        lse = lse_flat.reshape(*batch_shape, n_queries)
        ctx.save_for_backward(lse, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx: Any, grad_o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        lse, q, k, v, o = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward(q.float(), k.float(), v.float(), o.float(), grad_o.float(), lse.float(), ctx.is_causal)
        return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), None


if triton is not None:

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
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
        scale: tl.constexpr,
        d: tl.constexpr,
        q_tile_size: tl.constexpr,
        k_tile_size: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        q_offsets = query_tile_index * q_tile_size + tl.arange(0, q_tile_size)
        d_offsets = tl.arange(0, d)

        q = tl.load(
            q_ptr + batch_index * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd,
            mask=q_offsets[:, None] < n_queries,
            other=0.0,
        )
        m = tl.full((q_tile_size,), -float("inf"), tl.float32)
        l = tl.zeros((q_tile_size,), tl.float32)
        acc = tl.zeros((q_tile_size, d), tl.float32)

        for key_start in range(0, n_keys, k_tile_size):
            k_offsets = key_start + tl.arange(0, k_tile_size)
            k = tl.load(
                k_ptr + batch_index * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd,
                mask=k_offsets[:, None] < n_keys,
                other=0.0,
            )
            v = tl.load(
                v_ptr + batch_index * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd,
                mask=k_offsets[:, None] < n_keys,
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(k_offsets[None, :] < n_keys, scores, -float("inf"))
            if is_causal:
                scores = tl.where(q_offsets[:, None] >= k_offsets[None, :], scores, -1.0e6)
            scores = tl.where(q_offsets[:, None] < n_queries, scores, -float("inf"))

            m_next = tl.maximum(m, tl.max(scores, axis=1))
            p = tl.exp(scores - m_next[:, None])
            alpha = tl.exp(m - m_next)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l = l * alpha + tl.sum(p, axis=1)
            m = m_next

        out = acc / l[:, None]
        lse = m + tl.log(l)
        tl.store(
            o_ptr + batch_index * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od,
            out,
            mask=q_offsets[:, None] < n_queries,
        )
        tl.store(l_ptr + batch_index * stride_lb + q_offsets * stride_lq, lse, mask=q_offsets < n_queries)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is required for FlashAttentionTriton")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("FlashAttentionTriton requires CUDA tensors")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError("q, k, and v must share batch dimensions")
        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            raise ValueError("q, k, and v must share head dimension")

        batch = math.prod(q.shape[:-2])
        n_queries, d = q.shape[-2], q.shape[-1]
        n_keys = k.shape[-2]
        q_3d = q.contiguous().reshape(batch, n_queries, d)
        k_3d = k.contiguous().reshape(batch, n_keys, d)
        v_3d = v.contiguous().reshape(batch, n_keys, d)
        o_3d = torch.empty_like(q_3d)
        lse_2d = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
        q_tile_size = 64
        k_tile_size = 64
        num_warps = 4 if d <= 64 else 8
        _flash_fwd_kernel[(triton.cdiv(n_queries, q_tile_size), batch)](
            q_3d,
            k_3d,
            v_3d,
            o_3d,
            lse_2d,
            q_3d.stride(0),
            q_3d.stride(1),
            q_3d.stride(2),
            k_3d.stride(0),
            k_3d.stride(1),
            k_3d.stride(2),
            v_3d.stride(0),
            v_3d.stride(1),
            v_3d.stride(2),
            o_3d.stride(0),
            o_3d.stride(1),
            o_3d.stride(2),
            lse_2d.stride(0),
            lse_2d.stride(1),
            n_queries,
            n_keys,
            1.0 / math.sqrt(d),
            d,
            q_tile_size,
            k_tile_size,
            is_causal,
            num_warps=num_warps,
            num_stages=3,
        )
        o = o_3d.reshape_as(q)
        lse = lse_2d.reshape(*q.shape[:-2], n_queries)
        ctx.save_for_backward(lse, q, k, v, o)
        ctx.is_causal = is_causal
        return o

    @staticmethod
    def backward(ctx: Any, grad_o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        lse, q, k, v, o = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward(q.float(), k.float(), v.float(), o.float(), grad_o.float(), lse.float(), ctx.is_causal)
        return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), None
