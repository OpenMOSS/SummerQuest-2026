"""A2-K FlashAttention-2 implementations.

The PyTorch path is a tiled reference implementation.  The Triton path owns
the forward kernel; backward deliberately uses the assignment-allowed
recomputation path implemented with ordinary PyTorch operations.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is only available on supported CUDA platforms.
    triton = None
    tl = None


def _attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = q @ k.transpose(-2, -1) * scale
    if is_causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=k.device)
        scores = scores.masked_fill(query_positions[:, None] < key_positions[None, :], -1e6)

    probabilities = torch.exp(scores.float() - lse.float().unsqueeze(-1)).to(q.dtype)
    delta = (o.float() * do.float()).sum(dim=-1)
    dv = probabilities.transpose(-2, -1) @ do
    dp = do @ v.transpose(-2, -1)
    ds = probabilities * (dp - delta.unsqueeze(-1).to(dp.dtype))
    dq = (ds @ k) * scale
    dk = (ds.transpose(-2, -1) @ q) * scale
    return dq, dk, dv


@lru_cache(maxsize=2)
def _compiled_attention_backward(is_causal: bool):
    if not hasattr(torch, "compile"):
        return _attention_backward

    def backward_fn(q, k, v, o, do, lse):
        return _attention_backward(q, k, v, o, do, lse, is_causal)

    return torch.compile(backward_fn, fullgraph=True)


def _run_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.is_cuda:
        try:
            return _compiled_attention_backward(is_causal)(q, k, v, o, do, lse)
        except Exception:
            pass
    return _attention_backward(q, k, v, o, do, lse, is_causal)


class FlashAttentionPyTorch(torch.autograd.Function):
    """Pure-PyTorch tiled FlashAttention-2 reference implementation."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
            raise ValueError("Q, K, and V must have the same leading dimensions")
        if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
            raise ValueError("incompatible Q, K, and V shapes")

        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        scale = 1.0 / math.sqrt(q.shape[-1])
        query_tile_size = min(64, n_queries)
        key_tile_size = min(64, n_keys)
        output = torch.empty((*q.shape[:-1], v.shape[-1]), dtype=q.dtype, device=q.device)
        lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)

        for query_start in range(0, n_queries, query_tile_size):
            query_end = min(query_start + query_tile_size, n_queries)
            q_tile = q[..., query_start:query_end, :]
            running_max = torch.full(q_tile.shape[:-1], -torch.inf, dtype=torch.float32, device=q.device)
            running_sum = torch.zeros_like(running_max)
            accumulator = torch.zeros(
                (*q_tile.shape[:-1], v.shape[-1]),
                dtype=torch.float32,
                device=q.device,
            )

            for key_start in range(0, n_keys, key_tile_size):
                key_end = min(key_start + key_tile_size, n_keys)
                k_tile = k[..., key_start:key_end, :]
                v_tile = v[..., key_start:key_end, :]
                scores = (q_tile @ k_tile.transpose(-2, -1)).float() * scale
                if is_causal:
                    query_positions = torch.arange(query_start, query_end, device=q.device)
                    key_positions = torch.arange(key_start, key_end, device=q.device)
                    scores = scores.masked_fill(query_positions[:, None] < key_positions[None, :], -1e6)

                tile_max = scores.amax(dim=-1)
                new_max = torch.maximum(running_max, tile_max)
                correction = torch.exp(running_max - new_max)
                probabilities = torch.exp(scores - new_max.unsqueeze(-1))
                running_sum = correction * running_sum + probabilities.sum(dim=-1)
                accumulator = correction.unsqueeze(-1) * accumulator + probabilities @ v_tile.float()
                running_max = new_max

            output[..., query_start:query_end, :] = (accumulator / running_sum.unsqueeze(-1)).to(q.dtype)
            lse[..., query_start:query_end] = running_max + torch.log(running_sum)

        ctx.save_for_backward(lse, q, k, v, output)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        lse, q, k, v, o = ctx.saved_tensors
        dq, dk, dv = _run_attention_backward(q, k, v, o, do, lse, ctx.is_causal)
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
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
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        key_offsets = tl.arange(0, K_TILE_SIZE)
        d_offsets = tl.arange(0, D)
        query_mask = query_offsets < n_queries

        q = tl.load(
            q_ptr
            + batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + d_offsets[None, :] * stride_qd,
            mask=query_mask[:, None],
            other=0.0,
        )
        running_max = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
        running_sum = tl.zeros((Q_TILE_SIZE,), tl.float32)
        accumulator = tl.zeros((Q_TILE_SIZE, D), tl.float32)

        for key_start in range(0, n_keys, K_TILE_SIZE):
            current_keys = key_start + key_offsets
            key_mask = current_keys < n_keys
            k = tl.load(
                k_ptr
                + batch_index * stride_kb
                + current_keys[:, None] * stride_kk
                + d_offsets[None, :] * stride_kd,
                mask=key_mask[:, None],
                other=0.0,
            )
            v = tl.load(
                v_ptr
                + batch_index * stride_vb
                + current_keys[:, None] * stride_vk
                + d_offsets[None, :] * stride_vd,
                mask=key_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            valid_scores = query_mask[:, None] & key_mask[None, :]
            scores = tl.where(valid_scores, scores, -1.0e6)
            if IS_CAUSAL:
                scores = tl.where(query_offsets[:, None] >= current_keys[None, :], scores, -1.0e6)

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            correction = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(valid_scores, probabilities, 0.0)
            running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
            accumulator *= correction[:, None]
            accumulator = tl.dot(probabilities.to(v.dtype), v, acc=accumulator)
            running_max = new_max

        output = accumulator / running_sum[:, None]
        lse = running_max + tl.log(running_sum)
        tl.store(
            o_ptr
            + batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + d_offsets[None, :] * stride_od,
            output,
            mask=query_mask[:, None],
        )
        tl.store(
            l_ptr + batch_index * stride_lb + query_offsets * stride_lq,
            lse,
            mask=query_mask,
        )


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 with a student Triton forward and recompute backward."""

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if not q.is_cuda:
            raise ValueError("the Triton FlashAttention implementation requires CUDA tensors")
        if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
            raise ValueError("Q, K, and V must have the same leading dimensions")
        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            raise ValueError("the Triton kernel requires Q, K, and V to have the same head dimension")

        original_shape = q.shape
        batch = math.prod(q.shape[:-2])
        n_queries, dimension = q.shape[-2:]
        n_keys = k.shape[-2]
        q_3d = q.contiguous().reshape(batch, n_queries, dimension)
        k_3d = k.contiguous().reshape(batch, n_keys, dimension)
        v_3d = v.contiguous().reshape(batch, n_keys, dimension)
        output_3d = torch.empty_like(q_3d)
        lse_2d = torch.empty((batch, n_queries), dtype=torch.float32, device=q.device)
        tile_size = 32 if dimension >= 128 and q.element_size() == 4 else 64
        num_warps = 2 if tile_size == 32 else 4
        grid = (triton.cdiv(n_queries, tile_size), batch)
        _flash_forward_kernel[grid](
            q_3d,
            k_3d,
            v_3d,
            output_3d,
            lse_2d,
            *q_3d.stride(),
            *k_3d.stride(),
            *v_3d.stride(),
            *output_3d.stride(),
            *lse_2d.stride(),
            n_queries,
            n_keys,
            1.0 / math.sqrt(dimension),
            D=dimension,
            Q_TILE_SIZE=tile_size,
            K_TILE_SIZE=tile_size,
            IS_CAUSAL=bool(is_causal),
            num_warps=num_warps,
            num_stages=2,
        )
        output = output_3d.reshape(original_shape)
        lse = lse_2d.reshape(q.shape[:-1])
        ctx.save_for_backward(lse, q, k, v, output)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        lse, q, k, v, o = ctx.saved_tensors
        dq, dk, dv = _run_attention_backward(q, k, v, o, do, lse, ctx.is_causal)
        return dq, dk, dv, None
