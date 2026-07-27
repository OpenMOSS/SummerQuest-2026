from __future__ import annotations

import math

import torch

from .attention import _validate_attention_inputs, compiled_flash_attention_backward

try:
    import triton
    import triton.language as tl
except ImportError:  # The CPU-only development environment intentionally has no Triton dependency.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
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
        HEAD_DIM: tl.constexpr,
        QUERY_TILE_SIZE: tl.constexpr,
        KEY_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)

        query_offsets = query_tile_index * QUERY_TILE_SIZE + tl.arange(0, QUERY_TILE_SIZE)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query_mask = query_offsets < n_queries
        q = tl.load(
            q_ptr
            + batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + dimension_offsets[None, :] * stride_qd,
            mask=query_mask[:, None],
            other=0.0,
        )

        running_max = tl.full((QUERY_TILE_SIZE,), -float("inf"), tl.float32)
        running_sum = tl.zeros((QUERY_TILE_SIZE,), tl.float32)
        accumulator = tl.zeros((QUERY_TILE_SIZE, HEAD_DIM), tl.float32)

        for key_start in range(0, n_keys, KEY_TILE_SIZE):
            key_offsets = key_start + tl.arange(0, KEY_TILE_SIZE)
            key_mask = key_offsets < n_keys
            k = tl.load(
                k_ptr
                + batch_index * stride_kb
                + key_offsets[:, None] * stride_kk
                + dimension_offsets[None, :] * stride_kd,
                mask=key_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * scale
            valid_scores = query_mask[:, None] & key_mask[None, :]
            if is_causal:
                valid_scores = valid_scores & (query_offsets[:, None] >= key_offsets[None, :])
            scores = tl.where(valid_scores, scores, -1.0e6)

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            correction = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(valid_scores, probabilities, 0.0)
            new_sum = correction * running_sum + tl.sum(probabilities, axis=1)

            v = tl.load(
                v_ptr
                + batch_index * stride_vb
                + key_offsets[:, None] * stride_vk
                + dimension_offsets[None, :] * stride_vd,
                mask=key_mask[:, None],
                other=0.0,
            )
            accumulator *= correction[:, None]
            accumulator = tl.dot(probabilities.to(v.dtype), v, acc=accumulator)
            running_max = new_max
            running_sum = new_sum

        normalized = accumulator / running_sum[:, None]
        lse = running_max + tl.log(running_sum)
        tl.store(
            output_ptr
            + batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + dimension_offsets[None, :] * stride_od,
            normalized,
            mask=query_mask[:, None],
        )
        tl.store(
            lse_ptr + batch_index * stride_lb + query_offsets * stride_lq,
            lse,
            mask=query_mask,
        )


def triton_launch_config(head_dim: int) -> dict[str, int]:
    if head_dim not in {16, 32, 64, 128}:
        raise ValueError("Triton FlashAttention supports head dimensions 16, 32, 64, and 128")
    if head_dim <= 64:
        return {"query_tile_size": 64, "key_tile_size": 64, "num_warps": 4, "num_stages": 2}
    return {"query_tile_size": 64, "key_tile_size": 32, "num_warps": 8, "num_stages": 2}


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if triton is None:
            raise RuntimeError("Triton is not installed; use the CUDA server environment")
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise ValueError("the Triton FlashAttention path requires CUDA tensors")
        _validate_attention_inputs(q, k, v)

        leading_shape = q.shape[:-2]
        n_queries, head_dim = q.shape[-2:]
        n_keys = k.shape[-2]
        batch = math.prod(leading_shape)
        q_flat = q.contiguous().reshape(batch, n_queries, head_dim)
        k_flat = k.contiguous().reshape(batch, n_keys, head_dim)
        v_flat = v.contiguous().reshape(batch, n_keys, head_dim)
        output_flat = torch.empty_like(q_flat)
        lse_flat = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
        config = triton_launch_config(head_dim)
        grid = (triton.cdiv(n_queries, config["query_tile_size"]), batch)

        flash_fwd_kernel[grid](
            q_flat,
            k_flat,
            v_flat,
            output_flat,
            lse_flat,
            *q_flat.stride(),
            *k_flat.stride(),
            *v_flat.stride(),
            *output_flat.stride(),
            *lse_flat.stride(),
            n_queries,
            n_keys,
            1.0 / math.sqrt(head_dim),
            HEAD_DIM=head_dim,
            QUERY_TILE_SIZE=config["query_tile_size"],
            KEY_TILE_SIZE=config["key_tile_size"],
            is_causal=bool(is_causal),
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
        )
        output = output_flat.reshape(*leading_shape, n_queries, head_dim)
        lse = lse_flat.reshape(*leading_shape, n_queries)
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = compiled_flash_attention_backward(
            q,
            k,
            v,
            output,
            grad_output,
            lse,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None
