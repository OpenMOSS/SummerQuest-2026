"""Student-written Triton FlashAttention forward and backward kernels."""

from __future__ import annotations

import math
from typing import TypedDict

import torch

from .attention import validate_attention_inputs

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


class LaunchConfig(TypedDict):
    """Compile-time tile and launch parameters for one head dimension."""

    query_tile: int
    key_tile: int
    num_warps: int
    num_stages: int


def triton_launch_config(head_dim: int) -> LaunchConfig:
    """Return a conservative RTX 4090 launch configuration."""

    if head_dim in {32, 64}:
        return {
            "query_tile": 64,
            "key_tile": 64,
            "num_warps": 4,
            "num_stages": 1,
        }
    if head_dim == 128:
        return {
            "query_tile": 32,
            "key_tile": 32,
            "num_warps": 4,
            "num_stages": 1,
        }
    raise ValueError("supported head dimensions are 32, 64, and 128")


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
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
        SCALE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        INPUT_FP32: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = (
            query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        )
        key_offsets_base = tl.arange(0, BLOCK_N)
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
        running_max = tl.full(
            (BLOCK_M,),
            -float("inf"),
            tl.float32,
        )
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros(
            (BLOCK_M, HEAD_DIM),
            tl.float32,
        )

        for key_start in tl.range(0, n_keys, BLOCK_N):
            key_offsets = key_start + key_offsets_base
            key_mask = key_offsets < n_keys
            k = tl.load(
                k_ptr
                + batch_index * stride_kb
                + key_offsets[:, None] * stride_kk
                + dimension_offsets[None, :] * stride_kd,
                mask=key_mask[:, None],
                other=0.0,
            )
            if INPUT_FP32:
                scores = (
                    tl.dot(q, tl.trans(k), input_precision="ieee")
                    * SCALE
                )
            else:
                scores = tl.dot(q, tl.trans(k)) * SCALE

            valid = query_mask[:, None] & key_mask[None, :]
            if CAUSAL:
                valid &= (
                    query_offsets[:, None] >= key_offsets[None, :]
                )
            scores = tl.where(valid, scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            safe_max = tl.where(
                new_max == -float("inf"),
                0.0,
                new_max,
            )
            correction = tl.where(
                running_max == -float("inf"),
                0.0,
                tl.exp(running_max - safe_max),
            )
            probabilities = tl.where(
                valid,
                tl.exp(scores - safe_max[:, None]),
                0.0,
            )
            value = tl.load(
                v_ptr
                + batch_index * stride_vb
                + key_offsets[:, None] * stride_vk
                + dimension_offsets[None, :] * stride_vd,
                mask=key_mask[:, None],
                other=0.0,
            )
            running_sum = (
                correction * running_sum
                + tl.sum(probabilities, axis=1)
            )
            accumulator *= correction[:, None]
            if INPUT_FP32:
                accumulator += tl.dot(
                    probabilities,
                    value,
                    input_precision="ieee",
                )
            else:
                accumulator += tl.dot(
                    probabilities.to(tl.bfloat16),
                    value,
                )
            running_max = new_max

        normalized = accumulator / tl.maximum(
            running_sum,
            1e-20,
        )[:, None]
        output_offsets = (
            batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + dimension_offsets[None, :] * stride_od
        )
        tl.store(
            output_ptr + output_offsets,
            normalized,
            mask=query_mask[:, None],
        )
        tl.store(
            lse_ptr
            + batch_index * stride_lb
            + query_offsets * stride_lq,
            running_max + tl.log(tl.maximum(running_sum, 1e-20)),
            mask=query_mask,
        )

    @triton.jit
    def _flash_backward_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
        grad_output_ptr,
        grad_q_ptr,
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
        stride_gob,
        stride_goq,
        stride_god,
        stride_gqb,
        stride_gqq,
        stride_gqd,
        n_queries,
        n_keys,
        SCALE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        INPUT_FP32: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = (
            query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        )
        key_offsets_base = tl.arange(0, BLOCK_N)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        query_mask = query_offsets < n_queries

        q_offsets = (
            batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + dimension_offsets[None, :] * stride_qd
        )
        output_offsets = (
            batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + dimension_offsets[None, :] * stride_od
        )
        grad_output_offsets = (
            batch_index * stride_gob
            + query_offsets[:, None] * stride_goq
            + dimension_offsets[None, :] * stride_god
        )
        q = tl.load(
            q_ptr + q_offsets,
            mask=query_mask[:, None],
            other=0.0,
        )
        output = tl.load(
            output_ptr + output_offsets,
            mask=query_mask[:, None],
            other=0.0,
        )
        grad_output = tl.load(
            grad_output_ptr + grad_output_offsets,
            mask=query_mask[:, None],
            other=0.0,
        )
        lse = tl.load(
            lse_ptr
            + batch_index * stride_lb
            + query_offsets * stride_lq,
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(
            output.to(tl.float32) * grad_output.to(tl.float32),
            axis=1,
        )
        grad_q = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)

        for key_start in tl.range(0, n_keys, BLOCK_N):
            key_offsets = key_start + key_offsets_base
            key_mask = key_offsets < n_keys
            k = tl.load(
                k_ptr
                + batch_index * stride_kb
                + key_offsets[:, None] * stride_kk
                + dimension_offsets[None, :] * stride_kd,
                mask=key_mask[:, None],
                other=0.0,
            )
            value = tl.load(
                v_ptr
                + batch_index * stride_vb
                + key_offsets[:, None] * stride_vk
                + dimension_offsets[None, :] * stride_vd,
                mask=key_mask[:, None],
                other=0.0,
            )
            if INPUT_FP32:
                scores = (
                    tl.dot(q, tl.trans(k), input_precision="ieee")
                    * SCALE
                )
                grad_probabilities = tl.dot(
                    grad_output,
                    tl.trans(value),
                    input_precision="ieee",
                )
            else:
                scores = tl.dot(q, tl.trans(k)) * SCALE
                grad_probabilities = tl.dot(
                    grad_output,
                    tl.trans(value),
                )

            valid = query_mask[:, None] & key_mask[None, :]
            if CAUSAL:
                valid &= (
                    query_offsets[:, None] >= key_offsets[None, :]
                )
            probabilities = tl.where(
                valid,
                tl.exp(scores - lse[:, None]),
                0.0,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            if INPUT_FP32:
                grad_q += (
                    tl.dot(
                        grad_scores,
                        k,
                        input_precision="ieee",
                    )
                    * SCALE
                )
            else:
                grad_q += (
                    tl.dot(grad_scores.to(tl.bfloat16), k) * SCALE
                )

        grad_q_offsets = (
            batch_index * stride_gqb
            + query_offsets[:, None] * stride_gqq
            + dimension_offsets[None, :] * stride_gqd
        )
        tl.store(
            grad_q_ptr + grad_q_offsets,
            grad_q,
            mask=query_mask[:, None],
        )

    @triton.jit
    def _flash_backward_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
        grad_output_ptr,
        grad_k_ptr,
        grad_v_ptr,
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
        stride_gob,
        stride_goq,
        stride_god,
        stride_gkb,
        stride_gkk,
        stride_gkd,
        stride_gvb,
        stride_gvk,
        stride_gvd,
        n_queries,
        n_keys,
        SCALE: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CAUSAL: tl.constexpr,
        INPUT_FP32: tl.constexpr,
    ):
        key_block = tl.program_id(0)
        batch_index = tl.program_id(1)
        key_offsets = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        query_offsets_base = tl.arange(0, BLOCK_M)
        dimension_offsets = tl.arange(0, HEAD_DIM)
        key_mask = key_offsets < n_keys

        k = tl.load(
            k_ptr
            + batch_index * stride_kb
            + key_offsets[:, None] * stride_kk
            + dimension_offsets[None, :] * stride_kd,
            mask=key_mask[:, None],
            other=0.0,
        )
        value = tl.load(
            v_ptr
            + batch_index * stride_vb
            + key_offsets[:, None] * stride_vk
            + dimension_offsets[None, :] * stride_vd,
            mask=key_mask[:, None],
            other=0.0,
        )
        grad_k = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)
        grad_v = tl.zeros((BLOCK_N, HEAD_DIM), tl.float32)

        for query_start in tl.range(0, n_queries, BLOCK_M):
            query_offsets = query_start + query_offsets_base
            query_mask = query_offsets < n_queries
            q = tl.load(
                q_ptr
                + batch_index * stride_qb
                + query_offsets[:, None] * stride_qq
                + dimension_offsets[None, :] * stride_qd,
                mask=query_mask[:, None],
                other=0.0,
            )
            output = tl.load(
                output_ptr
                + batch_index * stride_ob
                + query_offsets[:, None] * stride_oq
                + dimension_offsets[None, :] * stride_od,
                mask=query_mask[:, None],
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr
                + batch_index * stride_gob
                + query_offsets[:, None] * stride_goq
                + dimension_offsets[None, :] * stride_god,
                mask=query_mask[:, None],
                other=0.0,
            )
            lse = tl.load(
                lse_ptr
                + batch_index * stride_lb
                + query_offsets * stride_lq,
                mask=query_mask,
                other=0.0,
            )
            delta = tl.sum(
                output.to(tl.float32)
                * grad_output.to(tl.float32),
                axis=1,
            )
            if INPUT_FP32:
                scores = (
                    tl.dot(q, tl.trans(k), input_precision="ieee")
                    * SCALE
                )
                grad_probabilities = tl.dot(
                    grad_output,
                    tl.trans(value),
                    input_precision="ieee",
                )
            else:
                scores = tl.dot(q, tl.trans(k)) * SCALE
                grad_probabilities = tl.dot(
                    grad_output,
                    tl.trans(value),
                )

            valid = query_mask[:, None] & key_mask[None, :]
            if CAUSAL:
                valid &= (
                    query_offsets[:, None] >= key_offsets[None, :]
                )
            probabilities = tl.where(
                valid,
                tl.exp(scores - lse[:, None]),
                0.0,
            )
            grad_scores = probabilities * (
                grad_probabilities - delta[:, None]
            )
            if INPUT_FP32:
                grad_k += (
                    tl.dot(
                        tl.trans(grad_scores),
                        q,
                        input_precision="ieee",
                    )
                    * SCALE
                )
                grad_v += tl.dot(
                    tl.trans(probabilities),
                    grad_output,
                    input_precision="ieee",
                )
            else:
                grad_k += (
                    tl.dot(
                        tl.trans(grad_scores).to(tl.bfloat16),
                        q,
                    )
                    * SCALE
                )
                grad_v += tl.dot(
                    tl.trans(probabilities).to(tl.bfloat16),
                    grad_output,
                )

        grad_k_offsets = (
            batch_index * stride_gkb
            + key_offsets[:, None] * stride_gkk
            + dimension_offsets[None, :] * stride_gkd
        )
        grad_v_offsets = (
            batch_index * stride_gvb
            + key_offsets[:, None] * stride_gvk
            + dimension_offsets[None, :] * stride_gvd
        )
        tl.store(
            grad_k_ptr + grad_k_offsets,
            grad_k,
            mask=key_mask[:, None],
        )
        tl.store(
            grad_v_ptr + grad_v_offsets,
            grad_v,
            mask=key_mask[:, None],
        )


def _require_triton_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    if triton is None:
        raise RuntimeError("Triton is not installed")
    validate_attention_inputs(q, k, v)
    if not q.is_cuda:
        raise ValueError("the Triton attention path requires CUDA tensors")
    if q.dtype not in {torch.bfloat16, torch.float32}:
        raise ValueError("the Triton path supports BF16 and FP32")
    triton_launch_config(q.shape[-1])


def _triton_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_triton_inputs(q, k, v)
    leading_shape = q.shape[:-2]
    n_queries, head_dim = q.shape[-2:]
    n_keys = k.shape[-2]
    flat_batch = math.prod(leading_shape)
    q_flat = q.contiguous().reshape(flat_batch, n_queries, head_dim)
    k_flat = k.contiguous().reshape(flat_batch, n_keys, head_dim)
    v_flat = v.contiguous().reshape(flat_batch, n_keys, head_dim)
    output_flat = torch.empty_like(q_flat)
    lse_flat = torch.empty(
        (flat_batch, n_queries),
        device=q.device,
        dtype=torch.float32,
    )
    config = triton_launch_config(head_dim)
    grid = (
        triton.cdiv(n_queries, config["query_tile"]),
        flat_batch,
    )
    _flash_forward_kernel[grid](
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
        SCALE=1.0 / math.sqrt(head_dim),
        HEAD_DIM=head_dim,
        BLOCK_M=config["query_tile"],
        BLOCK_N=config["key_tile"],
        CAUSAL=bool(is_causal),
        INPUT_FP32=q.dtype == torch.float32,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )
    return (
        output_flat.reshape(*leading_shape, n_queries, head_dim),
        lse_flat.reshape(*leading_shape, n_queries),
    )


def _triton_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require_triton_inputs(q, k, v)
    leading_shape = q.shape[:-2]
    n_queries, head_dim = q.shape[-2:]
    n_keys = k.shape[-2]
    flat_batch = math.prod(leading_shape)
    q_flat = q.contiguous().reshape(flat_batch, n_queries, head_dim)
    k_flat = k.contiguous().reshape(flat_batch, n_keys, head_dim)
    v_flat = v.contiguous().reshape(flat_batch, n_keys, head_dim)
    output_flat = output.contiguous().reshape(
        flat_batch,
        n_queries,
        head_dim,
    )
    lse_flat = lse.contiguous().reshape(flat_batch, n_queries)
    grad_output_flat = grad_output.contiguous().reshape(
        flat_batch,
        n_queries,
        head_dim,
    )
    grad_q_flat = torch.empty_like(q_flat)
    grad_k_flat = torch.empty_like(k_flat)
    grad_v_flat = torch.empty_like(v_flat)
    config = triton_launch_config(head_dim)
    common_meta = {
        "SCALE": 1.0 / math.sqrt(head_dim),
        "HEAD_DIM": head_dim,
        "BLOCK_M": config["query_tile"],
        "BLOCK_N": config["key_tile"],
        "CAUSAL": bool(is_causal),
        "INPUT_FP32": q.dtype == torch.float32,
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }

    dq_grid = (
        triton.cdiv(n_queries, config["query_tile"]),
        flat_batch,
    )
    _flash_backward_dq_kernel[dq_grid](
        q_flat,
        k_flat,
        v_flat,
        output_flat,
        lse_flat,
        grad_output_flat,
        grad_q_flat,
        *q_flat.stride(),
        *k_flat.stride(),
        *v_flat.stride(),
        *output_flat.stride(),
        *lse_flat.stride(),
        *grad_output_flat.stride(),
        *grad_q_flat.stride(),
        n_queries,
        n_keys,
        **common_meta,
    )
    dkdv_grid = (
        triton.cdiv(n_keys, config["key_tile"]),
        flat_batch,
    )
    _flash_backward_dkdv_kernel[dkdv_grid](
        q_flat,
        k_flat,
        v_flat,
        output_flat,
        lse_flat,
        grad_output_flat,
        grad_k_flat,
        grad_v_flat,
        *q_flat.stride(),
        *k_flat.stride(),
        *v_flat.stride(),
        *output_flat.stride(),
        *lse_flat.stride(),
        *grad_output_flat.stride(),
        *grad_k_flat.stride(),
        *grad_v_flat.stride(),
        n_queries,
        n_keys,
        **common_meta,
    )
    return (
        grad_q_flat.reshape_as(q),
        grad_k_flat.reshape_as(k),
        grad_v_flat.reshape_as(v),
    )


class FlashAttentionTriton(torch.autograd.Function):
    """Autograd interface backed by the student Triton kernels."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = _triton_forward(q, k, v, bool(is_causal))
        ctx.is_causal = bool(is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = _triton_backward(
            q,
            k,
            v,
            output,
            lse,
            grad_output,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None
