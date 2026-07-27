from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _causal_scores(scores: torch.Tensor, query_start: int, key_start: int) -> torch.Tensor:
    query_positions = torch.arange(query_start, query_start + scores.shape[-2], device=scores.device)
    key_positions = torch.arange(key_start, key_start + scores.shape[-1], device=scores.device)
    return scores.masked_fill(query_positions[:, None] < key_positions[None, :], -1e6)


def tiled_flash_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
    query_tile_size: int = 64,
    key_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 Algorithm 1 expressed with ordinary PyTorch tiles."""
    *batch_shape, n_queries, head_dimension = query.shape
    n_keys = key.shape[-2]
    output = torch.empty_like(query)
    lse = torch.empty((*batch_shape, n_queries), device=query.device, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_dimension)

    flat_query = query.reshape(-1, n_queries, head_dimension)
    flat_key = key.reshape(-1, n_keys, head_dimension)
    flat_value = value.reshape(-1, n_keys, head_dimension)
    flat_output = output.reshape(-1, n_queries, head_dimension)
    flat_lse = lse.reshape(-1, n_queries)

    for batch_index in range(flat_query.shape[0]):
        for query_start in range(0, n_queries, query_tile_size):
            query_end = min(query_start + query_tile_size, n_queries)
            query_tile = flat_query[batch_index, query_start:query_end]
            running_max = torch.full(
                (query_end - query_start,),
                -torch.inf,
                device=query.device,
                dtype=torch.float32,
            )
            running_sum = torch.zeros_like(running_max)
            accumulator = torch.zeros(
                (query_end - query_start, head_dimension),
                device=query.device,
                dtype=torch.float32,
            )
            for key_start in range(0, n_keys, key_tile_size):
                key_end = min(key_start + key_tile_size, n_keys)
                key_tile = flat_key[batch_index, key_start:key_end]
                value_tile = flat_value[batch_index, key_start:key_end]
                scores = torch.matmul(query_tile.float(), key_tile.float().transpose(-2, -1)) * scale
                if is_causal:
                    scores = _causal_scores(scores, query_start, key_start)
                tile_max = scores.max(dim=-1).values
                new_max = torch.maximum(running_max, tile_max)
                correction = torch.exp(running_max - new_max)
                probabilities = torch.exp(scores - new_max[:, None])
                running_sum = correction * running_sum + probabilities.sum(dim=-1)
                accumulator = correction[:, None] * accumulator + torch.matmul(probabilities, value_tile.float())
                running_max = new_max
            flat_output[batch_index, query_start:query_end] = (accumulator / running_sum[:, None]).to(query.dtype)
            flat_lse[batch_index, query_start:query_end] = running_max + torch.log(running_sum)
    return output, lse


def recompute_flash_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    output_gradient: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations 13--19 using recomputed probabilities and the D vector."""
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    if is_causal:
        query_positions = torch.arange(query.shape[-2], device=query.device)
        key_positions = torch.arange(key.shape[-2], device=query.device)
        mask = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    probabilities = torch.exp(scores - lse.float().unsqueeze(-1))
    d_value = torch.matmul(probabilities.transpose(-2, -1), output_gradient.float())
    d_probability = torch.matmul(output_gradient.float(), value.float().transpose(-2, -1))
    d_vector = (output.float() * output_gradient.float()).sum(dim=-1, keepdim=True)
    d_scores = probabilities * (d_probability - d_vector)
    d_query = torch.matmul(d_scores, key.float()) * scale
    d_key = torch.matmul(d_scores.transpose(-2, -1), query.float()) * scale
    return d_query.to(query.dtype), d_key.to(key.dtype), d_value.to(value.dtype)


compiled_recompute_flash_backward = torch.compile(recompute_flash_backward, fullgraph=True)


@triton.jit
def flash_forward_kernel(
    query_pointer,
    key_pointer,
    value_pointer,
    output_pointer,
    lse_pointer,
    query_batch_stride,
    query_sequence_stride,
    query_dimension_stride,
    key_batch_stride,
    key_sequence_stride,
    key_dimension_stride,
    value_batch_stride,
    value_sequence_stride,
    value_dimension_stride,
    output_batch_stride,
    output_sequence_stride,
    output_dimension_stride,
    lse_batch_stride,
    lse_sequence_stride,
    n_queries,
    n_keys,
    scale,
    head_dimension: tl.constexpr,
    query_tile_size: tl.constexpr,
    key_tile_size: tl.constexpr,
    block_dimension: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_offsets = query_tile_index * query_tile_size + tl.arange(0, query_tile_size)
    dimension_offsets = tl.arange(0, block_dimension)
    query_mask = (query_offsets[:, None] < n_queries) & (dimension_offsets[None, :] < head_dimension)
    query_addresses = query_pointer + batch_index * query_batch_stride + query_offsets[:, None] * query_sequence_stride + dimension_offsets[None, :] * query_dimension_stride
    query_tile = tl.load(query_addresses, mask=query_mask, other=0.0)

    running_max = tl.full((query_tile_size,), -float("inf"), tl.float32)
    running_sum = tl.zeros((query_tile_size,), tl.float32)
    accumulator = tl.zeros((query_tile_size, block_dimension), tl.float32)

    for key_start in range(0, n_keys, key_tile_size):
        key_offsets = key_start + tl.arange(0, key_tile_size)
        key_mask = (key_offsets[:, None] < n_keys) & (dimension_offsets[None, :] < head_dimension)
        key_addresses = key_pointer + batch_index * key_batch_stride + key_offsets[:, None] * key_sequence_stride + dimension_offsets[None, :] * key_dimension_stride
        value_addresses = value_pointer + batch_index * value_batch_stride + key_offsets[:, None] * value_sequence_stride + dimension_offsets[None, :] * value_dimension_stride
        key_tile = tl.load(key_addresses, mask=key_mask, other=0.0)
        value_tile = tl.load(value_addresses, mask=key_mask, other=0.0)
        scores = tl.dot(query_tile, tl.trans(key_tile), input_precision="ieee") * scale
        valid_scores = (query_offsets[:, None] < n_queries) & (key_offsets[None, :] < n_keys)
        if is_causal:
            valid_scores = valid_scores & (query_offsets[:, None] >= key_offsets[None, :])
        scores = tl.where(valid_scores, scores, -1.0e6)

        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(valid_scores, probabilities, 0.0)
        new_sum = correction * running_sum + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None]
        accumulator = tl.dot(
            probabilities.to(value_tile.dtype),
            value_tile,
            acc=accumulator,
            input_precision="ieee",
        )
        running_max = new_max
        running_sum = new_sum

    normalized_output = accumulator / running_sum[:, None]
    output_addresses = output_pointer + batch_index * output_batch_stride + query_offsets[:, None] * output_sequence_stride + dimension_offsets[None, :] * output_dimension_stride
    tl.store(output_addresses, normalized_output, mask=query_mask)
    lse_addresses = lse_pointer + batch_index * lse_batch_stride + query_offsets * lse_sequence_stride
    tl.store(lse_addresses, running_max + tl.log(running_sum), mask=query_offsets < n_queries)


def triton_flash_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not query.is_cuda or not key.is_cuda or not value.is_cuda:
        raise ValueError("the Triton FlashAttention path requires CUDA tensors")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("the Triton path expects [batch, sequence, head_dimension] tensors")
    if query.shape[0] != key.shape[0] or key.shape != value.shape or query.shape[-1] != key.shape[-1]:
        raise ValueError("incompatible Q/K/V shapes")
    batch_size, n_queries, head_dimension = query.shape
    n_keys = key.shape[-2]
    if head_dimension > 128:
        raise ValueError("head dimensions above 128 are not supported")
    block_dimension = triton.next_power_of_2(head_dimension)
    query_tile_size = 32 if head_dimension >= 128 else 64
    key_tile_size = 32 if head_dimension >= 128 else 64
    num_stages = 1 if head_dimension >= 128 else 2
    output = torch.empty_like(query)
    lse = torch.empty((batch_size, n_queries), device=query.device, dtype=torch.float32)
    grid = (triton.cdiv(n_queries, query_tile_size), batch_size)
    flash_forward_kernel[grid](
        query,
        key,
        value,
        output,
        lse,
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *output.stride(),
        *lse.stride(),
        n_queries,
        n_keys,
        1.0 / math.sqrt(head_dimension),
        head_dimension=head_dimension,
        query_tile_size=query_tile_size,
        key_tile_size=key_tile_size,
        block_dimension=block_dimension,
        is_causal=is_causal,
        num_warps=4,
        num_stages=num_stages,
    )
    return output, lse


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, is_causal=False):
        output, lse = tiled_flash_forward(query, key, value, bool(is_causal))
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        query, key, value, output, lse = ctx.saved_tensors
        d_query, d_key, d_value = compiled_recompute_flash_backward(
            query,
            key,
            value,
            output,
            output_gradient,
            lse,
            ctx.is_causal,
        )
        return d_query, d_key, d_value, None


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, is_causal=False):
        output, lse = triton_flash_forward(query, key, value, bool(is_causal))
        ctx.save_for_backward(query, key, value, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        query, key, value, output, lse = ctx.saved_tensors
        d_query, d_key, d_value = compiled_recompute_flash_backward(
            query,
            key,
            value,
            output,
            output_gradient,
            lse,
            ctx.is_causal,
        )
        return d_query, d_key, d_value, None
