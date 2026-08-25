import torch
import triton
import triton.language as tl

from .flash_attention_pytorch import compiled_flash_attention_backward


QUERY_TILE_SIZE = 64
KEY_TILE_SIZE = 64
NUM_WARPS = 4
NUM_STAGES = 3


@triton.jit
def flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    logsumexp_ptr,
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
    num_queries,
    num_keys,
    scale,
    num_key_tiles: tl.constexpr,
    head_dim: tl.constexpr,
    query_tile_size: tl.constexpr,
    key_tile_size: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(axis=0)
    batch_index = tl.program_id(axis=1)

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_index * stride_qb,
        shape=(num_queries, head_dim),
        strides=(stride_qq, stride_qd),
        offsets=(
            query_tile_index * query_tile_size,
            0,
        ),
        block_shape=(query_tile_size, head_dim),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch_index * stride_kb,
        shape=(head_dim, num_keys),
        strides=(stride_kd, stride_kk),
        offsets=(0, 0),
        block_shape=(head_dim, key_tile_size),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch_index * stride_vb,
        shape=(num_keys, head_dim),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(key_tile_size, head_dim),
        order=(1, 0),
    )

    output_block_ptr = tl.make_block_ptr(
        base=output_ptr + batch_index * stride_ob,
        shape=(num_queries, head_dim),
        strides=(stride_oq, stride_od),
        offsets=(
            query_tile_index * query_tile_size,
            0,
        ),
        block_shape=(query_tile_size, head_dim),
        order=(1, 0),
    )

    logsumexp_block_ptr = tl.make_block_ptr(
        base=logsumexp_ptr + batch_index * stride_lb,
        shape=(num_queries,),
        strides=(stride_lq,),
        offsets=(
            query_tile_index * query_tile_size,
        ),
        block_shape=(query_tile_size,),
        order=(0,),
    )

    q_tile = tl.load(
        q_block_ptr,
        boundary_check=(0,),
        padding_option="zero",
    )

    row_max = tl.full(
        (query_tile_size,),
        -float("inf"),
        tl.float32,
    )

    row_sum = tl.zeros(
        (query_tile_size,),
        tl.float32,
    )

    output_accumulator = tl.zeros(
        (query_tile_size, head_dim),
        tl.float32,
    )

    query_offsets = (
        query_tile_index * query_tile_size + tl.arange(0, query_tile_size)
    )

    for key_tile_index in tl.range(0, num_key_tiles):
        key_start = key_tile_index * key_tile_size

        k_tile = tl.load(
            k_block_ptr,
            boundary_check=(1,),
            padding_option="zero",
        )

        v_tile = tl.load(
            v_block_ptr,
            boundary_check=(0,),
            padding_option="zero",
        )

        score_tile = tl.dot(
            q_tile,
            k_tile,
            input_precision="ieee",
        ) * scale

        key_offsets = (
            key_start + tl.arange(0, key_tile_size)
        )

        score_mask = (
            key_offsets[None, :] < num_keys
        )

        if is_causal:
            score_mask = score_mask & (
                query_offsets[:, None] >= key_offsets[None, :]
            )

        score_tile = tl.where(
            score_mask,
            score_tile,
            -float("inf"),
        )

        tile_max = tl.max(
            score_tile,
            axis=1,
        )
        new_row_max = tl.maximum(
            row_max,
            tile_max,
        )

        old_correction = tl.exp(
            row_max - new_row_max
        )

        exp_scores = tl.exp(
            score_tile - new_row_max[:, None]
        )

        tile_row_sum = tl.sum(
            exp_scores,
            axis=1,
        )

        row_sum = (
            old_correction * row_sum + tile_row_sum
        )

        output_accumulator *= old_correction[:, None]

        output_accumulator = tl.dot(
            exp_scores.to(v_tile.dtype),
            v_tile,
            acc=output_accumulator,
            input_precision="ieee",
        )

        row_max = new_row_max

        k_block_ptr = k_block_ptr.advance(
            (0, key_tile_size)
        )

        v_block_ptr = v_block_ptr.advance(
            (key_tile_size, 0)
        )

    output_tile = (
        output_accumulator / row_sum[:, None]
    )

    logsumexp_tile = (row_max + tl.log(row_sum))

    tl.store(
        output_block_ptr, output_tile.to(q_tile.dtype), boundary_check=(0,),
    )

    tl.store(
        logsumexp_block_ptr,
        logsumexp_tile,
        boundary_check=(0,),
    )

def triton_flash_attention_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        query_tile_size: int = QUERY_TILE_SIZE,
        key_tile_size: int = KEY_TILE_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_queries, head_dim = q.shape
    num_keys = k.shape[-2]

    output = torch.empty_like(q)
    logsumexp = torch.empty(
        (batch_size, num_queries),
        device=q.device,
        dtype=torch.float32,
    )

    grid = (
        triton.cdiv(
            num_queries,
            query_tile_size,
        ),
        batch_size,
    )

    num_key_tiles = triton.cdiv(
        num_keys, key_tile_size,
    )

    scale = head_dim ** -0.5

    flash_attention_forward_kernel[grid](
        q,
        k,
        v,
        output,
        logsumexp,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        logsumexp.stride(0),
        logsumexp.stride(1),
        num_queries,
        num_keys,
        scale,
        num_key_tiles=num_key_tiles,
        head_dim=head_dim,
        query_tile_size=query_tile_size,
        key_tile_size=key_tile_size,
        is_causal=is_causal,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    return output, logsumexp


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, logsumexp = triton_flash_attention_forward(
            q=q,
            k=k,
            v=v,
            is_causal=is_causal,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            logsumexp,
        )
        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors

        grad_q, grad_k, grad_v = compiled_flash_attention_backward(
            q=q,
            k=k,
            v=v,
            output=output,
            grad_output=grad_output,
            logsumexp=logsumexp,
            is_causal=ctx.is_causal,
        )

        return grad_q, grad_k, grad_v, None
