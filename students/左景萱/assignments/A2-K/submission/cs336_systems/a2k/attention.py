from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in environments without Triton.
    triton = None
    tl = None


TRITON_CONFIG = {
    "block_m": 64,
    "block_n": 64,
    "num_warps": 4,
    "num_stages": 2,
}


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("q, k, and v must have matching batch dimensions and rank >= 2")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must have matching batch dimensions")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("incompatible q, k, and v sequence or embedding dimensions")
    if q.shape[-1] != v.shape[-1]:
        raise ValueError("this implementation requires q, k, and v to share an embedding dimension")
    if q.shape[-1] == 0 or k.shape[-2] == 0:
        raise ValueError("the embedding dimension and key sequence length must be non-zero")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise TypeError("q, k, and v must use a floating-point dtype")


def _accumulation_dtype(tensor: torch.Tensor) -> torch.dtype:
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
    batch_size = math.prod(tensor.shape[:-2])
    return tensor.reshape(batch_size, tensor.shape[-2], tensor.shape[-1])


def eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute scaled dot-product attention with explicit PyTorch operations.

    This intentionally materializes the score and probability matrices. It is
    the unfused baseline used to compare eager and ``torch.compile`` execution
    against the tiled implementations below.
    """
    _validate_inputs(q, k, v)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(q.shape[-1]))

    if is_causal:
        query_positions = torch.arange(q.shape[-2], device=q.device)
        key_positions = torch.arange(k.shape[-2], device=q.device)
        causal_mask = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~causal_mask, -torch.inf)

    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def _flash_forward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    block_q: int = 64,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_inputs(q, k, v)
    q_flat = _flatten_batch(q)
    k_flat = _flatten_batch(k)
    v_flat = _flatten_batch(v)
    batch_size, n_queries, d = q_flat.shape
    n_keys = k_flat.shape[1]
    scale = 1.0 / math.sqrt(d)
    accumulation_dtype = _accumulation_dtype(q)

    output = torch.empty_like(q_flat)
    lse = torch.empty((batch_size, n_queries), device=q.device, dtype=accumulation_dtype)

    for q_start in range(0, n_queries, block_q):
        q_end = min(q_start + block_q, n_queries)
        q_block = q_flat[:, q_start:q_end].to(accumulation_dtype)
        rows = q_end - q_start
        row_max = torch.full((batch_size, rows), -torch.inf, device=q.device, dtype=accumulation_dtype)
        row_sum = torch.zeros((batch_size, rows), device=q.device, dtype=accumulation_dtype)
        accumulator = torch.zeros((batch_size, rows, d), device=q.device, dtype=accumulation_dtype)

        for k_start in range(0, n_keys, block_k):
            k_end = min(k_start + block_k, n_keys)
            k_block = k_flat[:, k_start:k_end].to(accumulation_dtype)
            v_block = v_flat[:, k_start:k_end].to(accumulation_dtype)
            scores = torch.matmul(q_block, k_block.transpose(-1, -2)) * scale

            if is_causal:
                q_positions = torch.arange(q_start, q_end, device=q.device)
                k_positions = torch.arange(k_start, k_end, device=q.device)
                scores = scores.masked_fill(q_positions[:, None] < k_positions[None, :], -torch.inf)

            block_max = scores.amax(dim=-1)
            new_max = torch.maximum(row_max, block_max)
            old_scale = torch.exp(row_max - new_max)
            probabilities = torch.exp(scores - new_max.unsqueeze(-1))
            new_sum = old_scale * row_sum + probabilities.sum(dim=-1)
            accumulator = old_scale.unsqueeze(-1) * accumulator + torch.matmul(probabilities, v_block)
            row_max = new_max
            row_sum = new_sum

        output[:, q_start:q_end] = (accumulator / row_sum.unsqueeze(-1)).to(q.dtype)
        lse[:, q_start:q_end] = row_max + torch.log(row_sum)

    output_shape = (*q.shape[:-2], n_queries, d)
    lse_shape = (*q.shape[:-2], n_queries)
    return output.reshape(output_shape), lse.reshape(lse_shape)


def _flash_backward_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    accumulation_dtype = _accumulation_dtype(q)
    q_acc = q.to(accumulation_dtype)
    k_acc = k.to(accumulation_dtype)
    v_acc = v.to(accumulation_dtype)
    output_acc = output.to(accumulation_dtype)
    grad_output_acc = grad_output.to(accumulation_dtype)
    lse_acc = lse.to(accumulation_dtype)
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.matmul(q_acc, k_acc.transpose(-1, -2)) * scale

    if is_causal:
        q_positions = torch.arange(n_queries, device=q.device)
        k_positions = torch.arange(n_keys, device=q.device)
        scores = scores.masked_fill(q_positions[:, None] < k_positions[None, :], -torch.inf)

    probabilities = torch.exp(scores - lse_acc.unsqueeze(-1))
    delta = (output_acc * grad_output_acc).sum(dim=-1)
    grad_probabilities = torch.matmul(grad_output_acc, v_acc.transpose(-1, -2))
    grad_scores = probabilities * (grad_probabilities - delta.unsqueeze(-1))
    grad_q = torch.matmul(grad_scores, k_acc) * scale
    grad_k = torch.matmul(grad_scores.transpose(-1, -2), q_acc) * scale
    grad_v = torch.matmul(probabilities.transpose(-1, -2), grad_output_acc)
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


_flash_backward_compiled = torch.compile(_flash_backward_pytorch, dynamic=True)


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        output, lse = _flash_forward_pytorch(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward_compiled(q, k, v, output, grad_output, lse, ctx.is_causal)
        return grad_q, grad_k, grad_v, None


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
        softmax_scale,
        N_QUERIES: tl.constexpr,
        N_KEYS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        q_offsets = batch * stride_qb + query_offsets[:, None] * stride_qq + dim_offsets[None, :] * stride_qd
        q = tl.load(q_ptr + q_offsets, mask=(query_offsets[:, None] < N_QUERIES) & (dim_offsets[None, :] < HEAD_DIM), other=0.0)

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        key_end = N_KEYS
        if CAUSAL:
            # A query can only see keys at or before its position. Truncate the
            # loop at this query tile's last real row; the element-wise mask
            # below still handles the causal diagonal and ragged final tiles.
            key_end = tl.minimum(key_end, tl.minimum(N_QUERIES, (query_block + 1) * BLOCK_M))
        for key_start in tl.range(0, key_end, BLOCK_N):
            current_keys = key_start + key_offsets
            k_offsets = batch * stride_kb + current_keys[:, None] * stride_kk + dim_offsets[None, :] * stride_kd
            v_offsets = batch * stride_vb + current_keys[:, None] * stride_vk + dim_offsets[None, :] * stride_vd
            kv_mask = (current_keys[:, None] < N_KEYS) & (dim_offsets[None, :] < HEAD_DIM)
            k = tl.load(k_ptr + k_offsets, mask=kv_mask, other=0.0)
            v = tl.load(v_ptr + v_offsets, mask=kv_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale
            valid_queries = query_offsets[:, None] < N_QUERIES
            valid_keys = current_keys[None, :] < N_KEYS
            if CAUSAL:
                scores = tl.where(valid_keys & (query_offsets[:, None] >= current_keys[None, :]), scores, -float("inf"))
            scores = tl.where(valid_keys, scores, -float("inf"))
            # Invalid query rows are never stored. Giving them finite values avoids
            # undefined -inf - -inf arithmetic in the online-softmax update.
            scores = tl.where(valid_queries, scores, 0.0)

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            old_scale = tl.exp(row_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            row_sum = row_sum * old_scale + tl.sum(probabilities, axis=1)
            accumulator = tl.dot(probabilities.to(v.dtype), v, acc=accumulator * old_scale[:, None], input_precision="ieee")
            row_max = new_max

        output = accumulator / row_sum[:, None]
        output_offsets = batch * stride_ob + query_offsets[:, None] * stride_oq + dim_offsets[None, :] * stride_od
        output_mask = (query_offsets[:, None] < N_QUERIES) & (dim_offsets[None, :] < HEAD_DIM)
        tl.store(output_ptr + output_offsets, output, mask=output_mask)
        lse_offsets = batch * stride_lb + query_offsets * stride_lq
        tl.store(lse_ptr + lse_offsets, row_max + tl.log(row_sum), mask=query_offsets < N_QUERIES)

    @triton.jit
    def _flash_backward_preprocess_kernel(
        output_ptr,
        grad_output_ptr,
        delta_ptr,
        N_QUERIES: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Compute the row-wise softmax correction ``sum_d O[d] * dO[d]``."""
        query_block = tl.program_id(0)
        batch = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, BLOCK_D)
        batch_offset = (batch * N_QUERIES * HEAD_DIM).to(tl.int64)
        offsets = batch_offset + query_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
        mask = (query_offsets[:, None] < N_QUERIES) & (dim_offsets[None, :] < HEAD_DIM)

        # Make the reduction explicitly FP32 even when O and dO are BF16/FP16.
        output = tl.load(output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        grad_output = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        delta = tl.sum(output * grad_output, axis=1)
        delta_offsets = (batch * N_QUERIES).to(tl.int64) + query_offsets
        tl.store(delta_ptr + delta_offsets, delta, mask=query_offsets < N_QUERIES)

    @triton.jit
    def _flash_backward_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_q_ptr,
        softmax_scale,
        N_QUERIES: tl.constexpr,
        N_KEYS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Own one query tile and stream over K/V tiles to produce dQ."""
        query_block = tl.program_id(0)
        batch = tl.program_id(1)
        query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        key_offsets = tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        q_batch_offset = (batch * N_QUERIES * HEAD_DIM).to(tl.int64)
        kv_batch_offset = (batch * N_KEYS * HEAD_DIM).to(tl.int64)
        row_batch_offset = (batch * N_QUERIES).to(tl.int64)
        q_offsets = q_batch_offset + query_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
        q_mask = (query_offsets[:, None] < N_QUERIES) & (dim_offsets[None, :] < HEAD_DIM)
        q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        grad_output = tl.load(grad_output_ptr + q_offsets, mask=q_mask, other=0.0)
        lse = tl.load(lse_ptr + row_batch_offset + query_offsets, mask=query_offsets < N_QUERIES, other=0.0)
        delta = tl.load(delta_ptr + row_batch_offset + query_offsets, mask=query_offsets < N_QUERIES, other=0.0)
        grad_q = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        # No tensor in this loop is larger than a BLOCK_M x BLOCK_N attention
        # tile. In particular, scores and probabilities are never materialized
        # with shape N_QUERIES x N_KEYS.
        key_end = N_KEYS
        if CAUSAL:
            key_end = tl.minimum(key_end, tl.minimum(N_QUERIES, (query_block + 1) * BLOCK_M))
        for key_start in tl.range(0, key_end, BLOCK_N):
            current_keys = key_start + key_offsets
            kv_offsets = kv_batch_offset + current_keys[:, None] * HEAD_DIM + dim_offsets[None, :]
            kv_mask = (current_keys[:, None] < N_KEYS) & (dim_offsets[None, :] < HEAD_DIM)
            k = tl.load(k_ptr + kv_offsets, mask=kv_mask, other=0.0)
            v = tl.load(v_ptr + kv_offsets, mask=kv_mask, other=0.0)

            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale
            attention_mask = (query_offsets[:, None] < N_QUERIES) & (current_keys[None, :] < N_KEYS)
            if CAUSAL:
                attention_mask = attention_mask & (query_offsets[:, None] >= current_keys[None, :])
            scores = tl.where(attention_mask, scores, -float("inf"))
            probabilities = tl.exp(scores - lse[:, None])
            grad_probabilities = tl.dot(grad_output, tl.trans(v), input_precision="ieee")
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            scaled_grad_scores = grad_scores * softmax_scale
            grad_q = tl.dot(scaled_grad_scores.to(k.dtype), k, acc=grad_q, input_precision="ieee")

        tl.store(grad_q_ptr + q_offsets, grad_q, mask=q_mask)

    @triton.jit
    def _flash_backward_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        grad_output_ptr,
        lse_ptr,
        delta_ptr,
        grad_k_ptr,
        grad_v_ptr,
        softmax_scale,
        N_QUERIES: tl.constexpr,
        N_KEYS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Own one key tile and stream over Q/dO tiles to produce dK and dV."""
        key_block = tl.program_id(0)
        batch = tl.program_id(1)
        query_offsets = tl.arange(0, BLOCK_M)
        key_offsets = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
        dim_offsets = tl.arange(0, BLOCK_D)

        q_batch_offset = (batch * N_QUERIES * HEAD_DIM).to(tl.int64)
        kv_batch_offset = (batch * N_KEYS * HEAD_DIM).to(tl.int64)
        row_batch_offset = (batch * N_QUERIES).to(tl.int64)
        kv_offsets = kv_batch_offset + key_offsets[:, None] * HEAD_DIM + dim_offsets[None, :]
        kv_mask = (key_offsets[:, None] < N_KEYS) & (dim_offsets[None, :] < HEAD_DIM)
        k = tl.load(k_ptr + kv_offsets, mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + kv_offsets, mask=kv_mask, other=0.0)
        grad_k = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
        grad_v = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)

        query_start = 0
        if CAUSAL:
            # No query before the first key in this tile can contribute to its
            # dK/dV. BLOCK_N is a multiple of BLOCK_M for every launch below,
            # so the reduced start remains aligned to a query tile.
            query_start = key_block * BLOCK_N
        for query_start in tl.range(query_start, N_QUERIES, BLOCK_M):
            current_queries = query_start + query_offsets
            q_offsets = q_batch_offset + current_queries[:, None] * HEAD_DIM + dim_offsets[None, :]
            q_mask = (current_queries[:, None] < N_QUERIES) & (dim_offsets[None, :] < HEAD_DIM)
            q = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
            grad_output = tl.load(grad_output_ptr + q_offsets, mask=q_mask, other=0.0)
            lse = tl.load(lse_ptr + row_batch_offset + current_queries, mask=current_queries < N_QUERIES, other=0.0)
            delta = tl.load(delta_ptr + row_batch_offset + current_queries, mask=current_queries < N_QUERIES, other=0.0)

            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale
            attention_mask = (current_queries[:, None] < N_QUERIES) & (key_offsets[None, :] < N_KEYS)
            if CAUSAL:
                attention_mask = attention_mask & (current_queries[:, None] >= key_offsets[None, :])
            scores = tl.where(attention_mask, scores, -float("inf"))
            probabilities = tl.exp(scores - lse[:, None])
            grad_probabilities = tl.dot(grad_output, tl.trans(v), input_precision="ieee")
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            scaled_grad_scores = grad_scores * softmax_scale
            grad_k = tl.dot(tl.trans(scaled_grad_scores.to(q.dtype)), q, acc=grad_k, input_precision="ieee")
            grad_v = tl.dot(tl.trans(probabilities.to(grad_output.dtype)), grad_output, acc=grad_v, input_precision="ieee")

        tl.store(grad_k_ptr + kv_offsets, grad_k, mask=kv_mask)
        tl.store(grad_v_ptr + kv_offsets, grad_v, mask=kv_mask)


def _flash_forward_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None:
        raise RuntimeError("Triton is required for the Triton FlashAttention implementation")
    _validate_inputs(q, k, v)
    if not q.is_cuda:
        raise ValueError("the Triton FlashAttention implementation requires CUDA tensors")

    q_flat = _flatten_batch(q).contiguous()
    k_flat = _flatten_batch(k).contiguous()
    v_flat = _flatten_batch(v).contiguous()
    batch_size, n_queries, d = q_flat.shape
    n_keys = k_flat.shape[1]
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("the Triton kernel supports float16, bfloat16, and float32 inputs")
    if d > 128:
        raise ValueError("the Triton kernel supports embedding dimensions up to 128")

    output = torch.empty_like(q_flat)
    lse = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
    if batch_size == 0 or n_queries == 0:
        return output.reshape_as(q), lse.reshape(*q.shape[:-2], n_queries)

    block_d = max(16, triton.next_power_of_2(d))
    # FP32 operands occupy twice the shared memory of the BF16 performance
    # path. Smaller correctness-only tiles keep a padded D-block of 128 below
    # Ada's 99 KiB limit without changing the required BF16 benchmark kernel.
    if q.dtype == torch.float32 and block_d == 128:
        block_m, block_n = 32, 32
    else:
        block_m = TRITON_CONFIG["block_m"]
        block_n = TRITON_CONFIG["block_n"]
    grid = (triton.cdiv(n_queries, block_m), batch_size)
    _flash_forward_kernel[grid](
        q_flat,
        k_flat,
        v_flat,
        output,
        lse,
        *q_flat.stride(),
        *k_flat.stride(),
        *v_flat.stride(),
        *output.stride(),
        *lse.stride(),
        1.0 / math.sqrt(d),
        N_QUERIES=n_queries,
        N_KEYS=n_keys,
        HEAD_DIM=d,
        CAUSAL=bool(is_causal),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=TRITON_CONFIG["num_warps"],
        num_stages=TRITON_CONFIG["num_stages"],
    )
    return output.reshape_as(q), lse.reshape(*q.shape[:-2], n_queries)


def _flash_backward_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a memory-linear, tiled Triton FlashAttention backward pass.

    The only auxiliary allocation is the FP32 ``delta`` vector with one value
    per query row. Each output-gradient tile has a unique owning program, so
    dK/dV do not require a quadratic score buffer or global atomic reductions.
    """
    if triton is None:
        raise RuntimeError("Triton is required for the Triton FlashAttention implementation")
    _validate_inputs(q, k, v)
    if not q.is_cuda:
        raise ValueError("the Triton FlashAttention implementation requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("the Triton kernel supports float16, bfloat16, and float32 inputs")
    if q.shape[-1] > 128:
        raise ValueError("the Triton kernel supports embedding dimensions up to 128")
    if output.shape != q.shape or grad_output.shape != q.shape:
        raise ValueError("output and grad_output must have the same shape as q")
    if output.device != q.device or grad_output.device != q.device or lse.device != q.device:
        raise ValueError("output, grad_output, and lse must be on the same device as q")
    if output.dtype != q.dtype or grad_output.dtype != q.dtype:
        raise ValueError("output and grad_output must have the same dtype as q")
    expected_lse_shape = (*q.shape[:-2], q.shape[-2])
    if lse.shape != expected_lse_shape or lse.dtype != torch.float32:
        raise ValueError("lse must have shape q.shape[:-1] and dtype float32")

    q_flat = _flatten_batch(q).contiguous()
    k_flat = _flatten_batch(k).contiguous()
    v_flat = _flatten_batch(v).contiguous()
    output_flat = _flatten_batch(output).contiguous()
    grad_output_flat = _flatten_batch(grad_output).contiguous()
    batch_size, n_queries, d = q_flat.shape
    n_keys = k_flat.shape[1]

    # CUDA does not permit a zero-sized launch. The mathematically correct
    # gradients for an empty batch/query dimension are all zero.
    if batch_size == 0 or n_queries == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    lse_flat = lse.reshape(batch_size, n_queries).contiguous()
    delta = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
    grad_q = torch.empty_like(q_flat)
    grad_k = torch.empty_like(k_flat)
    grad_v = torch.empty_like(v_flat)
    block_d = max(16, triton.next_power_of_2(d))
    fp32_wide_inputs = q.dtype == torch.float32 and block_d == 128
    preprocess_block_m = 64 if fp32_wide_inputs else 128
    # Ada (RTX 4090) exposes less per-block shared memory than H100/H200.
    # BF16 uses 64-row ownership tiles. FP32 inputs padded to BLOCK_D=128 need
    # smaller tiles to remain below Ada's 99 KiB per-block shared-memory limit.
    if fp32_wide_inputs:
        dq_block_m, dq_block_n = 32, 32
        dkdv_block_m, dkdv_block_n = 32, 32
        backward_num_stages = 2
    else:
        dq_block_m, dq_block_n = 64, 32
        dkdv_block_m, dkdv_block_n = 32, 64
        backward_num_stages = 3

    preprocess_grid = (triton.cdiv(n_queries, preprocess_block_m), batch_size)
    _flash_backward_preprocess_kernel[preprocess_grid](
        output_flat,
        grad_output_flat,
        delta,
        N_QUERIES=n_queries,
        HEAD_DIM=d,
        BLOCK_M=preprocess_block_m,
        BLOCK_D=block_d,
        num_warps=4,
    )

    dq_grid = (triton.cdiv(n_queries, dq_block_m), batch_size)
    _flash_backward_dq_kernel[dq_grid](
        q_flat,
        k_flat,
        v_flat,
        grad_output_flat,
        lse_flat,
        delta,
        grad_q,
        1.0 / math.sqrt(d),
        N_QUERIES=n_queries,
        N_KEYS=n_keys,
        HEAD_DIM=d,
        CAUSAL=bool(is_causal),
        BLOCK_M=dq_block_m,
        BLOCK_N=dq_block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=backward_num_stages,
    )

    dkdv_grid = (triton.cdiv(n_keys, dkdv_block_n), batch_size)
    _flash_backward_dkdv_kernel[dkdv_grid](
        q_flat,
        k_flat,
        v_flat,
        grad_output_flat,
        lse_flat,
        delta,
        grad_k,
        grad_v,
        1.0 / math.sqrt(d),
        N_QUERIES=n_queries,
        N_KEYS=n_keys,
        HEAD_DIM=d,
        CAUSAL=bool(is_causal),
        BLOCK_M=dkdv_block_m,
        BLOCK_N=dkdv_block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=backward_num_stages,
    )
    return grad_q.reshape_as(q), grad_k.reshape_as(k), grad_v.reshape_as(v)


class FlashAttentionTriton(torch.autograd.Function):
    triton_config = TRITON_CONFIG

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        output, lse = _flash_forward_triton(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = _flash_backward_triton(q, k, v, output, grad_output, lse, ctx.is_causal)
        return grad_q, grad_k, grad_v, None


__all__ = [
    "FlashAttentionPyTorch",
    "FlashAttentionTriton",
    "TRITON_CONFIG",
    "eager_attention",
]
