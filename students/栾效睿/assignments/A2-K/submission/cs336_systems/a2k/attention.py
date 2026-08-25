from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


PYTORCH_TILE_SIZE = 128
TRITON_TILE_SIZE = 64
TRITON_NUM_WARPS = 4
TRITON_NUM_STAGES = 2
TRITON_FP32_TILE_SIZE = 32
TRITON_FP32_NUM_WARPS = 2
TRITON_FP32_NUM_STAGES = 1


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != k.ndim or q.ndim != v.ndim or q.ndim != 3 or q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("Q, K, and V must have shapes [batch, sequence, head_dim], [batch, keys, head_dim], and [batch, keys, head_dim]")


def _causal_scores(scores: torch.Tensor, query_offset: int, key_offset: int) -> torch.Tensor:
    queries = torch.arange(query_offset, query_offset + scores.shape[-2], device=scores.device)[:, None]
    keys = torch.arange(key_offset, key_offset + scores.shape[-1], device=scores.device)[None, :]
    return scores.masked_fill(queries < keys, -1_000_000.0)


def _flash_forward_tiled(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    batch, n_queries, d = q.shape
    output, lse, scale = torch.empty_like(q), torch.empty((batch, n_queries), device=q.device, dtype=torch.float32), d**-0.5
    for query_start in range(0, n_queries, PYTORCH_TILE_SIZE):
        q_tile, m, normalizer, acc = q[:, query_start : query_start + PYTORCH_TILE_SIZE].float(), torch.full((batch, min(PYTORCH_TILE_SIZE, n_queries - query_start)), -math.inf, device=q.device, dtype=torch.float32), torch.zeros((batch, min(PYTORCH_TILE_SIZE, n_queries - query_start)), device=q.device, dtype=torch.float32), torch.zeros((batch, min(PYTORCH_TILE_SIZE, n_queries - query_start), d), device=q.device, dtype=torch.float32)
        for key_start in range(0, k.shape[1], PYTORCH_TILE_SIZE):
            k_tile, v_tile = k[:, key_start : key_start + PYTORCH_TILE_SIZE].float(), v[:, key_start : key_start + PYTORCH_TILE_SIZE].float()
            scores = q_tile @ k_tile.transpose(-2, -1) * scale
            scores = _causal_scores(scores, query_start, key_start) if is_causal else scores
            next_m = torch.maximum(m, scores.amax(dim=-1))
            p, alpha = torch.exp(scores - next_m[..., None]), torch.exp(m - next_m)
            normalizer, acc, m = normalizer * alpha + p.sum(dim=-1), acc * alpha[..., None] + p @ v_tile, next_m
        output[:, query_start : query_start + q_tile.shape[1]], lse[:, query_start : query_start + q_tile.shape[1]] = (acc / normalizer[..., None]).to(q.dtype), m + normalizer.log()
    return output, lse


def flash_attention_backward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = q.shape[-1] ** -0.5
    dq, dk, dv, delta = torch.zeros_like(q, dtype=torch.float32), torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32), (o.float() * do.float()).sum(dim=-1)
    for key_start in range(0, k.shape[1], PYTORCH_TILE_SIZE):
        k_tile, v_tile = k[:, key_start : key_start + PYTORCH_TILE_SIZE].float(), v[:, key_start : key_start + PYTORCH_TILE_SIZE].float()
        for query_start in range(0, q.shape[1], PYTORCH_TILE_SIZE):
            q_tile, do_tile, l_tile = q[:, query_start : query_start + PYTORCH_TILE_SIZE].float(), do[:, query_start : query_start + PYTORCH_TILE_SIZE].float(), lse[:, query_start : query_start + PYTORCH_TILE_SIZE]
            scores = q_tile @ k_tile.transpose(-2, -1) * scale
            scores = _causal_scores(scores, query_start, key_start) if is_causal else scores
            p = torch.exp(scores - l_tile[..., None])
            ds = p * (do_tile @ v_tile.transpose(-2, -1) - delta[:, query_start : query_start + q_tile.shape[1], None])
            dq[:, query_start : query_start + q_tile.shape[1]] += ds @ k_tile * scale
            dk[:, key_start : key_start + k_tile.shape[1]] += ds.transpose(-2, -1) @ q_tile * scale
            dv[:, key_start : key_start + v_tile.shape[1]] += p.transpose(-2, -1) @ do_tile
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


if triton is not None:

    @triton.jit
    def flash_fwd_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        stride_qb: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
        stride_kb: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
        stride_vb: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
        stride_ob: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
        stride_lb: tl.constexpr, stride_lq: tl.constexpr,
        n_queries: tl.constexpr, n_keys: tl.constexpr, scale: tl.constexpr,
        d: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr, is_causal: tl.constexpr,
    ):
        query_tile, batch = tl.program_id(0), tl.program_id(1)
        offsets_m, offsets_n, offsets_d = query_tile * block_m + tl.arange(0, block_m), tl.arange(0, block_n), tl.arange(0, d)
        q_mask = offsets_m < n_queries
        q = tl.load(q_ptr + batch * stride_qb + offsets_m[:, None] * stride_qq + offsets_d[None, :] * stride_qd, mask=q_mask[:, None], other=0.0)
        m, normalizer, acc = tl.full((block_m,), -float("inf"), tl.float32), tl.zeros((block_m,), tl.float32), tl.zeros((block_m, d), tl.float32)
        for key_start in range(0, n_keys, block_n):
            key_offsets, key_mask = key_start + offsets_n, key_start + offsets_n < n_keys
            k = tl.load(k_ptr + batch * stride_kb + key_offsets[:, None] * stride_kk + offsets_d[None, :] * stride_kd, mask=key_mask[:, None], other=0.0)
            v = tl.load(v_ptr + batch * stride_vb + key_offsets[:, None] * stride_vk + offsets_d[None, :] * stride_vd, mask=key_mask[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            valid = key_mask[None, :] & (offsets_m[:, None] >= key_offsets[None, :]) if is_causal else key_mask[None, :]
            scores = tl.where(valid, scores, -1.0e6)
            next_m = tl.maximum(m, tl.max(scores, axis=1))
            p, alpha = tl.exp(scores - next_m[:, None]), tl.exp(m - next_m)
            normalizer = normalizer * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(v.dtype), v, acc=acc)
            m = next_m
        output = acc / normalizer[:, None]
        tl.store(o_ptr + batch * stride_ob + offsets_m[:, None] * stride_oq + offsets_d[None, :] * stride_od, output.to(q.dtype), mask=q_mask[:, None])
        tl.store(l_ptr + batch * stride_lb + offsets_m * stride_lq, m + tl.log(normalizer), mask=q_mask)


    @triton.jit
    def flash_bwd_delta_kernel(
        o_ptr, do_ptr, delta_ptr,
        stride_ob: tl.constexpr, stride_oq: tl.constexpr, stride_od: tl.constexpr,
        stride_dob: tl.constexpr, stride_doq: tl.constexpr, stride_dod: tl.constexpr,
        stride_db: tl.constexpr, stride_dq: tl.constexpr,
        n_queries: tl.constexpr, d: tl.constexpr, block_m: tl.constexpr,
    ):
        query_tile, batch = tl.program_id(0), tl.program_id(1)
        offsets_m, offsets_d = query_tile * block_m + tl.arange(0, block_m), tl.arange(0, d)
        mask = offsets_m < n_queries
        o = tl.load(o_ptr + batch * stride_ob + offsets_m[:, None] * stride_oq + offsets_d[None, :] * stride_od, mask=mask[:, None], other=0.0)
        do = tl.load(do_ptr + batch * stride_dob + offsets_m[:, None] * stride_doq + offsets_d[None, :] * stride_dod, mask=mask[:, None], other=0.0)
        tl.store(delta_ptr + batch * stride_db + offsets_m * stride_dq, tl.sum(o.to(tl.float32) * do.to(tl.float32), axis=1), mask=mask)


    @triton.jit
    def flash_bwd_dkdv_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, delta_ptr, dk_ptr, dv_ptr,
        stride_qb: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
        stride_kb: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
        stride_vb: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
        stride_dob: tl.constexpr, stride_doq: tl.constexpr, stride_dod: tl.constexpr,
        stride_lb: tl.constexpr, stride_lq: tl.constexpr, stride_db: tl.constexpr, stride_dq: tl.constexpr,
        stride_dkb: tl.constexpr, stride_dkk: tl.constexpr, stride_dkd: tl.constexpr,
        stride_dvb: tl.constexpr, stride_dvk: tl.constexpr, stride_dvd: tl.constexpr,
        n_queries: tl.constexpr, n_keys: tl.constexpr, scale: tl.constexpr,
        d: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr, is_causal: tl.constexpr,
    ):
        key_tile, batch = tl.program_id(0), tl.program_id(1)
        offsets_m, offsets_n, offsets_d = tl.arange(0, block_m), key_tile * block_n + tl.arange(0, block_n), tl.arange(0, d)
        key_mask = offsets_n < n_keys
        k = tl.load(k_ptr + batch * stride_kb + offsets_n[:, None] * stride_kk + offsets_d[None, :] * stride_kd, mask=key_mask[:, None], other=0.0)
        v = tl.load(v_ptr + batch * stride_vb + offsets_n[:, None] * stride_vk + offsets_d[None, :] * stride_vd, mask=key_mask[:, None], other=0.0)
        dk, dv = tl.zeros((block_n, d), tl.float32), tl.zeros((block_n, d), tl.float32)
        for query_start in range(0, n_queries, block_m):
            query_offsets, query_mask = query_start + offsets_m, query_start + offsets_m < n_queries
            q = tl.load(q_ptr + batch * stride_qb + query_offsets[:, None] * stride_qq + offsets_d[None, :] * stride_qd, mask=query_mask[:, None], other=0.0)
            do = tl.load(do_ptr + batch * stride_dob + query_offsets[:, None] * stride_doq + offsets_d[None, :] * stride_dod, mask=query_mask[:, None], other=0.0)
            lse = tl.load(l_ptr + batch * stride_lb + query_offsets * stride_lq, mask=query_mask, other=0.0)
            delta = tl.load(delta_ptr + batch * stride_db + query_offsets * stride_dq, mask=query_mask, other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            valid = query_mask[:, None] & key_mask[None, :] & (query_offsets[:, None] >= offsets_n[None, :]) if is_causal else query_mask[:, None] & key_mask[None, :]
            p = tl.exp(tl.where(valid, scores, -1.0e6) - lse[:, None])
            ds = p * (tl.dot(do, tl.trans(v), input_precision="ieee") - delta[:, None])
            dk = tl.dot(tl.trans(ds).to(q.dtype), q, acc=dk)
            dv = tl.dot(tl.trans(p).to(do.dtype), do, acc=dv)
        tl.store(dk_ptr + batch * stride_dkb + offsets_n[:, None] * stride_dkk + offsets_d[None, :] * stride_dkd, (dk * scale).to(k.dtype), mask=key_mask[:, None])
        tl.store(dv_ptr + batch * stride_dvb + offsets_n[:, None] * stride_dvk + offsets_d[None, :] * stride_dvd, dv.to(v.dtype), mask=key_mask[:, None])


    @triton.jit
    def flash_bwd_dq_kernel(
        q_ptr, k_ptr, v_ptr, do_ptr, l_ptr, delta_ptr, dq_ptr,
        stride_qb: tl.constexpr, stride_qq: tl.constexpr, stride_qd: tl.constexpr,
        stride_kb: tl.constexpr, stride_kk: tl.constexpr, stride_kd: tl.constexpr,
        stride_vb: tl.constexpr, stride_vk: tl.constexpr, stride_vd: tl.constexpr,
        stride_dob: tl.constexpr, stride_doq: tl.constexpr, stride_dod: tl.constexpr,
        stride_lb: tl.constexpr, stride_lq: tl.constexpr, stride_db: tl.constexpr, stride_dq: tl.constexpr,
        stride_dqb: tl.constexpr, stride_dqq: tl.constexpr, stride_dqd: tl.constexpr,
        n_queries: tl.constexpr, n_keys: tl.constexpr, scale: tl.constexpr,
        d: tl.constexpr, block_m: tl.constexpr, block_n: tl.constexpr, is_causal: tl.constexpr,
    ):
        query_tile, batch = tl.program_id(0), tl.program_id(1)
        offsets_m, offsets_n, offsets_d = query_tile * block_m + tl.arange(0, block_m), tl.arange(0, block_n), tl.arange(0, d)
        query_mask = offsets_m < n_queries
        q = tl.load(q_ptr + batch * stride_qb + offsets_m[:, None] * stride_qq + offsets_d[None, :] * stride_qd, mask=query_mask[:, None], other=0.0)
        do = tl.load(do_ptr + batch * stride_dob + offsets_m[:, None] * stride_doq + offsets_d[None, :] * stride_dod, mask=query_mask[:, None], other=0.0)
        lse = tl.load(l_ptr + batch * stride_lb + offsets_m * stride_lq, mask=query_mask, other=0.0)
        delta = tl.load(delta_ptr + batch * stride_db + offsets_m * stride_dq, mask=query_mask, other=0.0)
        dq = tl.zeros((block_m, d), tl.float32)
        for key_start in range(0, n_keys, block_n):
            key_offsets, key_mask = key_start + offsets_n, key_start + offsets_n < n_keys
            k = tl.load(k_ptr + batch * stride_kb + key_offsets[:, None] * stride_kk + offsets_d[None, :] * stride_kd, mask=key_mask[:, None], other=0.0)
            v = tl.load(v_ptr + batch * stride_vb + key_offsets[:, None] * stride_vk + offsets_d[None, :] * stride_vd, mask=key_mask[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            valid = query_mask[:, None] & key_mask[None, :] & (offsets_m[:, None] >= key_offsets[None, :]) if is_causal else query_mask[:, None] & key_mask[None, :]
            p = tl.exp(tl.where(valid, scores, -1.0e6) - lse[:, None])
            ds = p * (tl.dot(do, tl.trans(v), input_precision="ieee") - delta[:, None])
            dq = tl.dot(ds.to(k.dtype), k, acc=dq)
        tl.store(dq_ptr + batch * stride_dqb + offsets_m[:, None] * stride_dqq + offsets_d[None, :] * stride_dqd, (dq * scale).to(q.dtype), mask=query_mask[:, None])


def _triton_config(length: int, dtype: torch.dtype) -> tuple[int, int, int, int]:
    tile_limit = TRITON_FP32_TILE_SIZE if dtype == torch.float32 else TRITON_TILE_SIZE
    tile = min(tile_limit, 1 << (length.bit_length() - 1))
    num_warps, num_stages = (TRITON_FP32_NUM_WARPS, TRITON_FP32_NUM_STAGES) if dtype == torch.float32 else (TRITON_NUM_WARPS, TRITON_NUM_STAGES)
    return tile, tile, num_warps, num_stages


def flash_attention_backward_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, o: torch.Tensor, do: torch.Tensor, lse: torch.Tensor, is_causal: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if triton is None or not q.is_cuda:
        raise RuntimeError("FlashAttentionTriton requires CUDA and the Triton package")
    block_m, _, num_warps, num_stages = _triton_config(q.shape[1], q.dtype)
    block_n = _triton_config(k.shape[1], k.dtype)[0]
    delta, dq, dk, dv = torch.empty(q.shape[:2], device=q.device, dtype=torch.float32), torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    flash_bwd_delta_kernel[(triton.cdiv(q.shape[1], block_m), q.shape[0])](o, do, delta, *o.stride(), *do.stride(), *delta.stride(), q.shape[1], q.shape[-1], block_m, num_warps=num_warps, num_stages=num_stages)
    flash_bwd_dkdv_kernel[(triton.cdiv(k.shape[1], block_n), q.shape[0])](q, k, v, do, lse, delta, dk, dv, *q.stride(), *k.stride(), *v.stride(), *do.stride(), *lse.stride(), *delta.stride(), *dk.stride(), *dv.stride(), q.shape[1], k.shape[1], q.shape[-1] ** -0.5, q.shape[-1], block_m, block_n, is_causal, num_warps=num_warps, num_stages=num_stages)
    flash_bwd_dq_kernel[(triton.cdiv(q.shape[1], block_m), q.shape[0])](q, k, v, do, lse, delta, dq, *q.stride(), *k.stride(), *v.stride(), *do.stride(), *lse.stride(), *delta.stride(), *dq.stride(), q.shape[1], k.shape[1], q.shape[-1] ** -0.5, q.shape[-1], block_m, block_n, is_causal, num_warps=num_warps, num_stages=num_stages)
    return dq, dk, dv


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        _validate_inputs(q, k, v)
        o, lse = _flash_forward_tiled(q, k, v, is_causal)
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, o, lse = ctx.saved_tensors
        return *flash_attention_backward(q, k, v, o, do, lse, ctx.is_causal), None


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        _validate_inputs(q, k, v)
        if triton is None or not q.is_cuda:
            raise RuntimeError("FlashAttentionTriton requires CUDA and the Triton package")
        block_m, _, num_warps, num_stages = _triton_config(q.shape[1], q.dtype)
        block_n = _triton_config(k.shape[1], k.dtype)[0]
        o, lse = torch.empty_like(q), torch.empty(q.shape[:2], device=q.device, dtype=torch.float32)
        flash_fwd_kernel[(triton.cdiv(q.shape[1], block_m), q.shape[0])](
            q, k, v, o, lse,
            *q.stride(), *k.stride(), *v.stride(), *o.stride(), *lse.stride(),
            q.shape[1], k.shape[1], q.shape[-1] ** -0.5, q.shape[-1], block_m, block_n, is_causal,
            num_warps=num_warps, num_stages=num_stages,
        )
        ctx.is_causal = is_causal
        ctx.save_for_backward(q, k, v, o, lse)
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, o, lse = ctx.saved_tensors
        return *flash_attention_backward_triton(q, k, v, o, do, lse, ctx.is_causal), None
