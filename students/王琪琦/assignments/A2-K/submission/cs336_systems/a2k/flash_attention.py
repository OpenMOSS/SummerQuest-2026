from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """Unfused attention baseline that intentionally materializes scores."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if is_causal:
        qi = torch.arange(q.shape[-2], device=q.device)[:, None]
        ki = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(qi < ki, -torch.inf)
    return torch.softmax(scores, dim=-1) @ v


def _recomputed_backward(ctx, grad_o: torch.Tensor):
    q, k, v, _o, _lse = ctx.saved_tensors
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        output = explicit_attention(q_, k_, v_, ctx.is_causal)
        dq, dk, dv = torch.autograd.grad(output, (q_, k_, v_), grad_o)
    return dq, dk, dv, None


def _torch_tiled_forward(q, k, v, is_causal: bool, block_q: int = 64, block_k: int = 64):
    scale = 1.0 / math.sqrt(q.shape[-1])
    outputs, lses = [], []
    for q_start in range(0, q.shape[-2], block_q):
        q_tile = q[:, q_start : q_start + block_q].float()
        rows = q_tile.shape[-2]
        m = torch.full((q.shape[0], rows), -torch.inf, device=q.device)
        denominator = torch.zeros_like(m)
        accumulator = torch.zeros((*m.shape, v.shape[-1]), device=q.device)
        for k_start in range(0, k.shape[-2], block_k):
            k_tile = k[:, k_start : k_start + block_k].float()
            v_tile = v[:, k_start : k_start + block_k].float()
            scores = q_tile @ k_tile.transpose(-1, -2) * scale
            if is_causal:
                q_pos = torch.arange(q_start, q_start + rows, device=q.device)[:, None]
                k_pos = torch.arange(k_start, k_start + k_tile.shape[-2], device=q.device)[None, :]
                scores = scores.masked_fill(q_pos < k_pos, -torch.inf)
            new_m = torch.maximum(m, scores.amax(dim=-1))
            alpha = torch.exp(m - new_m)
            probabilities = torch.exp(scores - new_m[..., None])
            denominator = denominator * alpha + probabilities.sum(dim=-1)
            accumulator = accumulator * alpha[..., None] + probabilities @ v_tile
            m = new_m
        outputs.append((accumulator / denominator[..., None]).to(q.dtype))
        lses.append(m + torch.log(denominator))
    return torch.cat(outputs, dim=-2), torch.cat(lses, dim=-1)


class PyTorchFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        output, lse = _torch_tiled_forward(q, k, v, bool(is_causal))
        ctx.is_causal = bool(is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad_o):
        return _recomputed_backward(ctx, grad_o)


@triton.jit
def _flash_forward_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
    stride_qb: tl.constexpr, stride_qn: tl.constexpr, stride_qd: tl.constexpr,
    stride_kb: tl.constexpr, stride_kn: tl.constexpr, stride_kd: tl.constexpr,
    stride_vb: tl.constexpr, stride_vn: tl.constexpr, stride_vd: tl.constexpr,
    stride_ob: tl.constexpr, stride_on: tl.constexpr, stride_od: tl.constexpr,
    stride_lb: tl.constexpr, stride_ln: tl.constexpr,
    NQ: tl.constexpr, NK: tl.constexpr, D: tl.constexpr,
    SCALE: tl.constexpr, CAUSAL: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch = tl.program_id(1)
    q_offsets = query_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_offsets = tl.arange(0, D)
    q = tl.load(q_ptr + batch * stride_qb + q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd,
                mask=q_offsets[:, None] < NQ, other=0.0)
    running_max = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
    running_sum = tl.zeros((BLOCK_Q,), tl.float32)
    accumulator = tl.zeros((BLOCK_Q, D), tl.float32)
    for key_start in range(0, NK, BLOCK_K):
        key_offsets = key_start + tl.arange(0, BLOCK_K)
        k = tl.load(k_ptr + batch * stride_kb + key_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd,
                    mask=key_offsets[:, None] < NK, other=0.0)
        scores = tl.dot(q, tl.trans(k)) * SCALE
        valid = (q_offsets[:, None] < NQ) & (key_offsets[None, :] < NK)
        if CAUSAL:
            valid &= q_offsets[:, None] >= key_offsets[None, :]
        scores = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(running_max, block_max)
        correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
        v = tl.load(v_ptr + batch * stride_vb + key_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd,
                    mask=key_offsets[:, None] < NK, other=0.0)
        accumulator = accumulator * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
        running_max = new_max
    output = accumulator / running_sum[:, None]
    tl.store(o_ptr + batch * stride_ob + q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od,
             output, mask=q_offsets[:, None] < NQ)
    tl.store(l_ptr + batch * stride_lb + q_offsets * stride_ln,
             running_max + tl.log(running_sum), mask=q_offsets < NQ)


def _triton_forward(q, k, v, is_causal: bool):
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("Triton FlashAttention requires CUDA tensors")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected [batch, sequence, head_dim] tensors")
    if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
        raise ValueError("incompatible Q/K/V shapes")
    d = q.shape[-1]
    if d not in (16, 32, 64, 128):
        raise ValueError("head dimension must be one of 16, 32, 64, 128")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    output = torch.empty_like(q)
    lse = torch.empty((q.shape[0], q.shape[1]), device=q.device, dtype=torch.float32)
    block_q, block_k = 64, 64
    grid = (triton.cdiv(q.shape[1], block_q), q.shape[0])
    _flash_forward_kernel[grid](
        q, k, v, output, lse,
        *q.stride(), *k.stride(), *v.stride(), *output.stride(), *lse.stride(),
        NQ=q.shape[1], NK=k.shape[1], D=d, SCALE=1.0 / math.sqrt(d),
        CAUSAL=is_causal, BLOCK_Q=block_q, BLOCK_K=block_k,
        num_warps=4, num_stages=2,
    )
    return output, lse


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        output, lse = _triton_forward(q, k, v, bool(is_causal))
        ctx.is_causal = bool(is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        return output

    @staticmethod
    def backward(ctx, grad_o):
        return _recomputed_backward(ctx, grad_o)
