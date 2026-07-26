"""FlashAttention-2 的教学实现：PyTorch 分块参考、Triton 前向与重计算反向。

输入遵循作业测试使用的三维布局：

* ``Q``: ``[batch, query_length, d]``
* ``K``: ``[batch, key_length, d]``
* ``V``: ``[batch, key_length, d]``

普通 Attention 会在全局显存中物化 ``[batch, query_length, key_length]`` 的
分数和概率矩阵。这里的两个 forward 都固定一个 query tile，依次流过所有
key/value tile，每个 query 行只维护运行最大值 ``m``、指数和 ``l`` 与未归一化
输出累加器 ``acc``。因此 forward 的长期中间状态只随序列长度线性增长。

课程规定 backward 可由 PyTorch 公式配合 ``torch.compile`` 完成，所以本文件
没有把可选的 Triton backward 冒充成必做内容。反向会由 ``Q/K/V/O/L`` 重算概率，
并用 ``D = rowsum(O * dO)`` 避免显式构造 softmax Jacobian。
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable

import torch

from cs336_basics.nn_utils import softmax as basics_softmax


try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # Windows/CPU 本地环境仍需能够导入并测试 PyTorch 参考版。
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """物化完整分数矩阵的标准 Attention，作为正确性和性能基线。"""

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        query_index = torch.arange(q.shape[-2], device=q.device)[:, None]
        key_index = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = torch.where(query_index >= key_index, scores, -1e6)
    # 使用项目 A1 的 max/exp/sum/div softmax。这样普通 Attention benchmark
    # 与 Transformer 模型中的真实 baseline 一致，而不是悄悄换成框架融合算子。
    probabilities = basics_softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def pytorch_sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """调用 PyTorch 内置 SDPA，便于和框架自带 FlashAttention 路径对照。"""

    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def tiled_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    query_tile_size: int = 64,
    key_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用普通 PyTorch 操作直接翻译 FlashAttention-2 在线 softmax。

    这段代码追求可读性和数值可核对性，而不是速度。Python 两层循环会产生
    大量小算子，因此它通常比标准 Attention 更慢；它的价值是把 Triton kernel
    中的 ``m/l/acc`` 更新规则写成容易逐行调试的参考答案。
    """

    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("FlashAttention expects Q, K, V with shape [batch, sequence, d]")
    if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
        raise ValueError("Q, K, V must have the same batch size")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("this A2 kernel requires Q, K, V to use the same head dimension")

    batch_size, n_queries, d = q.shape
    n_keys = k.shape[-2]
    scale = 1.0 / math.sqrt(d)
    output_tiles: list[torch.Tensor] = []
    lse_tiles: list[torch.Tensor] = []

    # 在线 softmax 的统计量必须用 FP32。BF16 只用于输入存储和最终写回，
    # 否则长序列上的指数和会快速积累舍入误差。
    q_fp32 = q.float()
    k_fp32 = k.float()
    v_fp32 = v.float()

    for query_start in range(0, n_queries, query_tile_size):
        query_end = min(query_start + query_tile_size, n_queries)
        query_tile = q_fp32[:, query_start:query_end]
        tile_rows = query_end - query_start

        running_max = torch.full(
            (batch_size, tile_rows),
            -torch.inf,
            device=q.device,
            dtype=torch.float32,
        )
        running_sum = torch.zeros_like(running_max)
        output_accumulator = torch.zeros(
            (batch_size, tile_rows, d),
            device=q.device,
            dtype=torch.float32,
        )

        for key_start in range(0, n_keys, key_tile_size):
            key_end = min(key_start + key_tile_size, n_keys)
            key_tile = k_fp32[:, key_start:key_end]
            value_tile = v_fp32[:, key_start:key_end]
            scores = torch.matmul(query_tile, key_tile.transpose(-2, -1)) * scale

            if is_causal:
                query_index = torch.arange(query_start, query_end, device=q.device)[:, None]
                key_index = torch.arange(key_start, key_end, device=q.device)[None, :]
                scores = torch.where(query_index >= key_index, scores, -1e6)

            tile_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, tile_max)

            # 旧的 running_sum/acc 是相对于旧最大值计算的。出现更大分数后，
            # 必须先乘 exp(old_max - new_max) 才能和新 tile 放在同一尺度相加。
            old_rescale = torch.exp(running_max - new_max)
            unnormalized_probabilities = torch.exp(scores - new_max.unsqueeze(-1))
            running_sum = old_rescale * running_sum + unnormalized_probabilities.sum(dim=-1)
            output_accumulator = (
                old_rescale.unsqueeze(-1) * output_accumulator
                + torch.matmul(unnormalized_probabilities, value_tile)
            )
            running_max = new_max

        output_tiles.append((output_accumulator / running_sum.unsqueeze(-1)).to(v.dtype))
        lse_tiles.append(running_max + torch.log(running_sum))

    return torch.cat(output_tiles, dim=-2), torch.cat(lse_tiles, dim=-1)


def _flash_backward_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按讲义公式 13-19 重算概率并返回 dQ、dK、dV。"""

    scale = 1.0 / math.sqrt(q.shape[-1])
    q_fp32, k_fp32, v_fp32 = q.float(), k.float(), v.float()
    grad_output_fp32 = grad_output.float()

    scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * scale
    if is_causal:
        query_index = torch.arange(q.shape[-2], device=q.device)[:, None]
        key_index = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = torch.where(query_index >= key_index, scores, -1e6)

    # L 已经是 log(sum(exp(scores)))，因此 exp(scores - L) 直接得到概率。
    probabilities = torch.exp(scores - logsumexp.float().unsqueeze(-1))
    grad_v = torch.matmul(probabilities.transpose(-2, -1), grad_output_fp32)
    grad_probabilities = torch.matmul(grad_output_fp32, v_fp32.transpose(-2, -1))

    # D_i = sum_j P_ij * dP_ij = sum_d O_id * dO_id。
    # 使用 O 与 dO 只需保存一个行向量，避免形成 softmax Jacobian。
    d_vector = (output.float() * grad_output_fp32).sum(dim=-1, keepdim=True)
    grad_scores = probabilities * (grad_probabilities - d_vector)
    grad_q = torch.matmul(grad_scores, k_fp32) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q_fp32) * scale
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


_COMPILED_BACKWARD: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None
_COMPILE_WARNING_EMITTED = False


def flash_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA 上使用 torch.compile，CPU 本地测试使用同一公式的 eager 版本。"""

    global _COMPILED_BACKWARD, _COMPILE_WARNING_EMITTED
    if q.is_cuda and os.environ.get("CS336_FLASH_BACKWARD_EAGER") != "1":
        if _COMPILED_BACKWARD is None:
            _COMPILED_BACKWARD = torch.compile(_flash_backward_math, fullgraph=True)
        try:
            return _COMPILED_BACKWARD(q, k, v, output, grad_output, logsumexp, is_causal)
        except Exception as error:  # 保留一次可诊断警告，仍允许在不支持 compile 的环境验证正确性。
            if os.environ.get("CS336_FLASH_BACKWARD_REQUIRE_COMPILE") == "1":
                raise
            if not _COMPILE_WARNING_EMITTED:
                warnings.warn(f"compiled FlashAttention backward failed; falling back to eager: {error}", stacklevel=2)
                _COMPILE_WARNING_EMITTED = True
    return _flash_backward_math(q, k, v, output, grad_output, logsumexp, is_causal)


class PyTorchFlashAttention(torch.autograd.Function):
    """使用 PyTorch 分块 forward 和重计算 backward 的 autograd.Function。"""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, logsumexp = tiled_flash_forward(q, k, v, is_causal)
        # 官方测试会检查恰好存在一个 [batch, query_length] 的保存张量，即 L。
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = flash_backward(q, k, v, output, grad_output, logsumexp, ctx.is_causal)
        return grad_q, grad_k, grad_v, None


if TRITON_AVAILABLE:

    @triton.jit
    def flash_fwd_kernel(
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
        n_queries,
        n_keys,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """一个 program 负责一个 batch 中的一块 query 行。"""

        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        dimension_offsets = tl.arange(0, D)
        query_mask = query_offsets < n_queries

        q_offsets = (
            batch_index * stride_qb
            + query_offsets[:, None] * stride_qq
            + dimension_offsets[None, :] * stride_qd
        )
        q = tl.load(q_ptr + q_offsets, mask=query_mask[:, None], other=0.0)

        running_max = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
        running_sum = tl.zeros((Q_TILE_SIZE,), tl.float32)
        output_accumulator = tl.zeros((Q_TILE_SIZE, D), tl.float32)

        for key_start in range(0, n_keys, K_TILE_SIZE):
            key_offsets = key_start + tl.arange(0, K_TILE_SIZE)
            key_mask = key_offsets < n_keys
            k_offsets = (
                batch_index * stride_kb
                + key_offsets[:, None] * stride_kk
                + dimension_offsets[None, :] * stride_kd
            )
            v_offsets = (
                batch_index * stride_vb
                + key_offsets[:, None] * stride_vk
                + dimension_offsets[None, :] * stride_vd
            )
            k = tl.load(k_ptr + k_offsets, mask=key_mask[:, None], other=0.0)
            value = tl.load(v_ptr + v_offsets, mask=key_mask[:, None], other=0.0)

            scores = tl.dot(q, tl.trans(k)) * scale
            scores = tl.where(query_mask[:, None] & key_mask[None, :], scores, -float("inf"))
            if is_causal:
                causal_mask = query_offsets[:, None] >= key_offsets[None, :]
                scores = tl.where(causal_mask, scores, -1.0e6)

            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, tile_max)
            old_rescale = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(key_mask[None, :], probabilities, 0.0)

            running_sum = running_sum * old_rescale + tl.sum(probabilities, axis=1)
            output_accumulator *= old_rescale[:, None]
            output_accumulator = tl.dot(probabilities.to(value.dtype), value, acc=output_accumulator)
            running_max = new_max

        output = output_accumulator / running_sum[:, None]
        output_offsets = (
            batch_index * stride_ob
            + query_offsets[:, None] * stride_oq
            + dimension_offsets[None, :] * stride_od
        )
        lse_offsets = batch_index * stride_lb + query_offsets * stride_lq
        tl.store(output_ptr + output_offsets, output, mask=query_mask[:, None])
        tl.store(logsumexp_ptr + lse_offsets, running_max + tl.log(running_sum), mask=query_mask)


def _select_triton_config(head_dimension: int) -> tuple[int, int, int]:
    """按 head dimension 选择保守 tile；所有官方维度都是 2 的幂。"""

    if head_dimension <= 64:
        return 64, 64, 4
    return 32, 64, 4


class TritonFlashAttention(torch.autograd.Function):
    """Triton 融合 forward + PyTorch/compile 重计算 backward。"""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not TRITON_AVAILABLE:
            raise RuntimeError("Triton is not installed in this environment")
        if not q.is_cuda or not k.is_cuda or not v.is_cuda:
            raise ValueError("Triton FlashAttention requires CUDA tensors")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("Triton FlashAttention expects [batch, sequence, d] tensors")
        if q.shape[0] != k.shape[0] or k.shape[0] != v.shape[0]:
            raise ValueError("Q, K, V must have the same batch size")
        if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
            raise ValueError("this A2 kernel requires equal Q/K/V head dimensions")
        if q.shape[-1] not in (16, 32, 64, 128):
            raise ValueError("head dimension must be one of 16, 32, 64, 128")

        batch_size, n_queries, head_dimension = q.shape
        n_keys = k.shape[-2]
        output = torch.empty_like(q)
        logsumexp = torch.empty((batch_size, n_queries), device=q.device, dtype=torch.float32)
        query_tile_size, key_tile_size, num_warps = _select_triton_config(head_dimension)
        grid = (triton.cdiv(n_queries, query_tile_size), batch_size)

        flash_fwd_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            *logsumexp.stride(),
            n_queries,
            n_keys,
            1.0 / math.sqrt(head_dimension),
            D=head_dimension,
            Q_TILE_SIZE=query_tile_size,
            K_TILE_SIZE=key_tile_size,
            is_causal=is_causal,
            num_warps=num_warps,
            num_stages=2,
        )
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = is_causal
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = flash_backward(q, k, v, output, grad_output, logsumexp, ctx.is_causal)
        return grad_q, grad_k, grad_v, None
