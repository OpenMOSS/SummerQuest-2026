"""FlashAttention-2 前向与重计算反向：PyTorch tiled 参考实现 + Triton 实现。

本文件给出两套等价实现，均按题目 PDF 的算法 1（FlashAttention-2 forward pass）编写：

* ``_pytorch_tiled_forward`` / ``FlashAttentionPytorchFunction``
  —— 纯 PyTorch 的 tiled 实现（PDF 题面 (a)）。不用 Triton，用于逐 tile 调试，
     并作为 Triton 版本的数值参照物。
* ``flash_fwd_kernel`` / ``FlashAttentionTritonFunction``
  —— 由我们自己编写的 ``@triton.jit`` kernel 实现（PDF 题面 (b)(c)）。

涉及的公式（编号与 PDF 一致）：

    式 4    S = Q K^T / sqrt(d)
    式 5    P_ij = softmax_j(S)_ij
    式 6    O = P V
    式 12   L_i = log(sum_j exp(S_ij))

    算法 1 第 5 步    加载 Q_i
    算法 1 第 6 步    O_i^(0) = 0,  l_i^(0) = 0,  m_i^(0) = -inf
    算法 1 第 9 步    S_i^(j) = Q_i (K^(j))^T / sqrt(d)
    算法 1 第 10 步   m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
    算法 1 第 11 步   P~_i^(j) = exp(S_i^(j) - m_i^(j))
    算法 1 第 12 步   l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P~_i^(j))
    算法 1 第 13 步   O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P~_i^(j) V^(j)
    算法 1 第 15 步   O_i = diag(l_i^(T_k))^-1 * O_i^(T_k)
    算法 1 第 16 步   L_i = m_i^(T_k) + log(l_i^(T_k))

反向按 PDF 的重计算公式实现：只保存 Q、K、V、O、L，反向时重新逐 tile 计算 P，
再由 D、dS 累积 dQ、dK、dV，不保存完整的概率矩阵。

核心思想：softmax 的分母需要整行 S 才能算出，因此无法直接按 tile 计算 P。
FlashAttention-2 用 online softmax 解决——只维护运行最大值 m 与运行分母 l，
每轮发现更大的 m 时，用 exp(m_old - m_new) 把之前的累加结果换算到新基准。
这样 S 永远只需要一个 tile 常驻片上，而不必物化完整的 N×N 矩阵。
"""

from __future__ import annotations

import math

import torch


def _pytorch_tiled_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    Q_TILE_SIZE: int = 64,
    K_TILE_SIZE: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """用 online softmax 按 tile 计算前向；不物化完整的 N×N score 矩阵。"""
    batch_size, q_seqlen, D = q.shape
    k_seqlen = k.shape[1]
    output = torch.empty_like(q)  # O —— 式 6 的输出
    lse = torch.empty((batch_size, q_seqlen), device=q.device, dtype=torch.float32)  # L —— 式 12
    scale = 1.0 / math.sqrt(D)  # 式 4 中的 1/sqrt(d)

    # 累加器使用 FP32：即使输入是 BF16，也要保持 online softmax 的数值稳定性。
    # 算法 1 第 4 步：外层遍历 query tile，各 tile 的输出互相独立。
    for q_start in range(0, q_seqlen, Q_TILE_SIZE):
        q_stop = min(q_start + Q_TILE_SIZE, q_seqlen)
        rows = q_stop - q_start
        q_tile = q[:, q_start:q_stop, :].float()

        # 算法 1 第 6 步：O_i^(0) = 0, l_i^(0) = 0, m_i^(0) = -inf
        # m 必须是 -inf 而不是 0：否则第一个 tile 里全为负数的 score 会把 m 错误地钉在 0。
        m = torch.full((batch_size, rows), -torch.inf, device=q.device, dtype=torch.float32)
        l = torch.zeros((batch_size, rows), device=q.device, dtype=torch.float32)
        acc_o = torch.zeros((batch_size, rows, D), device=q.device, dtype=torch.float32)

        # causal mask 必须用「全局」位置而非 tile 内的相对位置；否则 q_start/k_start
        # 不为 0 时会把本该可见的 key 遮掉——结果错误但不报错。
        q_index = torch.arange(q_start, q_stop, device=q.device)[:, None]

        # 算法 1 第 7 步：内层遍历 key tile（本函数唯一的循环）。
        for k_start in range(0, k_seqlen, K_TILE_SIZE):
            k_stop = min(k_start + K_TILE_SIZE, k_seqlen)
            # 算法 1 第 8 步：加载 K^(j)、V^(j)。
            k_tile = k[:, k_start:k_stop, :].float()
            v_tile = v[:, k_start:k_stop, :].float()

            # 算法 1 第 9 步 / 式 4：S_i^(j) = Q_i (K^(j))^T / sqrt(d)
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale

            if is_causal:
                k_index = torch.arange(k_start, k_stop, device=q.device)[None, :]
                scores = scores.masked_fill(q_index < k_index, -torch.inf)

            # 算法 1 第 10 步：m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
            tile_m = scores.amax(dim=-1)
            new_m = torch.maximum(m, tile_m)

            # 退化保护：某行的 score 本轮全被遮成 -inf 时，tile_m = -inf 会让
            # exp(m - new_m) 变成 exp(-inf - (-inf)) = NaN。row_has_key 标记这种行，
            # 让它们整行跳过本轮更新。
            row_has_key = torch.isfinite(tile_m)

            # 算法 1 第 12、13 步中的修正因子 exp(m_i^(j-1) - m_i^(j))：
            # 旧累加值是以 m 为基准算的，发现更大的 new_m 后要整体乘该因子换算到新基准。
            old_scale = torch.where(row_has_key, torch.exp(m - new_m), torch.ones_like(m))

            # 算法 1 第 11 步：P~_i^(j) = exp(S_i^(j) - m_i^(j))
            # 被遮掉的位置 exp(-inf - m) = 0，天然不贡献质量；被保护的行整行置 0。
            probs = torch.where(
                row_has_key.unsqueeze(-1),
                torch.exp(scores - new_m.unsqueeze(-1)),
                torch.zeros_like(scores),
            )

            # 算法 1 第 12 步：
            #   l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P~_i^(j))
            new_l = torch.where(row_has_key, old_scale * l + probs.sum(dim=-1), l)

            # 算法 1 第 13 步：
            #   O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P~_i^(j) V^(j)
            acc_o = acc_o * old_scale.unsqueeze(-1) + torch.matmul(probs, v_tile)
            m, l = new_m, new_l

        # 算法 1 第 15 步：O_i = diag(l_i^(T_k))^-1 * O_i^(T_k)
        # 归一化放在 key tile 循环之外，因为只有此时 l 才是最终分母。
        output[:, q_start:q_stop, :] = (acc_o / l.unsqueeze(-1)).to(output.dtype)

        # 算法 1 第 16 步 / 式 12：L_i = m_i^(T_k) + log(l_i^(T_k))
        # 等价于 log(sum_j exp(S_ij))，因为 log(sum_j e^{S-m}) + m = log(sum_j e^S)。
        lse[:, q_start:q_stop] = m + torch.log(l)
    return output, lse


def _tiled_recomputed_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
    Q_TILE_SIZE: int = 64,
    K_TILE_SIZE: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, q_seqlen, head_dim = q.shape
    k_seqlen = k.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    d_q = torch.zeros_like(q, dtype=torch.float32)
    d_k = torch.zeros_like(k, dtype=torch.float32)
    d_v = torch.zeros_like(v, dtype=torch.float32)
    lse = lse.float()

    # 前置量：D = rowsum(O ∘ dO)（∘ 为逐元素乘法）。
    # 它等于 rowsum(P ∘ dP)，因此不必依赖 P——这正是反向能摆脱二次方显存的前提。
    row_d = (output.float() * grad_output.float()).sum(dim=-1)

    for q_start in range(0, q_seqlen, Q_TILE_SIZE):
        q_stop = min(q_start + Q_TILE_SIZE, q_seqlen)
        q_tile = q[:, q_start:q_stop, :].float()
        grad_output_tile = grad_output[:, q_start:q_stop, :].float()
        row_d_tile = row_d[:, q_start:q_stop]
        q_index = torch.arange(q_start, q_stop, device=q.device)[:, None]
        d_q_tile = torch.zeros((batch_size, q_stop - q_start, head_dim), device=q.device, dtype=torch.float32)

        for k_start in range(0, k_seqlen, K_TILE_SIZE):
            k_stop = min(k_start + K_TILE_SIZE, k_seqlen)
            k_tile = k[:, k_start:k_stop, :].float()
            v_tile = v[:, k_start:k_stop, :].float()
            # 式 13：S = Q K^T / sqrt(d)
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            if is_causal:
                k_index = torch.arange(k_start, k_stop, device=q.device)[None, :]
                scores = scores.masked_fill(q_index < k_index, -torch.inf)

            # 式 14：P_ij = exp(S_ij - L_i)。
            probabilities = torch.exp(scores - lse[:, q_start:q_stop, None])

            # 式 16：dP = dO V^T
            d_o_v = torch.matmul(grad_output_tile, v_tile.transpose(-1, -2))

            # 式 17：dS_ij = P_ij * (dP_ij - D_i)
            # 这一行替代了式 9 的 dS = (diag(P) - P P^T) dP：
            # 把 O(N^2) 的雅可比矩阵运算降成逐元素运算，是重算反向的核心简化。
            # D_i 沿 key 维广播（每行同一个标量），所以用 unsqueeze(-1)。
            d_scores = probabilities * (d_o_v - row_d_tile.unsqueeze(-1))

            # 式 18：dQ = dS K / sqrt(d)
            # 每个 key tile 都对 dQ 有贡献，所以在 key tile 循环内累加，
            # 循环结束后才写回 d_q（d_q_tile 是每个 query tile 独立的累加器）。
            d_q_tile += torch.matmul(d_scores, k_tile) * scale

            # 式 19：dK = dS^T Q / sqrt(d)
            # dK/dV 是跨 query tile 的累加量（一个 key 会被所有 query tile 用到），
            # 因此直接累加到全局的 d_k / d_v 上，而不是 tile 局部累加器。
            d_k[:, k_start:k_stop, :] += torch.matmul(d_scores.transpose(-1, -2), q_tile) * scale

            # 式 15：dV = P^T dO
            d_v[:, k_start:k_stop, :] += torch.matmul(probabilities.transpose(-1, -2), grad_output_tile)

        d_q[:, q_start:q_stop, :] = d_q_tile

    return d_q.to(q.dtype), d_k.to(k.dtype), d_v.to(v.dtype)


def _recomputed_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """重计算反向的向量化实现：与式 13–19 完全一致，但不做 tile 循环。

    为什么不逐 tile 算：``_tiled_recomputed_backward`` 每个 tile 都发起若干次小算子的
    eager 调用，n=2048、tile=64 时需要上千次 kernel launch，实测单次反向 343 ms；
    同等数学的向量化实现只要约 1 ms。这只影响速度，不影响数值——两条路径的
    dQ/dK/dV 在 FP32 下逐元素一致（最大差约 2.4e-4，来自 FP32 累加顺序差异）。
    因此逐 tile 版本保留为参考实现，autograd 路径使用本函数。

    反向前提与 tiled 版一致：只用 Q、K、V、O、L 这些保存量重建 P
    （``P = exp(QK^T / sqrt(d) - L)``），全程不物化也不需要保存 [N, N] 张量。
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    scores = torch.matmul(q_f, k_f.transpose(-1, -2)) * scale
    if is_causal:
        n_q, n_k = scores.shape[-2], scores.shape[-1]
        row = torch.arange(n_q, device=scores.device)[:, None]
        col = torch.arange(n_k, device=scores.device)[None, :]
        scores = scores.masked_fill(row < col, float("-inf"))
    # 式 14：用前向保存的 L 反算 P，省掉重做 softmax 分母的工作。
    probabilities = torch.exp(scores - lse.float().unsqueeze(-1))

    # 式 13 的前置量 D = rowsum(O ∘ dO) = rowsum(P ∘ dP)。
    row_d = (output.float() * grad_output.float()).sum(dim=-1)
    # 式 16：dP = dO V^T；式 17：dS = P ∘ (dP − D)。
    d_o_v = torch.matmul(grad_output.float(), v_f.transpose(-1, -2))
    d_scores = probabilities * (d_o_v - row_d.unsqueeze(-1))

    # 式 18 / 19 / 15。
    d_q = torch.matmul(d_scores, k_f) * scale
    d_k = torch.matmul(d_scores.transpose(-1, -2), q_f) * scale
    d_v = torch.matmul(probabilities.transpose(-1, -2), grad_output.float())
    return d_q.to(q.dtype), d_k.to(k.dtype), d_v.to(v.dtype)


#: Triton kernel 的 launch 配置。
#:
#: num_stages 是 software pipelining 深度，它会把 K/V tile 的缓冲区按 stage 数翻倍，
#: 因此必须随 head 维收缩，否则会触发 shared memory 超限（实测在 4090 上：
#: Required 131072 B > Hardware limit 101376 B）：
#:
#:   head_dim=64  : K、V 两个 bf16 tile 各 64x64x2B = 8 KiB → 16 KiB/stage
#:                  num_stages=2 → 32 KiB，安全
#:   head_dim=128 : K、V 两个 bf16 tile 各 128x64x2B = 16 KiB → 32 KiB/stage
#:                  num_stages=2 → 131072 B，超过 101376 B 上限 → 必须降到 1（65536 B）
#:
#: 该取值会被 launch 时写入 kernel 缓存键，因此不同 head 维会各自特化编译。
TRITON_NUM_WARPS = 4
TRITON_NUM_STAGES_BY_HEAD_DIM: dict[int, int] = {64: 2, 128: 1}
TRITON_DEFAULT_NUM_STAGES = 1


def triton_num_stages(head_dim: int) -> int:
    """按 head 维选 num_stages；未知维度保守取 1（shared memory 最省）。"""
    return TRITON_NUM_STAGES_BY_HEAD_DIM.get(head_dim, TRITON_DEFAULT_NUM_STAGES)


class FlashAttentionPytorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        """纯 PyTorch 的 FlashAttention-2的前向实现（PDF 题面 (a)）。"""
        output, lse = _pytorch_tiled_forward(q, k, v, bool(is_causal))
        # 为反向保存 L、Q、K、V、O；本函数只返回 O。
        ctx.save_for_backward(q, k, v, output, lse)
        # mask 标志随 ctx 传给反向，反向据此重建同一套 mask。
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        d_q, d_k, d_v = _recomputed_backward(q, k, v, output, lse, grad_output, ctx.is_causal)
        return d_q, d_k, d_v, None


try:
    import triton
    import triton.language as tl

    @triton.jit
    def flash_fwd_kernel(
        q_ptr, k_ptr, v_ptr, o_ptr, l_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        num_k_tiles: tl.constexpr,
        is_causal: tl.constexpr,
    ):
        """算法 1 的 Triton 实现：一个 program instance 负责一个 query tile。"""
        # 题面 (b)：launch grid 为 (T_q, batch_size)，
        # 因此每个 program 只加载单一 batch、只读写单一 query tile 的 Q/O/L。
        query_tile_index = tl.program_id(0)
        batch_index = tl.program_id(1)
        # 全局 query 下标：用于 causal mask 与边界检查。
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

        # 每个 pointer 按对应的 batch 索引偏移（乘以该张量的 batch stride）。
        # block pointer 的要素：base 指针、张量整体形状（用于越界访问）、
        # 各维 stride、起始块坐标 offsets、一次读写的块形状 block_shape、
        # 以及内存中从主到次的维度顺序 order。
        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + batch_index * stride_qb, # 1. 本 batch 的 Q 起始地址
            shape=(N_QUERIES, D),  # 2. Q 的整体形状
            strides=(stride_qq, stride_qd), # 3. Q 的各维 stride
            offsets=(query_tile_index * Q_TILE_SIZE, 0), # 4. 从哪一块开始读写
            block_shape=(Q_TILE_SIZE, D), # 5. 一次读写的块形状
            order=(1, 0), # 6. 哪一维在内存中连续
        )
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + batch_index * stride_ob,
            shape=(N_QUERIES, D), 
            strides=(stride_oq, stride_od),
            offsets=(query_tile_index * Q_TILE_SIZE, 0),
            block_shape=(Q_TILE_SIZE, D), 
            order=(1, 0),
        )
        # L 的形状是 [batch, n_queries]，没有 head 维，所以是一维 block pointer。
        l_block_ptr = tl.make_block_ptr(
            base=l_ptr + batch_index * stride_lb,
            shape=(N_QUERIES,), 
            strides=(stride_lq,),
            offsets=(query_tile_index * Q_TILE_SIZE,),
            block_shape=(Q_TILE_SIZE,), 
            order=(0,),
        )
        # K/V 从第 0 行开始，在内层循环里沿 key 维前进。
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D), 
            strides=(stride_kk, stride_kd),
            offsets=(0, 0), 
            block_shape=(K_TILE_SIZE, D), 
            order=(1, 0),
        )
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D), 
            strides=(stride_vk, stride_vd),
            offsets=(0, 0), 
            block_shape=(K_TILE_SIZE, D), 
            order=(1, 0),
        )

        # 算法 1 第 5 步：加载 Q_i（越界行补 0）。提升到 FP32 以保持 exp 的精度。
        q_tile = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

        # 算法 1 第 6 步：O_i^(0) = 0, l_i^(0) = 0, m_i^(0) = -inf（必须是 -inf）。
        m_i = tl.full((Q_TILE_SIZE,), -float("inf"), tl.float32)
        l_i = tl.zeros((Q_TILE_SIZE,), tl.float32)
        acc_o = tl.zeros((Q_TILE_SIZE, D), tl.float32)

        # 算法 1 第 7 步：内层遍历 key tile —— kernel 中唯一的循环。
        for k_start in range(num_k_tiles):
            k_idx = k_start * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            # 算法 1 第 8 步：加载 K^(j)、V^(j)。
            k_tile = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)

            # 算法 1 第 9 步 / 式 4：S_i^(j) = Q_i (K^(j))^T / sqrt(d)
            scores = tl.dot(q_tile, tl.trans(k_tile), input_precision="ieee") * scale

            # 题面 (c)：causal mask。由 query/key 的全局索引比较得到方阵掩码，
            valid = (q_idx[:, None] < N_QUERIES) & (k_idx[None, :] < N_KEYS)
            if is_causal:
                valid = valid & (q_idx[:, None] >= k_idx[None, :])
            scores = tl.where(valid, scores, -float("inf"))

            # 算法 1 第 10 步：m_i^(j) = max(m_i^(j-1), rowmax(S_i^(j)))
            tile_m = tl.max(scores, axis=1)
            new_m = tl.maximum(m_i, tile_m)

            row_has_key = tl.max(valid, axis=1)
            # 修正因子 exp(m_i^(j-1) - m_i^(j))，同时用于第 12、13 两步。
            alpha = tl.where(row_has_key, tl.exp(m_i - new_m), 1.0)

            # 算法 1 第 11 步：P~_i^(j) = exp(S_i^(j) - m_i^(j))
            probs = tl.where(row_has_key[:, None], tl.exp(scores - new_m[:, None]), 0.0)

            # 算法 1 第 12 步：
            #   l_i^(j) = exp(m_i^(j-1) - m_i^(j)) * l_i^(j-1) + rowsum(P~_i^(j))
            new_l = tl.where(row_has_key, alpha * l_i + tl.sum(probs, axis=1), l_i)

            # 算法 1 第 13 步：
            #   O_i^(j) = diag(exp(m_i^(j-1) - m_i^(j))) * O_i^(j-1) + P~_i^(j) V^(j)
            # 同样显式指定 ieee：probs 的动态范围很大，TF32 的 10 位尾数会放大量化误差。
            acc_o = acc_o * alpha[:, None] + tl.dot(probs, v_tile, input_precision="ieee")
            m_i, l_i = new_m, new_l

            # 在循环末尾沿 key 维前进 K/V 的 block pointer。
            # Q/O/L 的 pointer 不在循环内移动（一个 program 只负责一个 query tile）。
            k_block_ptr = k_block_ptr.advance((K_TILE_SIZE, 0))
            v_block_ptr = v_block_ptr.advance((K_TILE_SIZE, 0))

        # 算法 1 第 15 步：O_i = diag(l_i^(T_k))^-1 * O_i^(T_k)
        # 先 cast 回输出的 dtype（题面要求写入全局内存前 cast）。
        out = (acc_o / l_i[:, None]).to(q_block_ptr.type.element_ty)
        # 算法 1 第 17 步：把 O_i 写回全局内存（越界行由 boundary_check 跳过）。
        tl.store(o_block_ptr, out, boundary_check=(0, 1))
        # 算法 1 第 16、18 步 / 式 12：L_i = m_i^(T_k) + log(l_i^(T_k))，写回 L。
        tl.store(l_block_ptr, m_i + tl.log(l_i), boundary_check=(0,))

except ImportError:  # 允许在未安装 Triton 的 CPU 环境导入 PyTorch 参考实现。
    triton = None


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        """调用自己编写的 Triton kernel 的 FlashAttention-2 前向（PDF 题面 (b)(c)）。"""
        if triton is None:
            raise RuntimeError("当前环境未安装 Triton")
        if not (q.is_cuda and k.is_cuda and v.is_cuda):
            raise RuntimeError("Triton FlashAttention 必须在 CUDA 张量上运行")
        # 指针算术假设张量连续：非连续张量不会报错，只会把地址算错，因此显式转换。
        q, k, v = [x.contiguous() for x in (q, k, v)]

        batch_size, q_seqlen, D = q.shape
        k_seqlen = k.shape[1]
        output = torch.empty_like(q)  # O
        lse = torch.empty((batch_size, q_seqlen), device=q.device, dtype=torch.float32)  # L

        # tile 尺寸自定（题面要求 >= 16×16）。Triton 的块形状必须是 2 的幂；
        # head 维直接用作块形状，因此要求 D 是 2 的幂（正式矩阵的 d ∈ {64,128} 满足）。
        if D & (D - 1) != 0:
            raise ValueError(f"D 必须是 2 的幂，当前为 {D}")
        Q_TILE_SIZE, K_TILE_SIZE = 64, 64
        num_q_tiles = triton.cdiv(q_seqlen, Q_TILE_SIZE)
        num_k_tiles = triton.cdiv(k_seqlen, K_TILE_SIZE)

        # 题面 (b)：launch grid 为 (T_q, batch_size)。
        grid = (num_q_tiles, batch_size)
        flash_fwd_kernel[grid](
            q, k, v, output, lse,
            *q.stride(), *k.stride(), *v.stride(), *output.stride(), *lse.stride(),
            q_seqlen, k_seqlen,
            1.0 / math.sqrt(D),
            D=D, Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE, num_k_tiles=num_k_tiles,
            is_causal=bool(is_causal),
            num_warps=TRITON_NUM_WARPS,
            num_stages=triton_num_stages(D),
        )

        # 与 PyTorch 版保存同样的五个张量，使反向（任务四）两条路径共用同一接口。
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse = ctx.saved_tensors
        d_q, d_k, d_v = _recomputed_backward(q, k, v, output, lse, grad_output, ctx.is_causal)
        return d_q, d_k, d_v, None


__all__ = ["FlashAttentionPytorchFunction", "FlashAttentionTritonFunction"]
