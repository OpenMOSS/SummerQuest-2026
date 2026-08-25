import math

import torch


def _flashattention_backward_compiled_impl(
    q,
    k,
    v,
    out,
    dout,
    lse,
    scale,
    is_causal,
):
    """Required PyTorch/torch.compile backward with probability recomputation."""
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if is_causal:
        query_indices = torch.arange(q.shape[-2], device=q.device)
        key_indices = torch.arange(k.shape[-2], device=q.device)
        causal_mask = query_indices[:, None] >= key_indices[None, :]
        scores = torch.where(causal_mask, scores, -1e6)

    probabilities = torch.exp(scores - lse.float().unsqueeze(-1))
    d = torch.sum(out.float() * dout.float(), dim=-1)
    dp = torch.matmul(dout.float(), v.float().transpose(-2, -1))
    ds = probabilities * (dp - d.unsqueeze(-1))

    dq = torch.matmul(ds, k.float()) * scale
    dk = torch.matmul(ds.transpose(-2, -1), q.float()) * scale
    dv = torch.matmul(probabilities.transpose(-2, -1), dout.float())
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


_flashattention_backward_compiled = torch.compile(
    _flashattention_backward_compiled_impl
)


class FlashAttentionPytorch(torch.autograd.Function):
    """
    PyTorch implementation of FlashAttention.

    The implementation should:
        - avoid materializing the full attention matrix
        - compute attention block-by-block
        - save only tensors required for backward
        - save exactly one (B, Q) tensor corresponding to log-sum-exp
    """

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        ctx.input_count = len(ctx.needs_input_grad)
        B, Q, D = q.shape
        _, K, _ = k.shape

        scale = 1.0 / math.sqrt(D)

        BLOCK_Q = 64
        BLOCK_K = 64

        out = torch.empty_like(q)

        # Exactly one (B, Q) tensor must be saved
        lse = torch.empty((B, Q), dtype=torch.float32, device=q.device)

        for bq_start in range(0, Q, BLOCK_Q):
            bq_end = min(bq_start + BLOCK_Q, Q)
            q_block = q[:, bq_start:bq_end, :]
            block_q = bq_end - bq_start

            running_max = torch.full(
                (B, block_q),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            running_sum = torch.zeros(
                (B, block_q),
                dtype=torch.float32,
                device=q.device,
            )
            running_output = torch.zeros(
                (B, block_q, D),
                dtype=torch.float32,
                device=q.device,
            )

            for bk_start in range(0, K, BLOCK_K):
                bk_end = min(bk_start + BLOCK_K, K)
                k_block = k[:, bk_start:bk_end, :]
                v_block = v[:, bk_start:bk_end, :]

                # Compute local scores in float32 for stability
                scores = (
                    torch.matmul(q_block.float(), k_block.float().transpose(-2, -1))
                    * scale
                )

                if is_causal:
                    col_indices = torch.arange(bk_start, bk_end, device=q.device)
                    row_indices = torch.arange(bq_start, bq_end, device=q.device)
                    mask = row_indices.unsqueeze(1) < col_indices.unsqueeze(0)
                    # Match reference implementation
                    scores = scores.masked_fill(mask.unsqueeze(0), -1e6)

                block_max = torch.max(scores, dim=-1).values
                new_max = torch.maximum(running_max, block_max)

                exp_running = torch.exp(running_max - new_max)
                exp_block = torch.exp(scores - new_max.unsqueeze(-1))

                new_sum = running_sum * exp_running + torch.sum(exp_block, dim=-1)

                running_output = (
                    running_output * exp_running.unsqueeze(-1)
                    + torch.matmul(exp_block, v_block.float())
                )

                running_max = new_max
                running_sum = new_sum

            out[:, bq_start:bq_end, :] = (
                running_output / running_sum.unsqueeze(-1)
            ).to(q.dtype)

            lse[:, bq_start:bq_end] = torch.log(running_sum) + running_max

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.block_q = BLOCK_Q
        ctx.block_k = BLOCK_K

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out, lse = ctx.saved_tensors

        scale = ctx.scale
        is_causal = ctx.is_causal
        BLOCK_Q = ctx.block_q
        BLOCK_K = ctx.block_k

        B, Q, D = q.shape
        _, K, _ = k.shape

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        for bq_start in range(0, Q, BLOCK_Q):
            bq_end = min(bq_start + BLOCK_Q, Q)
            q_block = q[:, bq_start:bq_end, :]
            dout_block = dout[:, bq_start:bq_end, :]
            lse_block = lse[:, bq_start:bq_end]
            block_q = bq_end - bq_start

            row_sum = torch.zeros(
                (B, block_q),
                dtype=torch.float32,
                device=q.device,
            )

            # Pass 1: dv and row_sum
            for bk_start in range(0, K, BLOCK_K):
                bk_end = min(bk_start + BLOCK_K, K)
                k_block = k[:, bk_start:bk_end, :]
                v_block = v[:, bk_start:bk_end, :]

                scores = (
                    torch.matmul(q_block.float(), k_block.float().transpose(-2, -1))
                    * scale
                )

                if is_causal:
                    col_indices = torch.arange(bk_start, bk_end, device=q.device)
                    row_indices = torch.arange(bq_start, bq_end, device=q.device)
                    mask = row_indices.unsqueeze(1) < col_indices.unsqueeze(0)
                    scores = scores.masked_fill(mask.unsqueeze(0), -1e6)

                probs = torch.exp(scores - lse_block.unsqueeze(-1))

                dv[:, bk_start:bk_end, :] += torch.matmul(
                    probs.transpose(-2, -1),
                    dout_block.float(),
                ).to(v.dtype)

                dP = torch.matmul(
                    dout_block.float(),
                    v_block.float().transpose(-2, -1),
                )

                row_sum += torch.sum(dP * probs, dim=-1)

            # Pass 2: dq and dk
            for bk_start in range(0, K, BLOCK_K):
                bk_end = min(bk_start + BLOCK_K, K)
                k_block = k[:, bk_start:bk_end, :]
                v_block = v[:, bk_start:bk_end, :]

                scores = (
                    torch.matmul(q_block.float(), k_block.float().transpose(-2, -1))
                    * scale
                )

                if is_causal:
                    col_indices = torch.arange(bk_start, bk_end, device=q.device)
                    row_indices = torch.arange(bq_start, bq_end, device=q.device)
                    mask = row_indices.unsqueeze(1) < col_indices.unsqueeze(0)
                    scores = scores.masked_fill(mask.unsqueeze(0), -1e6)

                probs = torch.exp(scores - lse_block.unsqueeze(-1))

                dP = torch.matmul(
                    dout_block.float(),
                    v_block.float().transpose(-2, -1),
                )

                dS = probs * (dP - row_sum.unsqueeze(-1))

                dq[:, bq_start:bq_end, :] += (
                    torch.matmul(dS, k_block.float()) * scale
                ).to(q.dtype)

                dk[:, bk_start:bk_end, :] += (
                    torch.matmul(dS.transpose(-2, -1), q_block.float()) * scale
                ).to(k.dtype)

        gradients = (dq, dk, dv)
        return (*gradients, None) if ctx.input_count == 4 else gradients


def get_flashattention_autograd_function_pytorch():
    return FlashAttentionPytorch


# =============================================================================
# Triton FlashAttention
# =============================================================================
# Triton is optional on machines without CUDA.
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
#
# -----------------------------------------------------------------------------
# Forward Kernel
# -----------------------------------------------------------------------------
#
if HAS_TRITON:
    @triton.jit
    def _flashattention_forward_kernel(
            #
            # Q / K / V
            #
            Q_ptr,
            K_ptr,
            V_ptr,
            #
            # Output
            #
            O_ptr,
            LSE_ptr,
            #
            # Shapes
            #
            B,
            Q_LEN,
            K_LEN,
            D,
            #
            # Strides
            #
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
            #
            # Constants
            #
            SCALE: tl.constexpr,
            IS_CAUSAL: tl.constexpr,
            BLOCK_Q: tl.constexpr,
            BLOCK_K: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        """
        One Triton program computes one (query block).
        """
        # 程序索引
        pid_b = tl.program_id(0)
        pid_q = tl.program_id(1)
        # Q block 的行/列索引
        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offsets = tl.arange(0, BLOCK_D)
        q_valid = q_offsets < Q_LEN
        # 初始化 Q 的指针并加载 Q block
        q_ptrs = Q_ptr + pid_b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
        Q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
        # 初始化 running max, running sum, running output
        m_i = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
        acc_o = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
        # 循环 K blocks
        for k_start in range(0, K_LEN, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_valid = k_offsets < K_LEN
            # 加载 K 和 V
            k_ptrs = K_ptr + pid_b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
            v_ptrs = V_ptr + pid_b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd
            K = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
            V = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)
            # 计算 scores: [BLOCK_Q, BLOCK_K]
            s = tl.dot(Q, tl.trans(K)) * SCALE
            # apply causal mask and valid mask
            if IS_CAUSAL:
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                valid_mask = causal_mask & k_valid[None, :]
            else:
                valid_mask = k_valid[None, :]
            s = tl.where(valid_mask, s, float('-inf'))
            # online softmax update
            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            # 防止整行被 mask 时产生 NaN
            alpha = tl.where(m_new == float('-inf'), 0.0, alpha)
            p = tl.where(s == float('-inf'), 0.0, p)
            l_ij = tl.sum(p, axis=1)
            l_new = alpha * l_i + l_ij
            # 更新 running output
            acc_o = acc_o * alpha[:, None]
            acc_o = tl.dot(p.to(V.dtype), V, acc_o)
            m_i = m_new
            l_i = l_new
        # write output
        acc_o = acc_o / l_i[:, None]
        # 防止除零导致 NaN
        acc_o = tl.where(l_i[:, None] == 0.0, 0.0, acc_o)
        o_ptrs = O_ptr + pid_b * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
        tl.store(o_ptrs, acc_o.to(Q_ptr.dtype.element_ty), mask=q_valid[:, None])
        # write LSE
        lse = m_i + tl.log(l_i)
        lse_ptrs = LSE_ptr + pid_b * stride_lb + q_offsets * stride_lq
        tl.store(lse_ptrs, lse, mask=q_valid)
#
# -----------------------------------------------------------------------------
# Backward Kernel
# -----------------------------------------------------------------------------
#
if HAS_TRITON:
    @triton.jit
    def _flashattention_backward_kernel(
            Q_ptr,
            K_ptr,
            V_ptr,
            O_ptr,
            DO_ptr,
            LSE_ptr,
            DQ_ptr,
            DK_ptr,
            DV_ptr,
            B,
            Q_LEN,
            K_LEN,
            D,
            stride_qb, stride_qq, stride_qd,
            stride_kb, stride_kk, stride_kd,
            stride_vb, stride_vk, stride_vd,
            stride_ob, stride_oq, stride_od,
            stride_dob, stride_doq, stride_dod,
            stride_lb, stride_lq,
            stride_dqb, stride_dqq, stride_dqd,
            stride_dkb, stride_dkk, stride_dkd,
            stride_dvb, stride_dvk, stride_dvd,
            SCALE: tl.constexpr,
            IS_CAUSAL: tl.constexpr,
            BLOCK_Q: tl.constexpr,
            BLOCK_K: tl.constexpr,
            BLOCK_D: tl.constexpr,
    ):
        """
        Recompute attention block.
        Compute dV, dK, dQ
        """
        pid_b = tl.program_id(0)
        pid_q = tl.program_id(1)
        q_offsets = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        d_offsets = tl.arange(0, BLOCK_D)
        q_valid = q_offsets < Q_LEN
        # 加载 Q, O, DO, LSE
        q_ptrs = Q_ptr + pid_b * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd
        o_ptrs = O_ptr + pid_b * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od
        do_ptrs = DO_ptr + pid_b * stride_dob + q_offsets[:, None] * stride_doq + d_offsets[None, :] * stride_dod
        lse_ptrs = LSE_ptr + pid_b * stride_lb + q_offsets * stride_lq
        Q = tl.load(q_ptrs, mask=q_valid[:, None], other=0.0)
        Out = tl.load(o_ptrs, mask=q_valid[:, None], other=0.0)
        DO = tl.load(do_ptrs, mask=q_valid[:, None], other=0.0)
        LSE = tl.load(lse_ptrs, mask=q_valid, other=0.0)
        # 计算 Di
        Di = tl.sum(DO.to(tl.float32) * Out.to(tl.float32), axis=1)
        # 初始化 dQ
        dQ = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)
        # 循环 K blocks
        for k_start in range(0, K_LEN, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_valid = k_offsets < K_LEN
            k_ptrs = K_ptr + pid_b * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd
            v_ptrs = V_ptr + pid_b * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd
            K = tl.load(k_ptrs, mask=k_valid[:, None], other=0.0)
            V = tl.load(v_ptrs, mask=k_valid[:, None], other=0.0)
            # 重新计算 scores
            s = tl.dot(Q, tl.trans(K)) * SCALE
            if IS_CAUSAL:
                causal_mask = q_offsets[:, None] >= k_offsets[None, :]
                valid_mask = causal_mask & k_valid[None, :]
            else:
                valid_mask = k_valid[None, :]
            s = tl.where(valid_mask, s, float('-inf'))
            # 重新计算 P
            P = tl.exp(s - LSE[:, None])
            P = tl.where(valid_mask, P, 0.0)
            # 计算 dV
            # dV = P^T @ DO
            dV = tl.dot(tl.trans(P.to(DO.dtype)), DO)
            # 计算 dP 和 dS
            # dP = DO @ V^T
            dP = tl.dot(DO, tl.trans(V))
            dS = P * (dP - Di[:, None])
            # 提前缩放 dS，避免在累积 dQ 时重复乘 SCALE
            dS_scaled = dS * SCALE
            # 计算 dQ: dQ += dS_scaled @ K
            dQ = tl.dot(dS_scaled.to(K.dtype), K, dQ)
            # 计算 dK: dK = dS_scaled^T @ Q
            dK = tl.dot(tl.trans(dS_scaled.to(Q.dtype)), Q)
            # 原子写出 dK, dV (因为有多个 Q block 会累加到同一个 K block)
            dk_ptrs = DK_ptr + pid_b * stride_dkb + k_offsets[:, None] * stride_dkk + d_offsets[None, :] * stride_dkd
            dv_ptrs = DV_ptr + pid_b * stride_dvb + k_offsets[:, None] * stride_dvk + d_offsets[None, :] * stride_dvd
            tl.atomic_add(dk_ptrs, dK, mask=k_valid[:, None])
            tl.atomic_add(dv_ptrs, dV, mask=k_valid[:, None])
        # 写出 dQ
        dq_ptrs = DQ_ptr + pid_b * stride_dqb + q_offsets[:, None] * stride_dqq + d_offsets[None, :] * stride_dqd
        tl.store(dq_ptrs, dQ, mask=q_valid[:, None])

# =============================================================================
# Triton Autograd Function
# =============================================================================
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        ctx.input_count = len(ctx.needs_input_grad)
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed.")
        B, Q, D = q.shape
        _, K, _ = k.shape
        if D == 128 and q.element_size() == 4:
            BLOCK_Q = 32
            BLOCK_K = 32
        else:
            BLOCK_Q = 64
            BLOCK_K = 64
        BLOCK_D = D
        scale = 1.0 / math.sqrt(D)
        out = torch.empty_like(q)
        lse = torch.empty(
            (B, Q),
            dtype=torch.float32,
            device=q.device,
        )
        grid = (B, triton.cdiv(Q, BLOCK_Q))
        launch_config = get_triton_forward_config(q.dtype, D)
        _flashattention_forward_kernel[grid](
            q, k, v, out, lse,
            B, Q, K, D,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            lse.stride(0), lse.stride(1),
            SCALE=scale,
            IS_CAUSAL=is_causal,
            BLOCK_Q=BLOCK_Q,
            BLOCK_K=BLOCK_K,
            BLOCK_D=BLOCK_D,
            num_warps=launch_config["num_warps"],
            num_stages=launch_config["num_stages"],
        )
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
        )
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.block_q = BLOCK_Q
        ctx.block_k = BLOCK_K
        return out

    @staticmethod
    def backward(ctx, dout):
        if not HAS_TRITON:
            raise RuntimeError("Triton is not installed.")
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flashattention_backward_compiled(
            q,
            k,
            v,
            out,
            dout,
            lse,
            ctx.scale,
            ctx.is_causal,
        )
        gradients = (dq, dk, dv)
        return (*gradients, None) if ctx.input_count == 4 else gradients


def _flashattention_triton_backward(ctx, dout):
    """Experimental Triton backward, not used by adapters or formal results."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed.")
    q, k, v, out, lse = ctx.saved_tensors
    dq = torch.empty_like(q)
    # dK and dV receive atomic updates from multiple query blocks.
    # Triton 3.3 cannot compile bf16 atomic_add on RTX 4090, so use fp32
    # accumulation buffers for reduced-precision inputs and cast afterward.
    accumulation_dtype = (
        torch.float32
        if q.dtype in (torch.float16, torch.bfloat16)
        else q.dtype
    )
    dk_accum = torch.zeros_like(k, dtype=accumulation_dtype)
    dv_accum = torch.zeros_like(v, dtype=accumulation_dtype)
    block_q = ctx.block_q
    block_k = ctx.block_k
    device_properties = torch.cuda.get_device_properties(q.device)
    if (
        q.dtype == torch.float32
        and q.shape[2] >= 128
        and device_properties.shared_memory_per_block_optin < 139_520
    ):
        block_q = 16
        block_k = 16
    elif (
        q.shape[2] >= 64
        and device_properties.shared_memory_per_block_optin < 148_480
    ):
        block_q = 32
        block_k = 32

    grid = (q.shape[0], triton.cdiv(q.shape[1], block_q))
    _flashattention_backward_kernel[grid](
        q, k, v, out, dout, lse,
        dq, dk_accum, dv_accum,
        q.shape[0], q.shape[1], k.shape[1], q.shape[2],
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        lse.stride(0), lse.stride(1),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk_accum.stride(0), dk_accum.stride(1), dk_accum.stride(2),
        dv_accum.stride(0), dv_accum.stride(1), dv_accum.stride(2),
        SCALE=ctx.scale,
        IS_CAUSAL=ctx.is_causal,
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_D=q.shape[2],
    )
    dk = dk_accum.to(k.dtype)
    dv = dv_accum.to(v.dtype)
    return dq, dk, dv, None


class FlashAttentionTritonOptimizedBackward(torch.autograd.Function):
    """Triton forward plus the optional tiled Triton backward."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        ctx.input_count = len(ctx.needs_input_grad)
        return FlashAttentionTriton.forward(ctx, q, k, v, is_causal)

    @staticmethod
    def backward(ctx, dout):
        return _flashattention_triton_backward(ctx, dout)


def get_flashattention_autograd_function_triton():
    return FlashAttentionTriton


def get_flashattention_autograd_function_triton_optimized_backward():
    return FlashAttentionTritonOptimizedBackward


def get_triton_forward_config(dtype: torch.dtype, head_dim: int) -> dict[str, int]:
    """Return the launch configuration used by the required Triton forward."""
    if head_dim == 128 and dtype == torch.float32:
        block_q = 32
        block_k = 32
    else:
        block_q = 64
        block_k = 64
    return {
        "block_q": block_q,
        "block_k": block_k,
        "num_warps": 4,
        "num_stages": 2,
    }
