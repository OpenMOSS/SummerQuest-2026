import torch 
import math
import triton
import triton.language as tl

@triton.jit
def flashattention_triton_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oa, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    is_causal:tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    ):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oa, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0)
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES, ),
        strides=(stride_lq, ),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    # 加载并转换为 fp32 进行计算
    q = tl.load(Q_block_ptr).to(tl.float32)

    m_i = tl.full([Q_TILE_SIZE, 1], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE, 1], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    for k_tile_idx in range(0, N_KEYS, K_TILE_SIZE):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D),
            strides=(stride_kk, stride_kd),
            offsets=(k_tile_idx, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1,0)
        )
        k = tl.load(K_block_ptr).to(tl.float32)
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D),
            strides=(stride_vk, stride_vd),
            offsets=(k_tile_idx, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1,0)
        )
        v = tl.load(V_block_ptr).to(tl.float32)

        S = tl.dot(q, tl.trans(k)) * scale #[Q_TILE_SIZE, K_TILE_SIZE]

        offs_q = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        offs_k = k_tile_idx + tl.arange(0, K_TILE_SIZE)
        if  is_causal:  
            mask = offs_k[None, :] > offs_q[:, None]
            S = tl.where(mask, float('-inf'), S)


        m_new = tl.maximum(m_i, tl.max(S, axis=1, keep_dims=True))
        P = tl.exp(S - m_new)
        l_new = l_i * tl.exp(m_i - m_new) + tl.sum(P, axis=1, keep_dims=True)
        acc = acc * tl.exp(m_i - m_new) + tl.dot(P, v)

        m_i = m_new
        l_i = l_new

    o = acc / l_i
    lse = m_i + tl.log(l_i)
    lse_1d = lse.reshape([Q_TILE_SIZE]) 
    tl.store(L_block_ptr, lse_1d)
    tl.store(O_block_ptr ,o)



class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size, seq_len, head_dim = Q.shape
        scale = 1.0 / math.sqrt(head_dim)
        input_dtype = Q.dtype

        # 内部用 FP32 计算，返回前 cast 回原始 dtype
        O = torch.zeros(batch_size, seq_len, head_dim, device=Q.device, dtype=torch.float32)
        L = torch.empty(batch_size, seq_len, device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = 32  
        K_TILE_SIZE = 32
        
        grid = lambda meta : (triton.cdiv(seq_len, meta["Q_TILE_SIZE"]), batch_size)
        flashattention_triton_fwd[grid](
        Q_ptr=Q, K_ptr=K, V_ptr=V,
        O_ptr=O, L_ptr=L,
        stride_qb=Q.stride(0), stride_qq=Q.stride(1), stride_qd=Q.stride(2),
        stride_kb=K.stride(0), stride_kk=K.stride(1), stride_kd=K.stride(2),
        stride_vb=V.stride(0), stride_vk=V.stride(1), stride_vd=V.stride(2),
        stride_ob=O.stride(0), stride_oa=O.stride(1), stride_od=O.stride(2),
        stride_lb=L.stride(0), stride_lq=L.stride(1),
        N_QUERIES=seq_len, N_KEYS=seq_len, scale=scale,
        is_causal=is_causal,
        D=head_dim,
        Q_TILE_SIZE=Q_TILE_SIZE,
        K_TILE_SIZE=K_TILE_SIZE
        )
        
        # 将输出转换回原始 dtype
        O = O.to(input_dtype)
        
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, gradout):

        Q, K, V, O, L = ctx.saved_tensors
        is_causal = ctx.is_causal
        scale = 1.0 / math.sqrt(Q.shape[-1])
        input_dtype = Q.dtype

        # 内部用 FP32 计算，返回前 cast 回原始 dtype
        Q = Q.float()
        K = K.float()
        V = V.float()
        O = O.float()
        gradout = gradout.float()
        # L 已经是 FP32

        D = (gradout * O).sum(dim=-1) #[batch, seq_len]
        S = Q @ K.transpose(-2, -1) * scale
        

        if is_causal:
            col_idx = torch.arange(0, S.shape[-1], device=Q.device).unsqueeze(0)
            row_idx = torch.arange(0, S.shape[-2], device=Q.device).unsqueeze(1)
            mask = col_idx > row_idx
            S = S.masked_fill(mask, float('-inf'))

        P = torch.exp(S - L.unsqueeze(-1))

        dP = gradout @ V.transpose(-2, -1)
        dS = P * (dP - D.unsqueeze(-1))
        if is_causal:
            dS = dS.masked_fill(mask, 0.0)

        dQ = dS @ K * scale
        dK = dS.transpose(-2, -1) @ Q * scale 
        dV = P.transpose(-2, -1) @ gradout

        return dQ.to(input_dtype), dK.to(input_dtype), dV.to(input_dtype), None
