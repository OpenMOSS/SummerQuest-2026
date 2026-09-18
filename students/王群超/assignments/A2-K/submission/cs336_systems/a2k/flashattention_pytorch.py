import torch
import math

class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False, block_size=32):
        batch_size, seq_len, head_dim = Q.shape #[B, N, d]
        scale = 1.0/math.sqrt(head_dim)
        input_dtype = Q.dtype

        # 内部用 FP32 计算，返回前 cast 回原始 dtype
        q_float = Q.float()
        k_float = K.float()
        v_float = V.float()

        #初始化O和L
        O = torch.zeros_like(q_float)
        L = torch.empty(batch_size, seq_len, device=Q.device, dtype=torch.float32)


        for i in range(0, seq_len, block_size):
            i_end = min(i+block_size, seq_len)
            Q_i = q_float[: ,i : i_end , :]

            O_i = torch.zeros_like(Q_i) #[B, b_k, d]
            m_i = torch.full((batch_size, block_size, 1), float('-inf'), device=Q.device, dtype=torch.float32) #[B, b_k, 1]
            l_i = torch.zeros(batch_size, block_size, 1, device=Q.device, dtype=torch.float32)#[B, b_k, 1]
            for j in range(0, seq_len, block_size):
                
                j_end = min(j+block_size, seq_len)
                # 因果模式下，完全在右上角的块直接跳过
                if is_causal and j >= i_end:
                    continue

                K_j = k_float[:, j: j_end, :]
                V_j = v_float[:, j: j_end, :]

                S_ij = Q_i @ K_j.transpose(-2, -1) * scale #[B, H, b_k, b_k]

                if is_causal:
                    row_idx = torch.arange(i, i_end, device=Q.device).unsqueeze(1)
                    col_idx = torch.arange(j, j_end, device=Q.device).unsqueeze(0)
                    mask = col_idx > row_idx

                    mask = mask.unsqueeze(0).unsqueeze(0)
                    S_ij = S_ij.masked_fill(mask, float('-inf'))



                m_new = torch.maximum(m_i, S_ij.max(dim=-1, keepdim=True).values)#新的最大值4
            
                P = torch.exp(S_ij - m_new)#[B, H, b_k, b_k]
            
                l_new = l_i * torch.exp(m_i-m_new) + P.sum(dim=-1, keepdim=True)#[B, H, b_k, b_k]
            
                O_i = O_i * torch.exp(m_i-m_new) + P @ V_j

                m_i = m_new
                l_i = l_new
            
            O_i = O_i / l_i
            O[:, i: i_end, :] = O_i
            
            L[:, i: i_end] = (m_i + torch.log(l_i)).squeeze(-1)

        O = O.to(input_dtype)

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        ctx.block_size = block_size
    
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

        return dQ.to(input_dtype), dK.to(input_dtype), dV.to(input_dtype), None, None