import torch
import torch.nn as nn
import math

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_module=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # 接收外部传入的 RoPE 实例
        self.rope = rope_module
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        x 形状: (..., seq_len, d_model)
        token_positions 形状: (..., seq_len)
        """
        # 1. 动态获取前面的所有 batch 维度和 seq_len
        batch_dims = x.shape[:-2]
        seq_len = x.shape[-2]
        
        # 2. QKV 融合投影 -> (..., seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 3. 维度重排: 把 heads 抽离出来
        # (..., seq_len, num_heads, d_k) -> (..., num_heads, seq_len, d_k)
        # 注意：这里用 -3 和 -2 进行转置，可以完美兼容前面有任意多个 batch 维度的情况
        q = q.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        
        # 4. 核心逻辑：应用 RoPE
        if self.rope is not None and token_positions is not None:
            # token_positions 目前是 (..., seq_len)
            # 但 Q 和 K 是 (..., num_heads, seq_len, d_k)
            # 为了让 token_positions 能广播到所有 head 上，我们必须在倒数第 2 个位置插入一个维度 1
            # 变化: (..., seq_len) -> (..., 1, seq_len)
            token_positions_broadcast = token_positions.unsqueeze(-2)
            
            # 【严格遵守作业要求】: RoPE 只应用给 Q 和 K，绝对不能给 V！
            q = self.rope(q, token_positions_broadcast)
            k = self.rope(k, token_positions_broadcast)
        
        # 5. 计算 Scaled Dot-Product Attention
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        
        # 6. 恢复形状: (..., num_heads, seq_len, d_k) -> (..., seq_len, num_heads, d_k) -> (..., seq_len, d_model)
        # 这里必须紧跟 contiguous()，之前遇到的 cuDNN SDPA 报错
        context = context.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, self.d_model)
        
        out = self.out_proj(context)
        
        return out