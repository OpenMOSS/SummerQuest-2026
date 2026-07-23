import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # 确保 d_model 能被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # --- Stretch Goal: 合并 Q, K, V 投影 ---
        # 我们用一个线性层，输出维度是 3 * d_model
        # 注意: 现代大模型通常不加 bias (bias=False)，但为了兼容常规作业测试，这里保留默认的 bias=True
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (batch_size, seq_len, d_model)
        """
        B, T, C = x.size()
        
        # 1. 一次性计算 Q, K, V
        # qkv 形状: (B, T, 3 * C)
        qkv = self.qkv_proj(x)
        
        # 2. 将其在最后一个维度切分为 q, k, v (每个形状都是 (B, T, C))
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 3. 维度重排: (B, T, C) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        # 这一步把不同的头分离到了独立的维度上，方便后面做并行的矩阵乘法
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        # 4. 构建因果掩码 (Causal Mask)
        # 生成一个下三角矩阵 (包含对角线)，形状为 (T, T)
        # True 表示可以看，False 表示不能看 (未来的 Token)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        
        # 5. 调用你上一题实现的缩放点积注意力 (或者这里直接写死公式)
        # 注意：这里我们手写一遍公式，展示 mask 的应用逻辑
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 将 mask 为 False 的地方替换为 -inf
        scores = scores.masked_fill(~mask, float('-inf'))
        # 计算注意力权重
        attn_weights = torch.softmax(scores, dim=-1)
        # 与 Value 相乘: (B, num_heads, T, T) @ (B, num_heads, T, d_k) -> (B, num_heads, T, d_k)
        context = torch.matmul(attn_weights, v)
        
        # 6. 拼接多头并恢复形状
        # (B, num_heads, T, d_k) -> (B, T, num_heads, d_k) -> (B, T, C)
        # ⚠️ 极其关键: transpose 破坏了内存连续性，必须调用 contiguous()，否则 view 会报错，或者反向传播时触发你之前遇到的 cuDNN SDPA 警告！
        context = context.transpose(1, 2).contiguous().view(B, T, C)
        
        # 7. 最后的线性投影
        out = self.out_proj(context)
        
        return out