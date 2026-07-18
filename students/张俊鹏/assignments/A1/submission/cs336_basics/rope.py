import torch
import torch.nn as nn

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    经典 RoPE (奇偶交错):
    [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
    """
    x_rotated = torch.empty_like(x)
    x_rotated[..., 0::2] = -x[..., 1::2] # 偶数位置变成负的奇数位置
    x_rotated[..., 1::2] = x[..., 0::2]  # 奇数位置变成偶数位置
    return x_rotated

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        
        # 【修改这里！】
        # 将 freqs 的每一个元素在最后一个维度上原地重复 2 次，以匹配奇偶交错的维度
        # 结果形状依然是: (max_seq_len, d_k)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        对输入 x 应用旋转位置编码。
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len)
        """
        # 1. 利用高级索引直接提取对应的 cos 和 sin
        # 无论 x 前面有多少个 batch 维度，抽出来后的形状都会完美对齐为 (..., seq_len, d_k)
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        
        # 2. 应用旋转变换公式: x_rotated = x * cos + rotate_half(x) * sin
        # 注意: 预计算的缓存通常是 float32，这里计算完需要将其转回 x 原本的数据类型 (如 bfloat16)
        x_rotated = (x * cos_pos) + (rotate_half(x) * sin_pos)
        
        return x_rotated.to(x.dtype)