import torch
import math

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    计算缩放点积注意力。
    
    参数:
        query: 形状为 (..., seq_len, d_k) 的张量
        key:   形状为 (..., seq_len, d_k) 的张量
        value: 形状为 (..., seq_len, d_v) 的张量
        mask:  形状为 (seq_len, seq_len) 的布尔张量 (可选)。
               True 表示参与注意力计算，False 表示屏蔽该位置。
               
    返回:
        形状为 (..., seq_len, d_v) 的张量
    """
    # 1. 获取 d_k (query 最后一个维度的大小)
    d_k = query.size(-1)
    
    # 2. 计算未缩放的注意力分数 Q * K^T
    # 使用 transpose(-2, -1) 安全地交换最后两个维度，完美兼容任意数量的前置 batch 维度
    # Q: (..., seq_len, d_k) @ K^T: (..., d_k, seq_len) -> scores: (..., seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1))
    
    # 3. 缩放 (除以 sqrt(d_k))
    scores = scores / math.sqrt(d_k)
    
    # 4. 应用 Mask (如果提供)
    if mask is not None:
        # 作业要求: mask 值为 False 的位置，概率应该为 0。
        # 我们使用 masked_fill_ 将 ~mask (即 False 的位置) 替换为负无穷大 (-inf)
        # 这样在经过 softmax 时，e^(-inf) 就会变成 0
        scores = scores.masked_fill(~mask, float('-inf'))
        
    # 5. 应用 Softmax 获取注意力权重
    # 在最后一个维度 (dim=-1) 上计算，保证每一行的概率总和为 1
    # 提示: 你可以在这里调用你上一题手写的 softmax，或者直接用 PyTorch 原生的
    attention_weights = torch.softmax(scores, dim=-1)
    
    # 6. 将注意力权重与 Value 相乘
    # weights: (..., seq_len, seq_len) @ V: (..., seq_len, d_v) -> output: (..., seq_len, d_v)
    output = torch.matmul(attention_weights, value)
    
    return output