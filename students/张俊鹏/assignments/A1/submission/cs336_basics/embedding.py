import torch
import torch.nn as nn

"""
对于输入的 token id，返回其对应的向量
"""

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # 形状规定为 (num_embeddings, embedding_dim)，即 (词表大小, 词向量维度)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        
        nn.init.trunc_normal_(
            self.weight, 
            mean=0.0, 
            std=1.0, 
            a=-3.0, 
            b=3.0
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        根据给定的 token IDs 查找并返回对应的嵌入向量。
        """
        return self.weight[token_ids]