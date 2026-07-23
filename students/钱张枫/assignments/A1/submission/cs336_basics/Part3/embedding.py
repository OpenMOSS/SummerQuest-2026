from __future__ import annotations

import torch
from torch import nn


class Embedding(nn.Module):
    """词嵌入层，根据 token id 查表返回对应的 embedding 向量。"""

    num_embeddings: int
    embedding_dim: int
    weight: nn.Parameter

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # 保留词表大小和向量维度，便于调试和与 PyTorch nn.Embedding 接口对齐。
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        # embedding matrix 形状为 (num_embeddings, embedding_dim)，d_model 位于最后一维。
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 按作业要求使用截断正态初始化，限制极端初始值。
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 通过张量索引完成查表，输出形状为 token_ids.shape + (embedding_dim,)。
        return self.weight[token_ids]
