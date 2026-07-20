import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        embedding_matrix = torch.empty(
            num_embeddings, embedding_dim, device=device, dtype=dtype
        )
        nn.init.trunc_normal_(
            embedding_matrix,
            mean=0.0,
            std=1,
            a=-3.0,
            b=3.0,
        )
        self.weight = nn.Parameter(embedding_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        assert token_ids.dtype == torch.long
        return self.weight[token_ids]
