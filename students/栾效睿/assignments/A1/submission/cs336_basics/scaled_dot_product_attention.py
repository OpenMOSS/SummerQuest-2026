import torch
from .softmax import softmax


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    masked: torch.Tensor | None = None,
):
    d_k = Q.shape[-1]
    scores = torch.einsum("... q d,... k d -> ... q k", Q, K) / (d_k**0.5)
    if masked is not None:
        scores = scores.masked_fill(~masked, float("-inf"))
    return softmax(x=scores, dim=-1) @ V
