import torch


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    inverse_scale = q.shape[-1] ** -0.5
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * inverse_scale

    if is_causal:
        mask = torch.tril(torch.ones(attn_scores.shape[-2:], device=attn_scores.device, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)
    return torch.matmul(attn_weights, v)
