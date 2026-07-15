"""Transformer 语言模型的自回归解码与 nucleus sampling。"""

from __future__ import annotations

import torch

from cs336_basics.nn import softmax


def sample_top_p(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    """从一维概率分布中按 top-p（nucleus）采样一个 token ID。"""
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    # 保留累积质量首次达到 top_p 的那个 token，构成最小 nucleus 集合。
    keep_sorted = (torch.cumsum(sorted_probabilities, dim=-1) - sorted_probabilities) < top_p
    nucleus_probabilities = sorted_probabilities * keep_sorted
    nucleus_probabilities = nucleus_probabilities / nucleus_probabilities.sum()
    sampled_sorted_index = torch.multinomial(nucleus_probabilities, num_samples=1)
    return sorted_indices[sampled_sorted_index]


@torch.inference_mode()
def decode(
    model: torch.nn.Module,
    prompt_token_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """从一维 prompt 自回归采样，并返回包含 prompt 的完整 token 序列。

    每一步只将最后 ``model.context_length`` 个 token 提供给模型；这使生成长度可以
    超过训练时的 context length。温度必须大于零，温度缩放后再应用 top-p sampling。
    """
    if prompt_token_ids.ndim != 1 or prompt_token_ids.numel() == 0:
        raise ValueError("prompt_token_ids must be a non-empty 1D tensor")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not hasattr(model, "context_length"):
        raise AttributeError("model must expose context_length for autoregressive decoding")

    generated = prompt_token_ids.clone()
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            context = generated[-model.context_length :].unsqueeze(0)
            next_logits = model(context)[0, -1]
            probabilities = softmax(next_logits / temperature, dim=-1)
            next_token = sample_top_p(probabilities, top_p)
            generated = torch.cat((generated, next_token.to(dtype=generated.dtype)))
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    finally:
        model.train(was_training)
    return generated


# generate 是更直观的别名，便于在生成脚本中使用。
generate = decode
