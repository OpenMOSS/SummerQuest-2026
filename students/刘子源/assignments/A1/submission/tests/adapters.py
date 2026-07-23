# 官方测试只认下面这些函数名。这里就是转接头，别往里塞核心逻辑。

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any, BinaryIO

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import (
    Embedding,
    Linear,
    MultiheadSelfAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TransformerBlock,
    TransformerLM,
    scaled_dot_product_attention,
    silu,
)
from cs336_basics.nn_utils import cross_entropy, gradient_clipping, softmax
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.tokenizer import Tokenizer, train_bpe


def run_linear(d_in: int, d_out: int, weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    layer = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(in_features)


def run_embedding(
    vocab_size: int, d_model: int, weights: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    layer = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    layer = SwiGLU(d_model, d_ff, device=in_features.device, dtype=in_features.dtype)
    layer.load_state_dict(
        {"w1.weight": w1_weight, "w2.weight": w2_weight, "w3.weight": w3_weight}
    )
    return layer(in_features)


def run_scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    return scaled_dot_product_attention(Q, K, V, mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
) -> torch.Tensor:
    layer = MultiheadSelfAttention(
        d_model,
        num_heads,
        max_seq_len=in_features.shape[-2],
        use_rope=False,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    layer.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )
    return layer(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    layer = MultiheadSelfAttention(
        d_model,
        num_heads,
        max_seq_len=max_seq_len,
        theta=theta,
        use_rope=True,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    layer.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )
    return layer(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor],
    in_features: torch.Tensor,
) -> torch.Tensor:
    block = TransformerBlock(
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    block.load_state_dict(weights)
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: torch.Tensor,
) -> torch.Tensor:
    model = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=in_indices.device,
        dtype=weights["token_embeddings.weight"].dtype,
    )
    model.load_state_dict(weights)
    return model(in_indices)


def run_rmsnorm(
    d_model: int, eps: float, weights: torch.Tensor, in_features: torch.Tensor
) -> torch.Tensor:
    layer = RMSNorm(d_model, eps=eps, device=weights.device, dtype=weights.dtype)
    layer.weight.data.copy_(weights)
    return layer(in_features)


def run_silu(in_features: torch.Tensor) -> torch.Tensor:
    return silu(in_features)


def run_get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    return softmax(in_features, dim)


def run_cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    # 测试要的是类，不是实例，别顺手加括号。
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return get_lr_cosine_schedule(
        it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters
    )


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(Path(input_path), vocab_size, special_tokens, **kwargs)
