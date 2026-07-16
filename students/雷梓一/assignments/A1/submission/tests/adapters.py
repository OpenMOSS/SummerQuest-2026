from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics.attention import CausalMultiHeadSelfAttention, RotaryPositionalEmbedding, scaled_dot_product_attention
from cs336_basics.nn import Embedding, Linear, RMSNorm, SwiGLU, silu
from cs336_basics.tokenizer import Tokenizer, train_bpe
from cs336_basics.training import (
    AdamW,
    clip_gradients,
    cosine_learning_rate,
    cross_entropy,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.transformer import TransformerBlock, TransformerLM


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    module = Linear(d_in, d_out, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({"weight": weights})
    return module(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    module = Embedding(vocab_size, d_model, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({"weight": weights})
    return module(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    module = SwiGLU(d_model, d_ff, device=w1_weight.device, dtype=w1_weight.dtype)
    module.load_state_dict({"w1.weight": w1_weight, "w2.weight": w2_weight, "w3.weight": w3_weight})
    return module(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return scaled_dot_product_attention(Q, K, V, mask)


def _load_attention(
    module: CausalMultiHeadSelfAttention,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
) -> None:
    module.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    module = CausalMultiHeadSelfAttention(
        d_model,
        num_heads,
        in_features.shape[-2],
        use_rope=False,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    _load_attention(module, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    return module(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    module = CausalMultiHeadSelfAttention(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        use_rope=True,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    _load_attention(module, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    return module(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    return RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=in_query_or_key.device)(
        in_query_or_key, token_positions
    )


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    module = TransformerBlock(
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        device=in_features.device,
        dtype=in_features.dtype,
    )
    module.load_state_dict(weights)
    return module(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    sample_weight = next(iter(weights.values()))
    module = TransformerLM(
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta,
        device=in_indices.device,
        dtype=sample_weight.dtype,
    )
    module.load_state_dict(weights)
    return module(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    module = RMSNorm(d_model, eps, device=weights.device, dtype=weights.dtype)
    module.load_state_dict({"weight": weights})
    return module(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from cs336_basics.attention import softmax

    return softmax(in_features, dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    clip_gradients(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    return cosine_learning_rate(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


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
    return train_bpe(input_path, vocab_size, special_tokens, **kwargs)
