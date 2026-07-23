from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from cs336_basics import bpe as _bpe
from cs336_basics import model as _model
from cs336_basics import optim as _optim


def run_linear(d_in, d_out, weights, in_features):
    m = _model.Linear(d_in, d_out)
    m.weight.data = weights
    return m(in_features)


def run_embedding(vocab_size, d_model, weights, token_ids):
    m = _model.Embedding(vocab_size, d_model)
    m.weight.data = weights
    return m(token_ids)


def run_swiglu(d_model, d_ff, w1_weight, w2_weight, w3_weight, in_features):
    m = _model.SwiGLU(d_model, d_ff)
    m.w1.weight.data = w1_weight
    m.w2.weight.data = w2_weight
    m.w3.weight.data = w3_weight
    return m(in_features)


def run_scaled_dot_product_attention(Q, K, V, mask=None):
    return _model.scaled_dot_product_attention(Q, K, V, mask=mask)


def _load_mha(m, q_w, k_w, v_w, o_w):
    m.q_proj.weight.data = q_w
    m.k_proj.weight.data = k_w
    m.v_proj.weight.data = v_w
    m.output_proj.weight.data = o_w


def run_multihead_self_attention(
    d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features
):
    m = _model.MultiHeadSelfAttention(d_model, num_heads, rope=None)
    _load_mha(m, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    return m(in_features)


def run_multihead_self_attention_with_rope(
    d_model, num_heads, max_seq_len, theta,
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight,
    in_features, token_positions=None,
):
    rope = _model.RoPE(theta, d_model // num_heads, max_seq_len)
    m = _model.MultiHeadSelfAttention(d_model, num_heads, rope=rope)
    _load_mha(m, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight)
    return m(in_features, positions=token_positions)


def run_rope(d_k, theta, max_seq_len, in_query_or_key, token_positions):
    rope = _model.RoPE(theta, d_k, max_seq_len)
    return rope(in_query_or_key, token_positions)


def _load_block(m, weights):
    _load_mha(m.attn, weights["attn.q_proj.weight"], weights["attn.k_proj.weight"],
              weights["attn.v_proj.weight"], weights["attn.output_proj.weight"])
    m.ln1.weight.data = weights["ln1.weight"]
    m.ln2.weight.data = weights["ln2.weight"]
    m.ffn.w1.weight.data = weights["ffn.w1.weight"]
    m.ffn.w2.weight.data = weights["ffn.w2.weight"]
    m.ffn.w3.weight.data = weights["ffn.w3.weight"]


def run_transformer_block(d_model, num_heads, d_ff, max_seq_len, theta, weights, in_features):
    m = _model.TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta)
    _load_block(m, weights)
    return m(in_features)


def run_transformer_lm(
    vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
    weights, in_indices,
):
    m = _model.TransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta)
    m.token_embeddings.weight.data = weights["token_embeddings.weight"]
    m.ln_final.weight.data = weights["ln_final.weight"]
    m.lm_head.weight.data = weights["lm_head.weight"]
    for i, blk in enumerate(m.layers):
        prefix = f"layers.{i}."
        block_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
        _load_block(blk, block_weights)
    return m(in_indices)


def run_rmsnorm(d_model, eps, weights, in_features):
    m = _model.RMSNorm(d_model, eps=eps)
    m.weight.data = weights
    return m(in_features)


def run_silu(in_features):
    return _model.silu(in_features)


def run_get_batch(dataset, batch_size, context_length, device):
    return _optim.get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features, dim):
    return _model.softmax(in_features, dim=dim)


def run_cross_entropy(inputs, targets):
    return _model.cross_entropy(inputs, targets)


def run_gradient_clipping(parameters, max_l2_norm):
    return _optim.clip_grad_l2(list(parameters), max_l2_norm)


def get_adamw_cls():
    return _optim.AdamW


def run_get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    return _optim.cosine_lr(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(model, optimizer, iteration, out):
    return _optim.save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(src, model, optimizer):
    return _optim.load_checkpoint(src, model, optimizer)


def get_tokenizer(vocab, merges, special_tokens=None):
    return _bpe.BPETokenizer(vocab, merges, special_tokens)


def run_train_bpe(input_path, vocab_size, special_tokens, **kwargs):
    return _bpe.train_bpe(input_path, vocab_size, special_tokens, num_processes=kwargs.get("num_processes", 4))
