from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO, IO

import numpy as np
import regex as re
import torch
from torch import Tensor


GPT2_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def run_linear(d_in: int, d_out: int, weights: Tensor, in_features: Tensor) -> Tensor:
    del d_in, d_out
    return in_features @ weights.transpose(-1, -2)


def run_embedding(vocab_size: int, d_model: int, weights: Tensor, token_ids: Tensor) -> Tensor:
    del vocab_size, d_model
    return weights[token_ids]


def run_silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    del d_model, d_ff
    gate = run_silu(run_linear(w1_weight.shape[1], w1_weight.shape[0], w1_weight, in_features))
    value = run_linear(w3_weight.shape[1], w3_weight.shape[0], w3_weight, in_features)
    return run_linear(w2_weight.shape[1], w2_weight.shape[0], w2_weight, gate * value)


def run_rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    del d_model
    dtype = in_features.dtype
    x = in_features.float()
    normalized = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return normalized.to(dtype) * weights


def run_softmax(in_features: Tensor, dim: int) -> Tensor:
    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def run_scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask.to(dtype=torch.bool, device=scores.device), torch.finfo(scores.dtype).min)
    weights = run_softmax(scores, dim=-1)
    return torch.matmul(weights, V)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    del max_seq_len
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")
    x = in_query_or_key
    positions = token_positions.to(device=x.device)
    while positions.ndim < x.ndim - 1:
        positions = positions.unsqueeze(-2)
    inv_freq = theta ** (-torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32) / d_k)
    angles = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles).to(dtype=x.dtype)
    sin = torch.sin(angles).to(dtype=x.dtype)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def _project_heads(in_features: Tensor, weight: Tensor, num_heads: int) -> Tensor:
    projected = run_linear(weight.shape[1], weight.shape[0], weight, in_features)
    *leading, sequence_length, d_model = projected.shape
    d_head = d_model // num_heads
    return projected.reshape(*leading, sequence_length, num_heads, d_head).transpose(-3, -2)


def _merge_heads(in_features: Tensor) -> Tensor:
    *leading, num_heads, sequence_length, d_head = in_features.shape
    return in_features.transpose(-3, -2).reshape(*leading, sequence_length, num_heads * d_head)


def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    del d_model
    q = _project_heads(in_features, q_proj_weight, num_heads)
    k = _project_heads(in_features, k_proj_weight, num_heads)
    v = _project_heads(in_features, v_proj_weight, num_heads)
    sequence_length = in_features.shape[-2]
    attended = run_scaled_dot_product_attention(q, k, v, _causal_mask(sequence_length, in_features.device))
    return run_linear(o_proj_weight.shape[1], o_proj_weight.shape[0], o_proj_weight, _merge_heads(attended))


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    del d_model
    q = _project_heads(in_features, q_proj_weight, num_heads)
    k = _project_heads(in_features, k_proj_weight, num_heads)
    v = _project_heads(in_features, v_proj_weight, num_heads)
    sequence_length = in_features.shape[-2]
    if token_positions is None:
        token_positions = torch.arange(sequence_length, device=in_features.device)
    d_head = q.shape[-1]
    q = run_rope(d_head, theta, max_seq_len, q, token_positions)
    k = run_rope(d_head, theta, max_seq_len, k, token_positions)
    attended = run_scaled_dot_product_attention(q, k, v, _causal_mask(sequence_length, in_features.device))
    return run_linear(o_proj_weight.shape[1], o_proj_weight.shape[0], o_proj_weight, _merge_heads(attended))


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    x_norm = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)
    attn = run_multihead_self_attention_with_rope(
        d_model,
        num_heads,
        max_seq_len,
        theta,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        x_norm,
    )
    x = in_features + attn
    y_norm = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)
    ffn = run_swiglu(
        d_model,
        d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        y_norm,
    )
    return x + ffn


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    x = run_embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)
    for layer_index in range(num_layers):
        prefix = f"layers.{layer_index}."
        block_weights = {key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)}
        x = run_transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, block_weights, x)
    x = run_rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)
    return run_linear(d_model, vocab_size, weights["lm_head.weight"], x)


class TransformerLM(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_variant: str = "swiglu",
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_rmsnorm = use_rmsnorm
        self.norm_position = norm_position
        self.use_rope = use_rope
        self.ffn_variant = ffn_variant

        self.token_embeddings = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        self.lm_head = torch.nn.Parameter(torch.empty(vocab_size, d_model))
        self.layers = torch.nn.ParameterList()
        for _ in range(num_layers):
            self.layers.extend(
                [
                    torch.nn.Parameter(torch.empty(d_model, d_model)),
                    torch.nn.Parameter(torch.empty(d_model, d_model)),
                    torch.nn.Parameter(torch.empty(d_model, d_model)),
                    torch.nn.Parameter(torch.empty(d_model, d_model)),
                    torch.nn.Parameter(torch.ones(d_model)),
                    torch.nn.Parameter(torch.empty(d_ff, d_model)),
                    torch.nn.Parameter(torch.empty(d_model, d_ff)),
                    torch.nn.Parameter(torch.empty(d_ff, d_model)),
                    torch.nn.Parameter(torch.ones(d_model)),
                ]
            )
        self.ln_final = torch.nn.Parameter(torch.ones(d_model))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 0.02
        torch.nn.init.normal_(self.token_embeddings, mean=0.0, std=std)
        torch.nn.init.normal_(self.lm_head, mean=0.0, std=std)
        for index, param in enumerate(self.layers):
            if index % 9 in {4, 8}:
                torch.nn.init.ones_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=std)
        torch.nn.init.ones_(self.ln_final)

    def _layer_params(self, layer_index: int) -> dict[str, Tensor]:
        offset = layer_index * 9
        keys = [
            "attn.q_proj.weight",
            "attn.k_proj.weight",
            "attn.v_proj.weight",
            "attn.output_proj.weight",
            "ln1.weight",
            "ffn.w1.weight",
            "ffn.w2.weight",
            "ffn.w3.weight",
            "ln2.weight",
        ]
        return {key: self.layers[offset + idx] for idx, key in enumerate(keys)}

    def _maybe_norm(self, x: Tensor, weight: Tensor) -> Tensor:
        if not self.use_rmsnorm:
            return x
        return run_rmsnorm(self.d_model, 1e-5, weight, x)

    def _attention(self, x: Tensor, params: dict[str, Tensor]) -> Tensor:
        if self.use_rope:
            return run_multihead_self_attention_with_rope(
                self.d_model,
                self.num_heads,
                self.context_length,
                self.rope_theta,
                params["attn.q_proj.weight"],
                params["attn.k_proj.weight"],
                params["attn.v_proj.weight"],
                params["attn.output_proj.weight"],
                x,
            )
        return run_multihead_self_attention(
            self.d_model,
            self.num_heads,
            params["attn.q_proj.weight"],
            params["attn.k_proj.weight"],
            params["attn.v_proj.weight"],
            params["attn.output_proj.weight"],
            x,
        )

    def _ffn(self, x: Tensor, params: dict[str, Tensor]) -> Tensor:
        if self.ffn_variant == "silu":
            hidden = run_silu(run_linear(self.d_model, self.d_ff, params["ffn.w1.weight"], x))
            return run_linear(self.d_ff, self.d_model, params["ffn.w2.weight"], hidden)
        return run_swiglu(
            self.d_model,
            self.d_ff,
            params["ffn.w1.weight"],
            params["ffn.w2.weight"],
            params["ffn.w3.weight"],
            x,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        x = self.token_embeddings[token_ids]
        for layer_index in range(self.num_layers):
            params = self._layer_params(layer_index)
            if self.norm_position == "post":
                x = self._maybe_norm(x + self._attention(x, params), params["ln1.weight"])
                x = self._maybe_norm(x + self._ffn(x, params), params["ln2.weight"])
            else:
                x = x + self._attention(self._maybe_norm(x, params["ln1.weight"]), params)
                x = x + self._ffn(self._maybe_norm(x, params["ln2.weight"]), params)
        x = self._maybe_norm(x, self.ln_final)
        return run_linear(self.d_model, self.vocab_size, self.lm_head, x)


def run_get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[start : start + context_length] for start in starts])
    y = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return torch.as_tensor(x, dtype=torch.long, device=device), torch.as_tensor(y, dtype=torch.long, device=device)


def run_cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    log_probs = shifted - torch.log(torch.sum(torch.exp(shifted), dim=-1, keepdim=True))
    return -log_probs[torch.arange(targets.numel(), device=targets.device), targets.reshape(-1)].mean()


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [param for param in parameters if param.grad is not None]
    if not params:
        return
    total = torch.sqrt(sum(torch.sum(param.grad.detach() ** 2) for param in params))
    if total > max_l2_norm:
        scale = max_l2_norm / (total + 1e-6)
        for param in params:
            param.grad.mul_(scale)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                param.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step = state["step"]
                corrected_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                param.addcdiv_(exp_avg, torch.sqrt(exp_avg_sq).add_(eps), value=-corrected_lr)
        return loss


def get_adamw_cls() -> Any:
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


def _split_special(text: str, special_tokens: list[str]) -> Iterator[tuple[str, bool]]:
    if not special_tokens:
        yield text, False
        return
    pattern = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
    start = 0
    for match in pattern.finditer(text):
        if match.start() > start:
            yield text[start : match.start()], False
        yield match.group(0), True
        start = match.end()
    if start < len(text):
        yield text[start:], False


def _merge_token_sequence(tokens: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged = pair[0] + pair[1]
    output: list[bytes] | None = None
    index = 0
    while index < len(tokens):
        if index < len(tokens) - 1 and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            if output is None:
                output = list(tokens[:index])
            output.append(merged)
            index += 2
        else:
            if output is not None:
                output.append(tokens[index])
            index += 1
    return tokens if output is None else tuple(output)


def _merge_id_sequence(tokens: tuple[int, ...], pair: tuple[int, int], merged_id: int) -> tuple[int, ...]:
    output: list[int] | None = None
    index = 0
    while index < len(tokens):
        if index < len(tokens) - 1 and tokens[index] == pair[0] and tokens[index + 1] == pair[1]:
            if output is None:
                output = list(tokens[:index])
            output.append(merged_id)
            index += 2
        else:
            if output is not None:
                output.append(tokens[index])
            index += 1
    return tokens if output is None else tuple(output)


def _add_word_pairs(
    word: tuple[int, ...],
    count: int,
    pair_counts: Counter[tuple[int, int]],
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> None:
    seen: set[tuple[int, int]] = set()
    for pair in zip(word, word[1:]):
        pair_counts[pair] += count
        seen.add(pair)
    for pair in seen:
        pair_to_words[pair].add(word)


def _remove_word_pairs(
    word: tuple[int, ...],
    count: int,
    pair_counts: Counter[tuple[int, int]],
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]],
) -> None:
    seen: set[tuple[int, int]] = set()
    for pair in zip(word, word[1:]):
        pair_counts[pair] -= count
        if pair_counts[pair] <= 0:
            del pair_counts[pair]
        seen.add(pair)
    for pair in seen:
        words = pair_to_words.get(pair)
        if words is not None:
            words.discard(word)
            if not words:
                pair_to_words.pop(pair, None)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs: Any,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    del kwargs
    vocab: dict[int, bytes] = {index: bytes([index]) for index in range(256)}
    for special_token in special_tokens:
        token_bytes = special_token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    text = Path(input_path).read_text(encoding="utf-8")
    pretoken_counts: Counter[tuple[int, ...]] = Counter()
    for piece, is_special in _split_special(text, special_tokens):
        if is_special or not piece:
            continue
        for match in GPT2_PRETOKEN_PATTERN.finditer(piece):
            token = tuple(match.group(0).encode("utf-8"))
            if token:
                pretoken_counts[token] += 1

    merges: list[tuple[bytes, bytes]] = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for token, count in pretoken_counts.items():
        _add_word_pairs(token, count, pair_counts, pair_to_words)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break
        best_pair, _ = max(pair_counts.items(), key=lambda item: (item[1], (vocab[item[0][0]], vocab[item[0][1]])))
        merged_id = len(vocab)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[merged_id] = vocab[best_pair[0]] + vocab[best_pair[1]]

        affected_words = list(pair_to_words.get(best_pair, ()))
        merged_counts: Counter[tuple[int, ...]] = Counter()
        for token in affected_words:
            count = pretoken_counts.pop(token, 0)
            if count == 0:
                continue
            _remove_word_pairs(token, count, pair_counts, pair_to_words)
            merged_counts[_merge_id_sequence(token, best_pair, merged_id)] += count

        for token, count in merged_counts.items():
            existing = pretoken_counts.pop(token, 0)
            if existing:
                _remove_word_pairs(token, existing, pair_counts, pair_to_words)
            total_count = existing + count
            pretoken_counts[token] = total_count
            _add_word_pairs(token, total_count, pair_counts, pair_to_words)
    return vocab, merges


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.special_tokens = list(special_tokens or [])
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                self.vocab[len(self.vocab)] = token_bytes
        self.token_to_id = {token: index for index, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        tokens = tuple(bytes([byte]) for byte in pretoken.encode("utf-8"))
        if not tokens:
            return []
        while len(tokens) > 1:
            candidates = ((self.merge_ranks[pair], pair) for pair in zip(tokens, tokens[1:]) if pair in self.merge_ranks)
            best = min(candidates, default=None)
            if best is None:
                break
            tokens = _merge_token_sequence(tokens, best[1])
        return [self.token_to_id[token] for token in tokens]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece, is_special in _split_special(text, self.special_tokens):
            if is_special:
                ids.append(self.token_to_id[piece.encode("utf-8")])
                continue
            for match in GPT2_PRETOKEN_PATTERN.finditer(piece):
                ids.extend(self._encode_pretoken(match.group(0)))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int] | Iterable[int]) -> str:
        data = b"".join(self.vocab[int(index)] for index in ids)
        return data.decode("utf-8", errors="replace")


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return BPETokenizer(vocab, merges, special_tokens)
