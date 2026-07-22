from __future__ import annotations

import torch
from torch import nn

from cs336_basics.Part3.embedding import Embedding
from cs336_basics.Part3.linear import Linear
from cs336_basics.Part3.positionwise_feedforward import SiLUFeedForward
from cs336_basics.Part3.rmsnorm import RMSNorm
from cs336_basics.Part3.transformer_block import FFNType, NormMode, TransformerBlock


class TransformerLM(nn.Module):
    """由 token embedding、多个 Transformer block 与语言模型头组成的因果语言模型。"""

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    token_embeddings: Embedding
    layers: nn.ModuleList
    norm_mode: NormMode
    use_rope: bool
    ffn_type: FFNType
    ffn_hidden_dim: int
    ln_final: RMSNorm | None
    lm_head: Linear

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        norm_mode: NormMode = "pre",
        use_rope: bool = True,
        ffn_type: FFNType = "swiglu",
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero.")
        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if num_layers < 0:
            raise ValueError("num_layers must not be negative.")
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if d_ff <= 0:
            raise ValueError("d_ff must be greater than zero.")
        if norm_mode not in ("pre", "post", "none"):
            raise ValueError("norm_mode must be one of: 'pre', 'post', 'none'.")
        if ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type must be one of: 'swiglu', 'silu'.")
        if use_rope and (rope_theta is None or rope_theta <= 0):
            raise ValueError("rope_theta must be greater than zero when use_rope is True.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta if use_rope else None
        self.norm_mode = norm_mode
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        self.ffn_hidden_dim = (
            d_ff if ffn_type == "swiglu" else SiLUFeedForward.matched_hidden_dim(d_ff)
        )

        # token embedding 不叠加绝对位置编码；启用时由每个 block 中的 RoPE 提供位置信息。
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                    norm_mode=norm_mode,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        # Pre-Norm 需要 final RMSNorm；标准 Post-Norm 已在每个 residual 子层后归一化。
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_mode == "pre" else None
        # 输出头生成未归一化 logits；训练时由交叉熵损失函数负责 softmax。
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """返回每个输入 token 位置上的未归一化词表 logits。"""

        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch_size, sequence_length).")
        if token_ids.shape[-1] > self.context_length:
            raise ValueError(f"sequence length must not exceed context_length ({self.context_length}).")

        hidden_states = self.token_embeddings(token_ids)
        # 未传入 token_positions 时，block 使用 [0, ..., sequence_length - 1] 的连续位置。
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        if self.ln_final is not None:
            hidden_states = self.ln_final(hidden_states)
        return self.lm_head(hidden_states)
