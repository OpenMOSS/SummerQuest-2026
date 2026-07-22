import torch
import torch.nn as nn
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.multihead_self_attention_rope import MultiHeadSelfAttentionWithRoPE
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.rope import RotaryPositionalEmbedding
class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block (Section 3.4, Figure 2):
    y = x + MultiHeadSelfAttention(RMSNorm(x))
    z = y + FFN(RMSNorm(y))
"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.rope_module = RotaryPositionalEmbedding(d_k=d_model//num_heads, max_seq_len=max_seq_len, theta=theta)
        self.attn  = MultiHeadSelfAttentionWithRoPE(d_model=d_model, num_heads=num_heads, rope_module=self.rope_module)
        self.norm2 = RMSNorm(d_model)
        self.ffn   = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, token_positions) -> torch.Tensor:
        # sublayer 1: pre-norm + causal MHSA + residual
        x = x + self.attn(self.norm1(x), token_positions)
        # sublayer 2: pre-norm + SwiGLU FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x