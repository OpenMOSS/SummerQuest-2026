import torch
import torch.nn as nn
from einops import rearrange

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        super().__init__()            
        self.d_in = d_in
        self.d_out = d_out                 
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        std = (2 / (d_in + d_out)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = 1
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) 
        x_mean = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(x_mean + self.eps)
        result = x / rms * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
    
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class SiLUFFN(nn.Module):
    """普通 SiLU FFN(无门控),用于消融 SwiGLU。"""
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(silu(self.w1(x)))
    
    
def silu(x):
    return x * torch.sigmoid(x)

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        freqs = 1.0 / theta ** (torch.arange(0, d_k, 2) / d_k)
        pos = torch.arange(max_seq_len)
        angles = pos[:, None] * freqs[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        result = torch.stack((x1_rot, x2_rot), dim=-1).flatten(-2)
        return result
    
def softmax(x, dim=-1):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d_k ** 0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ V

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, use_rope=False, max_seq_len=None, theta=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        if use_rope:
            self.rope = RoPE(theta, self.d_k, max_seq_len)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        Q = self.q_proj(x); K = self.k_proj(x); V = self.v_proj(x)

        Q = rearrange(Q, 'b s (h d) -> b h s d', h=self.num_heads)
        K = rearrange(K, 'b s (h d) -> b h s d', h=self.num_heads)
        V = rearrange(V, 'b s (h d) -> b h s d', h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                seq_len = x.shape[-2]
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
            

        seq_len = x.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = rearrange(out, 'b h s d -> b s (h d)')

        return self.output_proj(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta,
                 use_rmsnorm=True, norm_position="pre", use_rope=True, ffn_type="swiglu"):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.norm_position = norm_position

        self.ln1 = rmsnorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, use_rope=use_rope,
                                           max_seq_len=max_seq_len, theta=theta)
        self.ln2 = rmsnorm(d_model)
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff)
        else:                                  
            self.ffn = SiLUFFN(d_model, d_ff)

    def _norm1(self, x):
        return self.ln1(x) if self.use_rmsnorm else x
    def _norm2(self, x):
        return self.ln2(x) if self.use_rmsnorm else x

    def forward(self, x, token_positions):
        if self.norm_position == "pre":
            # Pre-Norm: x + sublayer(norm(x))
            z = x + self.attn(self._norm1(x), token_positions)
            y = z + self.ffn(self._norm2(z))
        else:
            # Post-Norm: norm(x + sublayer(x))
            z = self._norm1(x + self.attn(x, token_positions))
            y = self._norm2(z + self.ffn(z))
        return y
    
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
                 use_rmsnorm=True, norm_position="pre", use_rope=True, ffn_type="swiglu"):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta,
                             use_rmsnorm=use_rmsnorm, norm_position=norm_position,
                             use_rope=use_rope, ffn_type=ffn_type)
            for _ in range(num_layers)
        ])
        self.ln_final = rmsnorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids, token_positions=None):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
