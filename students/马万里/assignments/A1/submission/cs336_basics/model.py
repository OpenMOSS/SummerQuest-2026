import torch 
import math
import torch.nn as nn 

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def forward(self, x):
        return x @ self.weight.T
    
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0, std=1.0)
    
    def forward(self, x):
        return self.weight[x]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        # 先转换成float32计算，避免溢出
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = (x_f32 / rms).to(x.dtype)
        return x_normed * self.weight

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = Linear(d_model, d_ff)
        self.W3 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(silu(self.W1(x)) * self.W3(x))
    
def rope(
    x: torch.Tensor,
    token_positions: torch.Tensor,
    d_k: int,
    theta: float
) -> torch.Tensor:
    assert d_k % 2 == 0, "RoPE requires even d_k"
    
    # 计算逆频率 1 / (theta^(2i/d_k))
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=x.device, dtype=torch.float32) / d_k))
    
    positions = token_positions.float().unsqueeze(-1)
    angles = positions * inv_freq
    
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 拆分奇偶维度
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    # 旋转
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    
    out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
    return out
    
def softmax(x, dim):
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None
):
    d_k = Q.size(-1)
    scores = torch.einsum('... q d, ... k d -> ... q k', Q, K) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    
    attn_weights = softmax(scores, dim=-1)
    
    output = torch.einsum('... q k, ... k d -> ... q d', attn_weights, V)
    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # 线性投影，然后拆分为多头
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 因果 mask：下三角为 True，上三角为 False
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        
        # 注意力计算
        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        
        # 合并多头：(B, num_heads, T, head_dim) -> (B, T, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.o_proj(attn_output)

class MultiHeadSelfAttentionWithRoPE(nn.Module):
    """多头自注意力，带有 RoPE 位置编码。"""

    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.theta = theta

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = x.shape

        # 投影并拆分多头
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 应用 RoPE
        if token_positions is not None:
            # 将 token_positions 形状调整为 (batch_size, 1, seq_len) 以便广播到多头
            if token_positions.dim() == 2:
                token_positions = token_positions.unsqueeze(1)  # (B, 1, T) 或 (1, 1, T)
            Q = rope(Q, token_positions, self.head_dim, self.theta)
            K = rope(K, token_positions, self.head_dim, self.theta)

        scores = torch.einsum('...qd,...kd->...qk', Q, K) / (self.head_dim ** 0.5)

        # 因果 mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~causal_mask, float("-inf"))

        attn_weights = softmax(scores, dim=-1)

        # 加权求和
        attn_output = torch.einsum('...qk,...kd->...qd', attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.o_proj(attn_output)
    
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        theta: float = 10000.0,
        ffn_cls=SwiGLU
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = ffn_cls(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        # 自动生成位置索引（如果模型没有外部传入）
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        block_type="pre_norm", # "pre_norm", "post_norm", "no_rmsnorm", "nope"
        use_silu_ffn=False,
    ):
        super().__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        
        if use_silu_ffn:
            ffn_cls = SiLUFFN
            d_ff = int(d_ff * 1.5)
        else:
            ffn_cls = SwiGLU

        if block_type == "pre_norm":
            block_cls = TransformerBlock
        elif block_type == "post_norm":
            block_cls = TransformerBlockPostNorm
        elif block_type == "no_rmsnorm":
            block_cls = TransformerBlockNoRMSNorm
        elif block_type == "nope":
            block_cls = TransformerBlockNoPE
        else:
            raise ValueError(f"Unknown block_type: {block_type}")
        
        self.layers = nn.ModuleList([
            block_cls(d_model, num_heads, d_ff, theta, ffn_cls)
            for _ in range(num_layers)
        ])
        
        if block_type != "no_rmsnorm":
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = nn.Identity()   # 无 norm
            
        self.lm_head = Linear(d_model, vocab_size)
        
    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
    
# 消融实验相关模型
class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.W2(silu(self.W1(x)))
    
class TransformerBlockPostNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000.0, ffn_cls=SwiGLU):
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta)
        self.ln1 = RMSNorm(d_model)
        self.ffn = ffn_cls(d_model, d_ff)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Post-Norm: 先 attention 后 norm，再残差
        attn_out = self.attn(x, token_positions)
        x = self.ln1(x + attn_out)

        # Post-Norm FFN
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x

class TransformerBlockNoRMSNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000.0, ffn_cls=SwiGLU):
        super().__init__()
        self.attn = MultiHeadSelfAttentionWithRoPE(d_model, num_heads, theta)
        self.ffn = ffn_cls(d_model, d_ff)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        x = x + self.attn(x, token_positions)
        x = x + self.ffn(x)
        return x
    
class TransformerBlockNoPE(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta=10000.0, ffn_cls=SwiGLU):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = ffn_cls(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x