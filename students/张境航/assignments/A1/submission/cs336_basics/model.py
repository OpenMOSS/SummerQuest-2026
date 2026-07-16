import torch
from torch import Tensor
from torch.nn import Module, Parameter, ModuleList
from cs336_basics.nn_utils import softmax

class Linear(Module):
    def __init__(self, d_in:int,d_out:int) -> None:
        super().__init__()

        self.d_in=d_in
        self.d_out=d_out

        self.weight = Parameter(torch.empty(d_out,d_in))

    def forward(self,x:Tensor) ->Tensor:
        return x @ self.weight.T
         
class Embedding(Module):
    def __init__(self,vocab_size:int,d_model:int)->None:
        super().__init__()

        self.vocab_size=vocab_size
        self.d_model=d_model

        self.weight=Parameter(torch.empty(vocab_size,d_model))

    def forward(self, token_ids:Tensor) -> Tensor:
        return self.weight[token_ids]
    
def silu(x: Tensor) -> Tensor:
        return x*torch.sigmoid(x)

class SwiGLU(Module):
    def __init__(self, d_model: int,d_ff:int) -> None:
        super().__init__()

        self.w1= Linear(d_in=d_model,d_out=d_ff)
        self.w2= Linear(d_in=d_ff,d_out=d_model)
        self.w3= Linear(d_in=d_model,d_out=d_ff)
    def forward(self,x:Tensor) -> Tensor:
        gate=silu(self.w1(x))
        value=self.w3(x)
        return self.w2(gate*value)

class RMSNorm(Module):
    def __init__(self, d_model: int,eps:float=1e-5)->None:
        super().__init__()
        
        self.d_model=d_model
        self.eps=eps
        self.weight=Parameter(torch.ones(d_model))

    def forward(self,x:Tensor) ->Tensor:
        original_dtype=x.dtype
        x_float =x.to(torch.float32)

        rms=torch.sqrt(torch.mean(x_float**2,dim=-1,keepdim=True)+self.eps)
        normalized=x_float/rms

        return normalized.to(original_dtype)*self.weight

def scaled_dot_product_attention(
    q:Tensor,
    k:Tensor,
    v:Tensor,
    mask:Tensor | None=None,
)-> Tensor:
    d_k=q.shape[-1]

    scores=q@k.transpose(-2, -1)
    scores= scores/(d_k**0.5)

    if mask is not None:
        scores=scores.masked_fill(~mask,float("-inf"))

    attention_weights=softmax(scores,dim=-1)

    return attention_weights @ v

class MultiHeadSelfAttention(Module):
    def __init__(self,d_model:int, num_heads:int) -> None:
        super().__init__()

        if d_model%num_heads !=0:
            raise ValueError("d_model must be divisible by num_heads")
        
        self.d_model=d_model
        self.num_heads=num_heads
        self.head_dim=d_model//num_heads
    
        self.q_proj=Linear(d_in=d_model,d_out=d_model)
        self.k_proj=Linear(d_in=d_model,d_out=d_model)
        self.v_proj=Linear(d_in=d_model,d_out=d_model)
        self.o_proj=Linear(d_in=d_model,d_out=d_model)

    def forward(self,x:Tensor) ->Tensor:
        *leading_dims, sequence_length,_=x.shape

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.reshape(*leading_dims,sequence_length,self.num_heads, self.head_dim)
        k=k.reshape(*leading_dims,sequence_length,self.num_heads, self.head_dim)
        v=v.reshape(*leading_dims,sequence_length,self.num_heads, self.head_dim)

        q=q.transpose(-3,-2)
        k=k.transpose(-3,-2)
        v=v.transpose(-3,-2)

        causal_mask=torch.tril(torch.ones((sequence_length,sequence_length),dtype=torch.bool,device=x.device))
        attended=scaled_dot_product_attention(q=q,k=k,v=v,mask=causal_mask)

        attended=attended.transpose(-3,-2)
        attended=attended.reshape(*leading_dims,sequence_length, self.d_model)

        return self.o_proj(attended)

class RoPE(Module):
    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even")

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        dimension_indices = torch.arange(0, d_k, 2, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimension_indices / d_k)

        positions = torch.arange(max_seq_len, dtype=torch.float32)

        angles = positions[:, None] * inverse_frequencies[None, :]

        self.register_buffer(
            "cos_cache",
            torch.cos(angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles),
            persistent=False,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor,
    ) -> Tensor:
        original_dtype = x.dtype

        cos = self.cos_cache[token_positions].to(
            device=x.device,
            dtype=torch.float32,
        )
        sin = self.sin_cache[token_positions].to(
            device=x.device,
            dtype=torch.float32,
        )

        x_float = x.to(torch.float32)

        x_even = x_float[..., 0::2]
        x_odd = x_float[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        rotated = torch.stack(
            [rotated_even, rotated_odd],
            dim=-1,
        )

        rotated = rotated.flatten(start_dim=-2)

        return rotated.to(original_dtype)
class MultiHeadSelfAttentionWithRoPE(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_in=d_model, d_out=d_model)
        self.k_proj = Linear(d_in=d_model, d_out=d_model)
        self.v_proj = Linear(d_in=d_model, d_out=d_model)
        self.o_proj = Linear(d_in=d_model, d_out=d_model)

        self.rope = RoPE(
            d_k=self.head_dim,
            theta=theta,
            max_seq_len=max_seq_len,
        )
    def forward(
        self,
        x: Tensor,
        token_positions:Tensor| None=None,
    )->Tensor:
        *leading_dims, sequence_length,_=x.shape

        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)

        q=q.reshape(*leading_dims,sequence_length,self.num_heads, self.head_dim)
        k=k.reshape(*leading_dims,sequence_length,self.num_heads, self.head_dim)
        v=v.reshape(*leading_dims,sequence_length,self.num_heads, self.head_dim)

        q=q.transpose(-3,-2)
        k=k.transpose(-3,-2)
        v=v.transpose(-3,-2)

        if token_positions is None:
            token_positions = torch.arange(
                sequence_length,
                device=x.device,
            )   

        if token_positions.ndim == 1:
            token_positions = token_positions.unsqueeze(0)

        q=self.rope(q,token_positions)
        k=self.rope(k,token_positions)
        causal_mask=torch.tril(torch.ones(sequence_length,sequence_length,dtype=torch.bool,device=x.device))
        attended=scaled_dot_product_attention(q=q,k=k,v=v,mask=causal_mask)
        attended=attended.transpose(-3,-2)
        attended=attended.reshape(*leading_dims,sequence_length, self.d_model)
        return self.o_proj(attended)

class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.ln1 = RMSNorm(d_model=d_model, eps=eps)

        self.attn = MultiHeadSelfAttentionWithRoPE(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
        )

        self.ln2 = RMSNorm(d_model=d_model, eps=eps)

        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
        )

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )

        x = x + self.ffn(
            self.ln2(x)
        )

        return x

class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        self.token_embeddings = Embedding(
            vocab_size=vocab_size,
            d_model=d_model,
        )

        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
        )

        self.lm_head = Linear(
            d_in=d_model,
            d_out=vocab_size,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]

        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"context length {self.context_length}"
            )

        token_positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits