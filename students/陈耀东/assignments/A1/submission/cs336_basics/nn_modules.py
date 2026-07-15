"""CS336 Assignment 1 的小型神经网络基础模块。

这个文件是学习脚手架：先把模块边界、参数命名和 shape 约定搭好，
核心数学实现留给你自己填写。

目标测试：
    uv run pytest tests/test_model.py::test_linear -q
    uv run pytest tests/test_model.py::test_embedding -q
    uv run pytest tests/test_model.py::test_silu_matches_pytorch -q
    uv run pytest tests/test_model.py::test_rmsnorm -q
    uv run pytest tests/test_model.py::test_swiglu -q
    uv run pytest tests/test_model.py::test_rope -q
    uv run pytest tests/test_model.py::test_scaled_dot_product_attention tests/test_model.py::test_4d_scaled_dot_product_attention -q
"""

from __future__ import annotations
from typing import Literal

from einops import rearrange

import torch
import math
from torch import Tensor, nn


class Linear(nn.Module):
    """不带 bias 的线性层。

    Shape 约定：
        x:      (..., in_features)
        weight: (out_features, in_features)
        out:    (..., out_features)

    注意：
        - 参数命名为 ``weight``，这样 adapter 可以加载测试给定的权重。
        - 不要使用 ``nn.Linear`` 或 ``torch.nn.functional.linear``。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        # 实现说明：按作业要求使用截断正态分布初始化。
        # 提示：标准差 std 与 in_features 和 out_features 有关。
        std = 1.0 / math.sqrt(in_features)
        nn.init.trunc_normal_(
        self.weight,
        mean=0.0,
        std=std,
        a=-2.0 * std,
        b=2.0 * std,
    )

    def forward(self, x: Tensor) -> Tensor:
        # 实现说明：在线性层输入 x 的最后一维上做线性变换。
        # 期望输出 shape：(..., out_features)
        return x @ self.weight.T


class Embedding(nn.Module):
    """Token embedding 查表层。

    Shape 约定：
        token_ids: (...) 整数张量
        weight:    (num_embeddings, embedding_dim)
        out:       (..., embedding_dim)

    注意：
        - 参数命名为 ``weight``，这样 adapter 可以加载测试给定的权重。
        - 不要使用 ``nn.Embedding`` 或 ``torch.nn.functional.embedding``。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        # 实现说明：按作业要求使用截断正态分布初始化。
        std = 1.0 / math.sqrt(embedding_dim)
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        # 实现说明：根据 token_ids 取出 self.weight 中对应的行。
        # 期望输出 shape：token_ids.shape + (embedding_dim
        return self.weight[token_ids]


def silu(x: Tensor) -> Tensor:
    """SiLU 激活函数脚手架。

    Shape 约定：
        x:   (...)
        out: (...)

    注意：
        - 直接按数学定义实现。
        - 测试会和 torch.nn.functional.silu 的输出比较。
    """

    # 实现说明：逐元素应用 SiLU。
    return x * torch.sigmoid(x)



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization，简称 RMSNorm。

    Shape 约定：
        x:      (..., d_model)
        weight: (d_model,)
        out:    (..., d_model)

    注意：
        - 只在最后一维上做归一化。
        - 参数命名为 ``weight``，这样 adapter 可以加载测试给定的权重。
        - 为了数值稳定，可以考虑先用 float32 计算归一化，再转回原 dtype。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x / rms * self.weight
        return out.to(in_dtype)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络脚手架。

    Shape 约定：
        x:         (..., d_model)
        w1.weight: (d_ff, d_model)
        w2.weight: (d_model, d_ff)
        w3.weight: (d_ff, d_model)
        out:       (..., d_model)

    数学目标：
        out = w2(silu(w1(x)) * w3(x))

    注意：
        - 这里可以复用前面已经写好的 Linear 和 silu。
        - 三个子层命名为 ``w1``、``w2``、``w3``，方便 adapter 复制测试权重。
        - 不要使用 ``torch.nn.functional.silu``；本作业希望使用你自己的 ``silu``。
        - 这个模块只加工每个 token 自己的最后一维特征，不混合 sequence 维。
        - 输出必须回到 ``(..., d_model)``，后面才能接 residual connection。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 实现说明：创建三个 Linear 子层，并保存为 self.w1、self.w2、self.w3。
        #   self.w1: d_model -> d_ff，产生候选特征分支
        #   self.w3: d_model -> d_ff，产生门控/值分支
        #   self.w2: d_ff -> d_model，把中间维度投回残差流维度
        # 注意：
        #   - Linear(in_features, out_features) 的权重 shape 是 (out_features, in_features)。
        #   - 测试会直接复制 w1.weight / w2.weight / w3.weight。
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)



    def forward(self, x: Tensor) -> Tensor:
        # 实现说明：按公式实现 SwiGLU。
        # 步骤提示：
        #   1. 用 w1 得到候选特征，shape 从 (..., d_model) 变成 (..., d_ff)。
        #   2. 对候选特征应用你前面实现的 silu。
        #   3. 用 w3 得到同样 shape 的门控/值分支。
        #   4. 两个分支逐元素相乘，shape 仍是 (..., d_ff)。
        #   5. 用 w2 投回 (..., d_model)。
        # 期望输出 shape：(..., d_model)
        return self.w2 (silu(self.w1(x))*self.w3(x))


class SiLUFeedForward(nn.Module):
    """不带门控的 SiLU 前馈网络，用于 SwiGLU 消融实验。

    数学形式为 ``w2(silu(w1(x)))``。当隐藏维度取 ``4 * d_model`` 时，
    两个权重矩阵的参数量与隐藏维度约为 ``8/3 * d_model`` 的三矩阵
    SwiGLU 近似相同。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE 旋转位置编码脚手架。

    Shape 约定：
        x:               (..., sequence_length, d_k)
        token_positions: (..., sequence_length)
        out:             (..., sequence_length, d_k)

    数学目标：
        对最后一维按相邻 pair 旋转：
        (x_0, x_1), (x_2, x_3), ...

    注意：
        - d_k 必须是偶数。
        - 旋转角度由 token_positions 和 theta 决定。
        - token_positions 是显式传入的，不要默认永远是 0..T-1。
        - 实现时要照顾任意前置维度 ``...``。
        - RoPE 通常作用在 query/key 上，让 attention score 感知位置信息。
        - RoPE 不改变张量 shape，只改变最后一维向量的方向。
    """

    def __init__(
        self,
        d_k: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # 实现说明：预计算频率、cos 和 sin。
        # 推荐路线：
        #   1. 构造 pair 索引 i = 0, 1, ..., d_k//2 - 1。
        #   2. 按 omega_i = theta ** (-2*i/d_k) 得到每个 pair 的频率。
        #   3. 构造位置 p = 0, 1, ..., max_seq_len - 1。
        #   4. angle[p, i] = p * omega_i，shape 为 (max_seq_len, d_k//2)。
        #   5. 可以把 cos(angle)、sin(angle) 用 register_buffer 保存。
        # 注意：
        #   - buffer 不是可学习参数，但会跟随模块移动 device。
        #   - 如果不预计算，也可以在 forward 里根据 token_positions 即时计算。
        i = torch.arange(self.d_k//2,device=device,dtype=torch.float32)
        omega_i = self.theta ** (-2.0 * i/self.d_k)
        p = torch.arange(self.max_seq_len,device=device,dtype=torch.int)
        angle = torch.outer(p,omega_i)
        self.register_buffer("cos_cached", torch.cos(angle), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angle), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # 实现说明：实现 RoPE 旋转。
        # 步骤提示：
        #   1. 拆出偶数维 x[..., 0::2] 和奇数维 x[..., 1::2]。
        #   2. 根据 token_positions 取出或计算 cos/sin。
        #      cos/sin 的目标 shape 通常是 (..., sequence_length, d_k//2)。
        #   3. 应用二维旋转公式：
        #      even_out = even * cos - odd * sin
        #      odd_out  = even * sin + odd * cos
        #   4. 把 even_out 和 odd_out 交错放回最后一维。
        #   5. 返回 shape 与输入完全相同的张量。
        # 常见坑：
        #   - 不要把前半维和后半维配对；本作业按相邻维度配对。
        #   - 不要忽略 token_positions。
        #   - 注意 cos/sin 的 device 和 dtype。
        # 期望输出 shape：(..., sequence_length, d_k)
        even = x[...,0::2]
        odd = x[...,1::2]
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        even_out = even * cos - odd * sin
        odd_out = even * sin + odd * cos

        out = torch.stack([even_out, odd_out], dim=-1)
        out = out.flatten(-2)
        return out


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scaled dot-product attention 脚手架。

    Shape 约定：
        Q:    (..., queries, d_k)
        K:    (..., keys, d_k)
        V:    (..., keys, d_v)
        mask: (..., queries, keys)，或 None
        out:  (..., queries, d_v)

    数学目标：
        softmax(Q K^T / sqrt(d_k)) V

    mask 语义：
        - True 表示允许 attend。
        - False 表示需要屏蔽，对应 score 应该变成 -inf 或很小的负数。

    注意：
        - softmax 必须沿最后一维 keys 做。
        - 需要支持任意前置维度，包括测试里的 4D 输入。
        - 这个函数只实现 attention 核心，不负责 Q/K/V 投影，也不负责多头拆分。
    """

    # 实现说明：计算 attention score。
    # 推荐路线：
    #   1. 从 Q.shape[-1] 取得 d_k。
    #   2. 计算 scores = Q @ K.transpose(-2, -1) / sqrt(d_k)。
    #      scores 的 shape 应该是 (..., queries, keys)。
    #   3. 如果 mask 不是 None，把 mask 为 False 的位置改成 -inf 或 dtype 极小值。
    #      本作业约定：True 表示允许 attend，False 表示屏蔽。
    #   4. 对 scores 的最后一维做 softmax，得到 attention weights。
    #   5. 返回 weights @ V，shape 应该是 (..., queries, d_v)。
    # 常见坑：
    #   - softmax 维度必须是 key 维，也就是 dim=-1。
    #   - 不要写死 batch/head 维度数量，用最后两维做矩阵乘即可。
    #   - mask 语义不要反。
    d_k = Q.shape[-1]
    score = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(score, dim=-1)
    return weights @ V


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention 脚手架。

    Shape 约定：
        x:              (..., sequence_length, d_model)
        q_proj.weight:  (d_model, d_model)
        k_proj.weight:  (d_model, d_model)
        v_proj.weight:  (d_model, d_model)
        output_proj.weight: (d_model, d_model)
        out:            (..., sequence_length, d_model)

    机制：
        1. 用 q/k/v 三个投影把输入变成 Q/K/V。
        2. 把最后一维 d_model 拆成 num_heads 个 head。
        3. 每个 head 独立做 scaled dot-product attention。
        4. 把所有 head 拼回 d_model。
        5. 用 output_proj 混合不同 head 的输出。

    注意：
        - 这里是不带 RoPE 的版本。
        - 需要使用 causal mask，防止当前位置看未来 token。
        - 四个线性层都不要带 bias；可以复用前面实现的 Linear。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 实现说明：确认 d_model 能被 num_heads 整除。
        # 实现说明：创建 self.q_proj、self.k_proj、self.v_proj、self.output_proj。
        # 提示：每个投影都是 d_model -> d_model，权重 shape 为 (d_model, d_model)。
        assert d_model % num_heads == 0, \
            f"d_model={d_model} 必须能被 num_heads={num_heads} 整除"
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x: Tensor) -> Tensor:
        # 实现说明：把 (..., sequence_length, d_model)
        #       变成 (..., num_heads, sequence_length, d_k)。
        # 提示：可以使用 rearrange，或者 reshape + movedim / transpose。
        return rearrange(x,"... seq (h d) -> ... h seq d",h=self.num_heads)


    def _merge_heads(self, x: Tensor) -> Tensor:
        # 实现说明：把 (..., num_heads, sequence_length, d_k)
        #       变回 (..., sequence_length, d_model)。
        return rearrange(x,"... h seq d -> ... seq (h d)")

    def _causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        # 实现说明：构造 shape 为 (sequence_length, sequence_length) 的布尔 mask。
        # 本作业约定 True 表示允许 attend，False 表示屏蔽。
        # 对 causal mask 来说，key 位置 j <= query 位置 i 时允许 attend。
        return torch.tril(
            torch.ones(sequence_length, sequence_length, device=device,dtype=torch.bool)
        )

    def forward(self, x: Tensor) -> Tensor:
        # 实现说明：实现不带 RoPE 的 multi-head self-attention。
        # 推荐路线：
        #   1. q = self.q_proj(x)，k = self.k_proj(x)，v = self.v_proj(x)。
        #   2. 拆成多头：(..., num_heads, sequence_length, d_k)。
        #   3. 构造 causal mask，并让它能广播到所有前置维度和 head。
        #   4. 调用 scaled_dot_product_attention(q, k, v, mask)。
        #   5. 合并 head。
        #   6. 做 output_proj。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        mask =self._causal_mask(x.shape[-2], x.device)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = self._merge_heads(out)
        return self.output_proj(out)


class MultiHeadSelfAttentionWithRoPE(MultiHeadSelfAttention):
    """带 RoPE 的 multi-head self-attention 脚手架。

    与普通 MHA 的区别：
        - 拆 head 后，对 Q 和 K 应用 RoPE。
        - V 不应用 RoPE，因为 V 是被读取的内容，不负责计算匹配分数。

    Shape 约定：
        x:               (..., sequence_length, d_model)
        token_positions: (..., sequence_length)，或 None
        out:             (..., sequence_length, d_model)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads, device=device, dtype=dtype)
        self.max_seq_len = max_seq_len
        self.theta = theta

        # 实现说明：创建 RoPE 模块。
        # 注意：RoPE 的 d_k 应该是每个 head 的维度 self.d_k，而不是 d_model。
        self.rope = RotaryPositionalEmbedding(theta=theta,d_k=self.d_k,device=device,max_seq_len=self.max_seq_len)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        # 实现说明：实现带 RoPE 的 multi-head self-attention。
        # 推荐路线：
        #   1. 先做 q/k/v 投影并拆 head。
        #   2. 如果 token_positions 为 None，则根据 sequence_length 构造 0..T-1。
        #   3. 对 q 和 k 应用 self.rope。
        #   4. v 不旋转。
        #   5. 后续 attention、合并 head、output_proj 与普通 MHA 一样。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if token_positions is None:
            token_positions = torch.arange(x.shape[-2], device=x.device)
        q = self.rope(q,token_positions)
        k = self.rope(k,token_positions)
        mask = self._causal_mask(x.shape[-2], x.device)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = self._merge_heads(out)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """支持课程四项架构消融的 Transformer block。

    Shape 约定：
        x:   (batch, sequence_length, d_model)
        out: (batch, sequence_length, d_model)

    结构：
        y = x + attn(ln1(x))
        z = y + ffn(ln2(y))

    注意：
        - 默认使用 pre-norm、RoPE 和 SwiGLU，与公共测试行为一致。
        - normalization 可选 pre、post 或 none。
        - positional_encoding 可选 rope 或 none。
        - ffn_type 可选 swiglu 或 silu。
        - attention 子层和 ffn 子层的输出都必须是 d_model，才能做 residual add。
        - 权重命名要和 adapter 里的 state dict 对齐。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        normalization: Literal["pre", "post", "none"] = "pre",
        positional_encoding: Literal["rope", "none"] = "rope",
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
    ) -> None:
        super().__init__()

        if normalization not in {"pre", "post", "none"}:
            raise ValueError(f"不支持的 normalization：{normalization}")
        if positional_encoding not in {"rope", "none"}:
            raise ValueError(f"不支持的 positional_encoding：{positional_encoding}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"不支持的 ffn_type：{ffn_type}")

        self.normalization = normalization
        self.positional_encoding = positional_encoding
        self.ffn_type = ffn_type

        if normalization == "none":
            self.ln1 = None
            self.ln2 = None
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        if positional_encoding == "rope":
            self.attn = MultiHeadSelfAttentionWithRoPE(
                d_model=d_model,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                theta=theta,
                device=device,
                dtype=dtype,
            )
        else:
            self.attn = MultiHeadSelfAttention(
                d_model=d_model,
                num_heads=num_heads,
                device=device,
                dtype=dtype,
            )

        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def _run_attention(self, x: Tensor, token_positions: Tensor | None) -> Tensor:
        """根据位置编码模式调用对应 attention，统一 block 的前向路径。"""
        if isinstance(self.attn, MultiHeadSelfAttentionWithRoPE):
            return self.attn(x, token_positions)
        return self.attn(x)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.normalization == "none":
            y = x + self._run_attention(x, token_positions)
            return y + self.ffn(y)

        assert self.ln1 is not None and self.ln2 is not None
        if self.normalization == "post":
            y = self.ln1(x + self._run_attention(x, token_positions))
            return self.ln2(y + self.ffn(y))

        y = x + self._run_attention(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model 脚手架。

    Shape 约定：
        token_ids: (batch, sequence_length)
        logits:    (batch, sequence_length, vocab_size)

    总体结构：
        token_ids
        -> token_embeddings
        -> TransformerBlock × num_layers
        -> ln_final
        -> lm_head
        -> logits

    注意：
        - forward 返回 logits，不要做 softmax。
        - sequence_length 使用当前输入长度，不要固定写成 context_length。
        - 每一层 block 都需要同一组 token_positions，供 RoPE 使用。
        - lm_head 是 d_model -> vocab_size 的无 bias 线性层。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        normalization: Literal["pre", "post", "none"] = "pre",
        positional_encoding: Literal["rope", "none"] = "rope",
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.normalization = normalization
        self.positional_encoding = positional_encoding
        self.ffn_type = ffn_type

        # 实现说明：创建 token embedding。
        # 提示：权重 shape 应为 (vocab_size, d_model)。
        self.token_embeddings = Embedding(vocab_size,d_model,device=device,dtype=dtype)

        # 实现说明：创建 num_layers 个 TransformerBlock，并放入 nn.ModuleList。
        # 每个 block 使用相同的 d_model / num_heads / d_ff / context_length / rope_theta。
        # num_layers 个 TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
                normalization=normalization,
                positional_encoding=positional_encoding,
                ffn_type=ffn_type,
            )
            for _ in range(num_layers)
        ])

        # 实现说明：创建 final RMSNorm。
        self.ln_final = None if normalization == "none" else RMSNorm(d_model=d_model,device=device,dtype=dtype)

        # 实现说明：创建 lm_head。
        # 提示：Linear(d_model, vocab_size) 的 weight shape 正好是 (vocab_size, d_model)。
        self.lm_head = Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        # 实现说明：实现 TransformerLM 前向传播。
        # 推荐路线：
        #   1. sequence_length = token_ids.shape[-1]。
        #   2. 用 token_embeddings 把 token id 变成 hidden states。
        #   3. 构造 token_positions = 0..sequence_length-1，device 与 token_ids 一致。
        #      常见 shape 可用 (1, sequence_length)，方便广播到 batch。
        #   4. 依次通过每个 TransformerBlock。
        #   5. 通过 ln_final。
        #   6. 通过 lm_head 得到 logits。
        #   7. 直接返回 logits，不要 softmax。
        sequence_length = token_ids.shape[-1]
        x = self.token_embeddings(token_ids)
        token_positions = torch.arange(sequence_length,device=token_ids.device).unsqueeze(0)
        for layer in self.layers:
            x = layer(x, token_positions)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return self.lm_head(x)
