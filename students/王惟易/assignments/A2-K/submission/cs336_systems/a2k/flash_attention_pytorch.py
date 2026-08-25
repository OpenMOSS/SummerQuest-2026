import torch


def _process_query_tile(
        q_tile: torch.Tensor, # (B, Bq, D)
        k: torch.Tensor,      # (B, Nk, D)
        v: torch.Tensor,      # (B, Nk, Dv)
        query_start: int,
        key_tile_size: int,
        is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, query_tile_length, _ = q_tile.shape
    num_keys = k.shape[-2]
    value_dim = v.shape[-1]
    inverse_scale = q_tile.shape[-1] ** -0.5

    row_max = torch.full(
        (batch_size, query_tile_length),
        float("-inf"),
        device=q_tile.device,
        dtype=torch.float32,
    )
    row_sum = torch.zeros_like(row_max)
    output_accumulator = torch.zeros(
        (batch_size, query_tile_length, value_dim),
        device=q_tile.device,
        dtype=torch.float32,
    )

    for key_start in range(0, num_keys, key_tile_size):
        key_end = min(key_start + key_tile_size, num_keys)

        k_tile = k[:, key_start:key_end, :] # (B, Bk, D)
        v_tile = v[:, key_start:key_end, :] # (B, Bk, Dv)

        score_tile = (
            q_tile.float() @ k_tile.float().transpose(-2, -1)
        ) * inverse_scale # (B, Bq, Bk)

        # 因果条件： key_position <= query_position
        if is_causal:
            query_positions = torch.arange(
                query_start,
                query_start + query_tile_length,
                device=q_tile.device
            )
            key_positions = torch.arange(
                key_start,
                key_end,
                device=q_tile.device
            )

            causal_mask = (
                query_positions[:, None] >= key_positions[None, :]
            ) # (Bq, Bk)

            score_tile = score_tile.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

        # 每个 query 在当前 key tile 中的最大值
        tile_max = score_tile.max(dim=-1).values # (B, Bq)
        new_row_max = torch.maximum(row_max, tile_max)

        old_correction = torch.exp(row_max - new_row_max)
        exp_scores = torch.exp(score_tile - new_row_max.unsqueeze(-1))

        # 当前 tile 对分母的贡献。与一维 online softmax 对比，堆叠 Bq 层
        tile_row_sum = exp_scores.sum(dim=-1) # (B, Bq)

        # 当前 tile 对分子的贡献。与一维 online softmax 对比，堆叠 Bq 层，并且把 value 从标量推广成矢量
        tile_output = exp_scores @ v_tile.float() # (B, Bq, Dv)

        new_row_sum = (
            old_correction * row_sum + tile_row_sum
        ) # (B, Bq)

        new_output_accumulator = (
            old_correction.unsqueeze(-1) * output_accumulator + tile_output
        ) # (B, Bq, Dv)

        row_max = new_row_max
        row_sum = new_row_sum
        output_accumulator = new_output_accumulator

    output_tile = (output_accumulator / row_sum.unsqueeze(-1))

    logsumexp_tile = row_max + torch.log(row_sum)

    return output_tile.to(q_tile.dtype), logsumexp_tile


def tiled_flash_attention_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        query_tile_size: int = 64,
        key_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_queries = q.shape[-2]

    output_tiles = []
    logsumexp_tiles = []

    for query_start in range(0, num_queries, query_tile_size):
        query_end = min(query_start + query_tile_size, num_queries)

        q_tile = q[:, query_start:query_end, :] # (B, Bq, D)

        output_tile, logsumexp_tile = _process_query_tile(
            q_tile=q_tile,
            k=k,
            v=v,
            query_start=query_start,
            key_tile_size=key_tile_size,
            is_causal=is_causal
        )

        output_tiles.append(output_tile)
        logsumexp_tiles.append(logsumexp_tile)

    output = torch.cat(output_tiles, dim=1)     # (B, Nq, Dv)
    logsumexp = torch.cat(logsumexp_tiles, dim=1)

    return output, logsumexp


def flash_attention_backward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        grad_output: torch.Tensor,
        logsumexp: torch.Tensor,
        is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inverse_scale = q.shape[-1] ** -0.5

    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    output_float = output.float()
    grad_output_float = grad_output.float()

    scores = inverse_scale * (q_float @ k_float.transpose(-2, -1))

    if is_causal:
        num_queries = q.shape[-2]
        num_keys = k.shape[-2]

        query_positions = torch.arange(
            num_queries,
            device=q.device,
        )
        key_positions = torch.arange(
            num_keys,
            device=q.device
        )

        causal_mask = (
            query_positions[:, None] >= key_positions[None, :]
        )
        scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    probabilities = torch.exp(scores - logsumexp.unsqueeze(-1))

    row_correction = (grad_output_float * output_float).sum(dim=-1, keepdim=True)

    grad_v = probabilities.transpose(-2, -1) @ grad_output_float

    grad_probabilities = grad_output_float @ v_float.transpose(-2, -1)

    grad_scores = probabilities * (grad_probabilities - row_correction)

    grad_q = inverse_scale * (grad_scores @ k_float)

    grad_k = inverse_scale * (grad_scores.transpose(-2, -1) @ q_float)

    return (
        grad_q.to(q.dtype),
        grad_k.to(k.dtype),
        grad_v.to(v.dtype),
    )

compiled_flash_attention_backward = torch.compile(
    flash_attention_backward,
    fullgraph=True,
)

class FlashAttentionPyTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                is_causal: bool = False,
                ) -> torch.Tensor:
        output, logsumexp = tiled_flash_attention_forward(
            q=q,
            k=k,
            v=v,
            is_causal=is_causal,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            output,
            logsumexp,
        )

        ctx.is_causal = is_causal

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        一个函数的输入输出都是矩阵。backward 不构造 Jacobian，而是直接算 Jacobian 转置乘上游梯度，即 VJP (vector-Jacobian product)，这样就可以避免保存巨大的四维 Jacobian。
        形式上，输出 Y 对输入 X 的真正导数为一个四维的线性映射，记它为 dY = Jacobian_f(dX)。注意全微分关系 dJ = <G_Y, dY>_F = <G_Y, Jacobian_f(dX)>_F，而我们已经约定 dJ = <G_X, dX>_F，所以我们有 <G_X, dX>_F = <G_Y, Jacobian_f(dX)>_F，变形得 <G_X, dX>_F = <Jacobian_f^TG_Y, dX>_F，所以 G_X = Jacobian_f^TG_Y。
        我们要求的就是 G_X, G_Y 是已知量（grad_output），看起来直接用 G_X = Jacobian_f^TG_Y 计算即可。但是由于 Jacobian_f 是一个四维张量，直接构造它会占用巨大的内存，所以我们只会隐式计算 Jacobian_f^TG_Y，也就是通过微分和矩阵恒等式得到相同结果。
        一个例子：
        forward: O = P @ V
        backward: dJ = <G_O, dO>_F，dO = dP @ V + P @ dV，所以 dJ = <G_O, dP @ V>_F + <G_O, P @ dV>_F，
        dJ = <G_O @ V^T, dP>_F + <P^T @ G_O, dV>_F。有因为 dJ = <G_P, dP>_F + <G_V, dV>_F，读系数得 G_P = G_O @ V^T, G_V = P^T @ G_O。
        像这样不断往前推，就能得到 G_Q, G_K 和 G_V，即为所求。
        不妨就在这里往下推完。G_V 已经拿到了，很好，接下来还需要 G_Q 和 G_K。不过还是一步一步往前推：P = softmax(S), S = αQK^T
        forward: P = softmax(S)，换一种写法，P[i, j] = exp(S[i, j]) / Σ_r exp(S[i, r]) (对每个 row 独立进行)
        backward: (最复杂，需要展开)
        <G_P, dP> = Σ_i,j <G_P[i,j], dP[i,j]> (此时退化为标量乘法，但暂时仍用 <> 记号) = Σ_i,j <G_P[i,j], d(exp(S[i,j])/Z[i])>
        = Σ_i,j <G_P[i,j], exp(S[i,j])dS[i,j]/Z[i]>
        + Σ_i,j <G_P[i,j], -exp(S[i,j])/Z[i]^2 Σ_r exp(S[i,r])dS[i,r]>
        = Σ_i,j <exp(S[i,j])/Z[i] G_P[i,j], dS[i,j]>
        + Σ_i,j Σ_r <-exp(S[i,j])/Z[i]^2 exp(S[i,r]) G_P[i,j], dS[i,r]>
        Σ_i,j <G_S[i,j], dS[i,j]>
        然后需要把dS[i,j]”收集“起来，然后”分配“给dP[i,j]，最终实现通过P和G_P表示G_S：
        G_S[i,k] = P[i,k] (G_P[i,k] - Σ_j P[i,j]G_P[i,j])
        forward: S = α(Q @ K^T)
        backward: <G_S, dS>_F = <G_S, α(dQ @ K^T)>_F + <G_S, α(Q @ dK^T)_F = <αG_S @ K, dQ>_F + <αG_S^T @ Q, dK> = <G_Q, dQ>_F + <G_K, dK>_F，根据系数得 G_Q = αG_S @ K, G_K = αG_S^T @ Q
        """

        q, k, v, output, logsumexp = ctx.saved_tensors

        grad_q, grad_k, grad_v = compiled_flash_attention_backward(
            q=q,
            k=k,
            v=v,
            output=output,
            grad_output=grad_output,
            logsumexp=logsumexp,
            is_causal=ctx.is_causal,
        )

        return grad_q, grad_k, grad_v, None
