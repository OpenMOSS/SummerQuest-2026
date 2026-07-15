"""GPT-2 s/m/l/XL param, memory, and forward-FLOPs analysis.

Formulas
--------
Per Transformer block (pre-norm, no bias in QKV/O like the tests):
- Attention linears (Q, K, V, O): 4 * d_model^2 params, 4 * 2 * d_model^2 = 8 * d_model^2 MACs per token
  (forward FLOPs = 2 * MACs).
- FFN GPT-2-style (up + down, 4*d_model inner): 8 * d_model^2 params, 16 * d_model^2 MACs per token.
- Scaled dot-product attention (score + weighted sum), per token: ~4 * d_model * L FLOPs
  where L is context length (2*d_model*L MACs for scores + 2*d_model*L MACs for values).

Total forward FLOPs per token ≈ (24 * d_model^2 * n_layer + 4 * d_model * L * n_layer) * 2
                              + 2 * d_model * vocab_size (embedding lookup+unembedding)
"""

from __future__ import annotations

CONFIGS = {
    "gpt2-small":  {"d_model": 768,  "n_layer": 12, "n_head": 12, "d_ff": 3072,  "vocab": 50257, "L": 1024},
    "gpt2-medium": {"d_model": 1024, "n_layer": 24, "n_head": 16, "d_ff": 4096,  "vocab": 50257, "L": 1024},
    "gpt2-large":  {"d_model": 1280, "n_layer": 36, "n_head": 20, "d_ff": 5120,  "vocab": 50257, "L": 1024},
    "gpt2-xl":     {"d_model": 1600, "n_layer": 48, "n_head": 25, "d_ff": 6400,  "vocab": 50257, "L": 1024},
}


def analyze(cfg):
    d = cfg["d_model"]; L = cfg["L"]; n = cfg["n_layer"]; V = cfg["vocab"]; d_ff = cfg["d_ff"]
    # Params
    attn_params = 4 * d * d                         # Q, K, V, O
    ffn_params = 2 * d * d_ff                       # up + down (GPT-2 uses two linears)
    block_params = attn_params + ffn_params + 4 * d # + 2 layer norms (γ, β each d)
    embed_params = V * d + L * d                    # token + position embeddings
    lm_head = 0  # tied with token embeddings in GPT-2
    total_params = n * block_params + embed_params + 2 * d  # + final layernorm

    # Forward FLOPs per token (dense linears count 2*d_in*d_out; softmax + activation ignored)
    attn_lin_flops = 2 * (4 * d * d)                # Q, K, V, O linears
    attn_dot_flops = 2 * (2 * d * L)                # scores + attention @ V
    ffn_flops = 2 * (2 * d * d_ff)                  # two linears in FFN
    per_layer = attn_lin_flops + attn_dot_flops + ffn_flops
    lm_head_flops = 2 * d * V
    per_token_flops = n * per_layer + lm_head_flops
    per_seq_flops = per_token_flops * L

    # Memory (params only, fp32/bf16)
    mem_bytes_fp32 = total_params * 4
    mem_bytes_bf16 = total_params * 2
    return {
        "params": total_params,
        "params_M": total_params / 1e6,
        "mem_fp32_GB": mem_bytes_fp32 / 1e9,
        "mem_bf16_GB": mem_bytes_bf16 / 1e9,
        "fwd_flops_per_token": per_token_flops,
        "fwd_flops_per_seq_G": per_seq_flops / 1e9,
    }


if __name__ == "__main__":
    print(f"{'model':12s}  {'params(M)':>10s}  {'fp32(GB)':>9s}  {'bf16(GB)':>9s}  {'fwd/tok(GF)':>12s}  {'fwd/seq(GF)':>12s}")
    for name, cfg in CONFIGS.items():
        r = analyze(cfg)
        print(f"{name:12s}  {r['params_M']:>10.1f}  {r['mem_fp32_GB']:>9.2f}  {r['mem_bf16_GB']:>9.2f}  "
              f"{r['fwd_flops_per_token']/1e9:>12.3f}  {r['fwd_flops_per_seq_G']:>12.1f}")

    print()
    print("AdamW state per parameter: 2 optimizer moments (float32 = 8 bytes) + master params (float32=4 bytes)")
    print("Weight memory + AdamW state (mixed precision, bf16 params + fp32 master + fp32 m,v):")
    for name, cfg in CONFIGS.items():
        r = analyze(cfg)
        weights_bf16 = r["params"] * 2
        master_fp32 = r["params"] * 4
        moments = r["params"] * 8
        grads_bf16 = r["params"] * 2
        total = (weights_bf16 + master_fp32 + moments + grads_bf16) / 1e9
        print(f"  {name:12s} weights+optim ≈ {total:.2f} GB")
