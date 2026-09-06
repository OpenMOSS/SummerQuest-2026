import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import scaled_dot_product_attention

@nvtx.range("scaled_dot_product_attention")
def annotated_sdpa(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with nvtx.range("attention/scores"):
        scores = torch.einsum("...qd,...kd->...qk", Q, K) / (d_k ** 0.5)
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
    with nvtx.range("attention/softmax"):
        attn_weights = torch.softmax(scores, dim=-1)
    with nvtx.range("attention/value"):
        out = torch.einsum("...qk,...kd->...qd", attn_weights, V)
    return out

def patch_attention():
    """Replace the model's attention function with the annotated version."""
    import cs336_basics.model as model_module
    model_module.scaled_dot_product_attention = annotated_sdpa