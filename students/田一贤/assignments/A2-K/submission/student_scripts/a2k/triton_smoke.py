from __future__ import annotations

import torch

from cs336_systems.a2k.attention import FlashAttentionTriton


def _dense(q, k, v, causal):
    scores = q.float() @ k.float().transpose(-1, -2) / (q.shape[-1] ** 0.5)
    if causal:
        qi = torch.arange(q.shape[-2], device=q.device)[:, None]
        kj = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(qi[None] < kj[None], -1.0e9)
    return (torch.softmax(scores, dim=-1) @ v.float()).to(q.dtype), torch.logsumexp(
        scores, dim=-1
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("status=not_run_no_cuda")
        return
    torch.manual_seed(20260811)
    rows = []
    for causal in (False, True):
        q = torch.randn(
            2, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        do = torch.randn_like(q)
        out = FlashAttentionTriton.apply(q, k, v, causal)
        saved_lse = [
            tensor
            for tensor in out.grad_fn.saved_tensors
            if tensor.shape == (q.shape[0], q.shape[1])
        ]
        ref, ref_lse = _dense(q, k, v, causal)
        out_error = float((out - ref).abs().max())
        lse_error = float((saved_lse[0] - ref_lse).abs().max())
        out.backward(do)
        q2, k2, v2 = (tensor.detach().requires_grad_() for tensor in (q, k, v))
        ref2, _ = _dense(q2, k2, v2, causal)
        ref2.backward(do)
        grad_error = max(
            float((q.grad - q2.grad).abs().max()),
            float((k.grad - k2.grad).abs().max()),
            float((v.grad - v2.grad).abs().max()),
        )
        rows.append(
            {
                "causal": causal,
                "output_max_abs_error": out_error,
                "lse_max_abs_error": lse_error,
                "gradient_max_abs_error": grad_error,
            }
        )
    print({"status": "pass", "rows": rows})


if __name__ == "__main__":
    main()
