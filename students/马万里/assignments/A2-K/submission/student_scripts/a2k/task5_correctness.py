from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from cs336_systems.a2k.flash_attention import FlashAttentionPytorchFunction, FlashAttentionTritonFunction

REF_MAGNITUDE_FLOOR = 1e-6


def reference(q, k, v, is_causal):
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if is_causal:
        q_index = torch.arange(q.shape[1], device=q.device)[:, None]
        k_index = torch.arange(k.shape[1], device=q.device)[None, :]
        scores = scores.masked_fill(q_index < k_index, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype), lse


def compare(actual, expected, atol, rtol):
    """逐元素混合判据：|a - b| <= atol + rtol * |b|。

    返回 (max_abs_error, max_rel_error, exceed_count, nan_mismatch, total_elements)。
    - max_abs_error：最大绝对误差（严格指标，与容差同量纲）。
    - max_rel_error：只在 |b| >= REF_MAGNITUDE_FLOOR 的元素上取最大值；否则
      分母接近 0 会让相对误差虚高几个数量级，失去诊断意义。
    - exceed_count：超过混合阈值的元素个数，为 0 才算 pass。这是判定依据。
    - nan_mismatch：两侧 NaN 位置不一致，单独报出来（NaN != NaN，无法用差值衡量）。
    """
    a = actual.float()
    b = expected.float()
    if a.shape != b.shape:
        return float("inf"), float("inf"), int(a.numel()), True, int(a.numel())
    total = int(a.numel())
    nan_mismatch = not torch.equal(torch.isnan(a), torch.isnan(b))
    finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs()
    allowed = atol + rtol * b.abs()
    exceed = int((diff[finite] > allowed[finite]).sum().item())
    if not finite.any():
        return float("inf"), float("inf"), exceed, nan_mismatch, total
    max_abs = float(diff[finite].max().item())
    mask = finite & (b.abs() >= REF_MAGNITUDE_FLOOR)
    max_rel = float((diff[mask] / b[mask].abs()).max().item()) if mask.any() else 0.0
    return max_abs, max_rel, exceed, nan_mismatch, total


def row(base, **kwargs):
    return {"implementation": "", "seed": None, "sequence_length": 0, "head_dim": 0,
            "dtype": "", "is_causal": False, "quantity": "", "max_abs_error": None,
            "max_rel_error": None, "exceed_count": None, "total_elements": None,
            "nan_mismatch": None, "atol": None, "rtol": None, "status": "", "error": "",
            **base, **kwargs}


def run_case(function, q, k, v, is_causal, atol, rtol, seed):
    q_impl, k_impl, v_impl = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    output = function.apply(q_impl, k_impl, v_impl, is_causal)
    output_ref, lse_ref = reference(q_impl.detach(), k_impl.detach(), v_impl.detach(), is_causal)

    candidates = [t for t in output.grad_fn.saved_tensors if t.shape == (q.shape[0], q.shape[1])]
    if len(candidates) != 1:
        raise RuntimeError(f"期望恰好 1 个形状为 {(q.shape[0], q.shape[1])} 的保存张量，实际 {len(candidates)} 个")
    saved_lse = candidates[0]

    grad_output = torch.randn_like(output, generator=torch.Generator(device=output.device).manual_seed(seed))
    output.backward(grad_output)

    q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    reference(q_ref, k_ref, v_ref, is_causal)[0].backward(grad_output)

    values = (("forward_output", output, output_ref), ("lse", saved_lse, lse_ref),
              ("dQ", q_impl.grad, q_ref.grad), ("dK", k_impl.grad, k_ref.grad),
              ("dV", v_impl.grad, v_ref.grad))
    results = []
    for name, actual, expected in values:
        stats = compare(actual, expected, atol, rtol)
        results.append((name, *stats))
    return results, grad_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, nargs="+", default=[128],
                        help="一个或多个序列长度；结果合并写入同一个 JSON")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 23_552 * 2**20 / total_bytes), device=0)
    records = []
    for length in args.length:
      for seed in (0, 1, 2):
        for head_dim in (32, 64, 128):
            for dtype in (torch.float32, torch.bfloat16):
                for is_causal in (False, True):
                    for name, function in (("pytorch_tiled", FlashAttentionPytorchFunction),
                                           ("triton", FlashAttentionTritonFunction)):
                        base = {"implementation": name, "seed": seed, "sequence_length": length,
                                "head_dim": head_dim, "dtype": str(dtype).replace("torch.", ""),
                                "is_causal": is_causal}
                        if name == "triton" and args.device != "cuda":
                            records.append(row(base, quantity="all", status="skip", error="CUDA unavailable"))
                            continue
                        try:
                            torch.manual_seed(seed)
                            q = torch.randn((2, length, head_dim), device=args.device, dtype=dtype,
                                            requires_grad=True)
                            k = torch.randn_like(q, requires_grad=True)
                            v = torch.randn_like(q, requires_grad=True)
                            atol, rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-4)
                            case_results, _ = run_case(function, q, k, v, is_causal, atol, rtol, seed)
                            for quantity, max_abs, max_rel, exceed, nan_mismatch, total in case_results:
                                passed = exceed == 0 and not nan_mismatch
                                records.append(row(
                                    base, quantity=quantity,
                                    max_abs_error=max_abs, max_rel_error=max_rel,
                                    exceed_count=exceed, total_elements=total,
                                    nan_mismatch=nan_mismatch, atol=atol, rtol=rtol,
                                    status="pass" if passed else "fail",
                                    error="" if passed else (
                                        "NaN 位置不一致" if nan_mismatch
                                        else f"{exceed}/{total} 个元素超过 atol+rtol*|ref|")))
                        except torch.cuda.OutOfMemoryError as exc:
                            records.append(row(base, quantity="all", status="oom", error=str(exc)[:300]))
                        except Exception as exc:
                            records.append(row(base, quantity="all", status="fail",
                                               error=f"{type(exc).__name__}: {exc}"[:500]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
