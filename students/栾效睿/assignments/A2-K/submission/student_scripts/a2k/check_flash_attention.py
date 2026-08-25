from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch

from cs336_systems.a2k.attention import FlashAttentionPyTorch, FlashAttentionTriton
from student_scripts.a2k.flash_common import attention_reference, error_stats, make_attention_inputs
from student_scripts.a2k.utils import configure_cuda, load_json, write_json


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "correctness.json"
METADATA_PATH = ROOT / "results" / "run_metadata.json"
SEEDS = (336, 337, 338)
HEAD_DIMS = (32, 64, 128)
SEQUENCE_LENGTH = 256
TOLERANCE = {"rtol": 1e-2, "atol": 1e-2}
IMPLEMENTATIONS = {"pytorch_tiled": FlashAttentionPyTorch, "triton_flashattention": FlashAttentionTriton}


def saved_lse(output: torch.Tensor) -> torch.Tensor:
    return next(t for t in output.grad_fn.saved_tensors if t.ndim == 2 and t.shape == output.shape[:2])


def compare(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    return {**error_stats(actual, expected), "pass": bool(torch.allclose(actual, expected, **TOLERANCE))}


def evaluate(name: str, function: type[torch.autograd.Function], seed: int, head_dim: int, is_causal: bool) -> dict[str, Any]:
    record = {"implementation": name, "seed": seed, "batch_size": 1, "sequence_length": SEQUENCE_LENGTH, "head_dim": head_dim, "dtype": "float32", "is_causal": is_causal, "tolerance": TOLERANCE}
    try:
        q, k, v, do = make_attention_inputs(seed, SEQUENCE_LENGTH, head_dim, torch.float32)
        q_ref, k_ref, v_ref = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
        expected_output, expected_lse = attention_reference(q_ref, k_ref, v_ref, is_causal)
        expected_output.backward(do)
        output = function.apply(q, k, v, is_causal)
        lse = saved_lse(output).detach()
        output.backward(do)
        checks = {
            "output": compare(output.detach(), expected_output.detach()),
            "logsumexp": compare(lse, expected_lse.detach()),
            "dQ": compare(q.grad, q_ref.grad),
            "dK": compare(k.grad, k_ref.grad),
            "dV": compare(v.grad, v_ref.grad),
        }
        return {**record, "checks": checks, "status": "pass" if all(check["pass"] for check in checks.values()) else "fail"}
    except Exception as error:
        return {**record, "checks": {}, "status": f"error:{type(error).__name__}", "error": str(error).splitlines()[0][:300]}


def main() -> None:
    metadata = configure_cuda()
    torch.cuda.init()
    records = [evaluate(name, function, seed, head_dim, is_causal) for name, function in IMPLEMENTATIONS.items() for seed in SEEDS for head_dim in HEAD_DIMS for is_causal in (False, True)]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    write_json(OUTPUT, {"command": "python -m student_scripts.a2k.check_flash_attention", "commit": commit, "metadata": metadata, "records": records})
    run_metadata = load_json(METADATA_PATH)
    run_metadata.setdefault("commands", {})["flash_correctness"] = "python -m student_scripts.a2k.check_flash_attention"
    run_metadata["flash_correctness"] = {"commit": commit, "seeds": SEEDS, "batch_size": 1, "sequence_length": SEQUENCE_LENGTH, "head_dims": HEAD_DIMS, "dtype": "float32", "causal": (False, True), "tolerance": TOLERANCE}
    run_metadata.update(metadata)
    write_json(METADATA_PATH, run_metadata)
    print(f"wrote {OUTPUT}; {sum(record['status'] == 'pass' for record in records)}/{len(records)} passed")


if __name__ == "__main__":
    main()
