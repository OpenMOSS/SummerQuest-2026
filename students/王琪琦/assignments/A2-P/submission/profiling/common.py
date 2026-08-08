from __future__ import annotations

import json
import math
import platform
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as F

from cs336_basics.model import TransformerLM


MODELS = {
    "small": dict(vocab_size=10000, d_model=768, num_layers=12, num_heads=12, d_ff=3072),
    "medium": dict(vocab_size=10000, d_model=1024, num_layers=24, num_heads=16, d_ff=4096),
    "large": dict(vocab_size=10000, d_model=1280, num_layers=36, num_heads=20, d_ff=5120),
    "xl": dict(vocab_size=10000, d_model=2560, num_layers=32, num_heads=32, d_ff=10240),
}


def build_model(size: str, context: int, device: torch.device) -> torch.nn.Module:
    return TransformerLM(
        context_length=context,
        rope_theta=10_000.0,
        device=device,
        **MODELS[size],
    )


def annotate_attention(model: torch.nn.Module) -> None:
    """Add real score, softmax, and value ranges to the A1 attention modules."""
    def annotated_forward(self, inputs, token_positions=None):
        sequence_length = inputs.shape[-2]
        queries = self._split_heads(self.q_proj(inputs))
        keys = self._split_heads(self.k_proj(inputs))
        values = self._split_heads(self.v_proj(inputs))
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=inputs.device)
            positions = token_positions.unsqueeze(-2)
            queries, keys = self.rope(queries, positions), self.rope(keys, positions)
        mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=inputs.device).tril()
        with torch.profiler.record_function("attention/scores"):
            scores = queries @ keys.transpose(-1, -2)
            scores = scores / math.sqrt(queries.shape[-1])
            scores = scores.masked_fill(~mask, -torch.inf)
        with torch.profiler.record_function("attention/softmax"):
            row_max = scores.max(dim=-1, keepdim=True).values
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
            exponentials = torch.where(mask, torch.exp(scores - row_max), torch.zeros_like(scores))
            denominator = exponentials.sum(dim=-1, keepdim=True)
            weights = torch.where(denominator > 0, exponentials / denominator.clamp_min(torch.finfo(scores.dtype).tiny), torch.zeros_like(exponentials))
        with torch.profiler.record_function("attention/value"):
            attended = weights @ values
        return self.output_proj(self._merge_heads(attended))

    for module in model.modules():
        if module.__class__.__name__ == "MultiHeadSelfAttention":
            module.forward = MethodType(annotated_forward, module)


def amp(device: torch.device, dtype: str):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dtype == "bf16" else nullcontext()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def step(model, optimizer, tokens, targets, mode: str, dtype: str, device: torch.device):
    if mode == "train_step":
        with torch.profiler.record_function("optimizer/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
    elif mode == "forward_backward":
        model.zero_grad(set_to_none=True)
    grad = torch.no_grad() if mode == "forward" else torch.enable_grad()
    with grad, amp(device, dtype), torch.profiler.record_function("forward"):
        logits = model(tokens)
    loss = None
    if mode != "forward":
        with torch.profiler.record_function("loss"):
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        with torch.profiler.record_function("backward"):
            loss.backward()
    if mode == "train_step":
        with torch.profiler.record_function("optimizer"):
            optimizer.step()
    return loss, logits


def environment(device: torch.device) -> dict:
    props = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(), "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": props.name,
        "gpu_memory_mib": round(props.total_memory / 2**20, 1),
    }


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
