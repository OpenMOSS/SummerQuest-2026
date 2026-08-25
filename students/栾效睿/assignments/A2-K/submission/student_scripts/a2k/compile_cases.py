from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import torch

from cs336_basics.model import BasicsTransformerLM, scaled_dot_product_attention
from cs336_basics.nn_utils import cross_entropy
from student_scripts.a2k.utils import (
    benchmark_cuda,
    benchmark_cuda_step,
    latency_columns,
    measure_cuda_peak,
    sample_quantiles,
    timed_cuda_call,
)


BATCH_SIZE = 1
ATTENTION_CONFIGS = ((512, 64), (2_048, 128), (8_192, 128))
MODEL_CONTEXT = 512
MODEL_WARMUP_STEPS = 5
MODEL_MEASUREMENT_STEPS = 10
MODEL_COMPARE_RTOL = 1e-2
MODEL_COMPARE_ATOL = 1.5e-2
MODEL_CONFIG = {
    "vocab_size": 10_000,
    "d_model": 768,
    "d_ff": 3_072,
    "num_layers": 12,
    "num_heads": 12,
}


def _reset_dynamo() -> None:
    import torch._dynamo
    from torch._dynamo.utils import counters

    torch._dynamo.reset()
    counters.clear()


def dynamo_stats() -> tuple[int | str, int | str]:
    try:
        from torch._dynamo.utils import counters

        return sum(counters["graph_break"].values()), counters["stats"].get("unique_graphs", "")
    except (AttributeError, KeyError, TypeError):
        return "", ""


def assert_model_outputs_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=MODEL_COMPARE_RTOL, atol=MODEL_COMPARE_ATOL)


def _benchmark_model_phase(step: Callable[[], Any]) -> tuple[tuple[float, float, float], float, float]:
    samples, _, peak_allocated, peak_reserved = benchmark_cuda_step(step, MODEL_WARMUP_STEPS, MODEL_MEASUREMENT_STEPS)
    return sample_quantiles(samples), peak_allocated, peak_reserved


def run_attention(row: dict[str, Any]) -> dict[str, Any]:
    sequence_length, head_dim = int(row["sequence_length"]), int(row["head_dim"])
    q, k, v = (torch.randn((BATCH_SIZE, sequence_length, head_dim), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    causal_mask = torch.ones((sequence_length, sequence_length), device="cuda", dtype=torch.bool).tril_()
    output_gradient, inputs = torch.randn_like(q), (q, k, v)
    attention = scaled_dot_product_attention

    if row["implementation"] == "compiled":
        _reset_dynamo()
        attention = torch.compile(attention, backend="inductor", fullgraph=True, dynamic=False)
        cold_output, row["forward_cold_start_ms"] = timed_cuda_call(lambda: attention(q, k, v, causal_mask))
        with torch.no_grad():
            torch.testing.assert_close(cold_output, scaled_dot_product_attention(q, k, v, causal_mask), rtol=1e-2, atol=1e-2)
        del cold_output
        gc.collect()

    def forward() -> torch.Tensor:
        return attention(q, k, v, causal_mask)

    row.update(latency_columns("forward", benchmark_cuda(forward)))
    backward_output = forward()

    def backward(saved_output: torch.Tensor = backward_output) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(saved_output, inputs, output_gradient, retain_graph=True)

    if row["implementation"] == "compiled":
        cold_gradients, row["backward_cold_start_ms"] = timed_cuda_call(backward)
        del cold_gradients
        row["total_cold_start_ms"] = row["forward_cold_start_ms"] + row["backward_cold_start_ms"]
    row.update(latency_columns("backward", benchmark_cuda(backward)))
    del backward, backward_output
    gc.collect()

    def forward_backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(forward(), inputs, output_gradient)

    row.update(latency_columns("forward_backward", benchmark_cuda(forward_backward)))
    row["peak_allocated_mib"], row["peak_reserved_mib"] = measure_cuda_peak(forward_backward)
    return row


def run_model(row: dict[str, Any]) -> dict[str, Any]:
    model = BasicsTransformerLM(context_length=MODEL_CONTEXT, **MODEL_CONFIG).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    parameters = tuple(model.parameters())
    tokens = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, MODEL_CONTEXT), device="cuda")
    targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (BATCH_SIZE, MODEL_CONTEXT), device="cuda")
    measured_model = model

    if row["implementation"] == "compiled":
        _reset_dynamo()
        measured_model = torch.compile(model, backend="inductor", fullgraph=False, dynamic=False)

        def cold_forward() -> torch.Tensor:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return measured_model(tokens)

        cold_output, row["forward_cold_start_ms"] = timed_cuda_call(cold_forward)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            assert_model_outputs_close(cold_output, model(tokens))
        del cold_output
        gc.collect()

    def forward() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return measured_model(tokens)

    forward_stats = _benchmark_model_phase(forward)
    row.update(latency_columns("forward", forward_stats[0]))
    peaks = [forward_stats[1:]]

    def loss() -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return cross_entropy(measured_model(tokens), targets)

    backward_loss = loss()

    def backward(saved_loss: torch.Tensor = backward_loss) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(saved_loss, parameters, retain_graph=True)

    if row["implementation"] == "compiled":
        cold_gradients, row["backward_cold_start_ms"] = timed_cuda_call(backward)
        del cold_gradients
        row["total_cold_start_ms"] = row["forward_cold_start_ms"] + row["backward_cold_start_ms"]
    backward_stats = _benchmark_model_phase(backward)
    row.update(latency_columns("backward", backward_stats[0]))
    peaks.append(backward_stats[1:])
    del backward, backward_loss
    gc.collect()

    def forward_backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(loss(), parameters)

    forward_backward_stats = _benchmark_model_phase(forward_backward)
    row.update(latency_columns("forward_backward", forward_backward_stats[0]))
    peaks.append(forward_backward_stats[1:])

    def training_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        training_loss = loss()
        training_loss.backward()
        optimizer.step()

    training_stats = _benchmark_model_phase(training_step)
    row.update(latency_columns("training_step", training_stats[0]))
    peaks.append(training_stats[1:])
    row["peak_allocated_mib"] = max(peak[0] for peak in peaks)
    row["peak_reserved_mib"] = max(peak[1] for peak in peaks)
    return row
