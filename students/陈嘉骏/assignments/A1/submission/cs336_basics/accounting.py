from __future__ import annotations

import math
from dataclasses import asdict, dataclass


FLOAT32_BYTES = 4
H100_TF32_FLOPS_PER_SECOND = 495e12


@dataclass(frozen=True)
class TransformerShape:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int


@dataclass(frozen=True)
class TransformerAccounting:
    shape: TransformerShape
    parameter_breakdown: dict[str, int]
    total_parameters: int
    parameter_memory_bytes: int
    forward_flop_breakdown: dict[str, int]
    total_forward_flops: int
    forward_flop_proportions: dict[str, float]


@dataclass(frozen=True)
class AdamWMemoryAccounting:
    batch_size: int
    parameter_bytes: int
    activation_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    total_bytes: int


def nearest_multiple(value: float, multiple: int) -> int:
    if value <= 0 or multiple <= 0:
        raise ValueError("value and multiple must be positive.")
    return multiple * math.floor(value / multiple + 0.5)


def assignment_d_ff(d_model: int) -> int:
    return nearest_multiple((8.0 / 3.0) * d_model, 64)


def transformer_accounting(shape: TransformerShape, batch_size: int = 1) -> TransformerAccounting:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    vocab_size = shape.vocab_size
    sequence_length = shape.context_length
    num_layers = shape.num_layers
    d_model = shape.d_model
    d_ff = shape.d_ff

    parameter_breakdown = {
        "token_embeddings": vocab_size * d_model,
        "attention": num_layers * 4 * d_model * d_model,
        "feed_forward": num_layers * 3 * d_model * d_ff,
        "rmsnorm": (2 * num_layers + 1) * d_model,
        "lm_head": vocab_size * d_model,
    }
    total_parameters = sum(parameter_breakdown.values())

    forward_flop_breakdown = {
        "qkv_projections": batch_size * num_layers * 6 * sequence_length * d_model * d_model,
        "attention_scores": batch_size * num_layers * 2 * sequence_length * sequence_length * d_model,
        "attention_values": batch_size * num_layers * 2 * sequence_length * sequence_length * d_model,
        "attention_output_projection": batch_size * num_layers * 2 * sequence_length * d_model * d_model,
        "feed_forward": batch_size * num_layers * 6 * sequence_length * d_model * d_ff,
        "lm_head": batch_size * 2 * sequence_length * d_model * vocab_size,
    }
    total_forward_flops = sum(forward_flop_breakdown.values())
    proportions = {
        component: component_flops / total_forward_flops
        for component, component_flops in forward_flop_breakdown.items()
    }
    return TransformerAccounting(
        shape=shape,
        parameter_breakdown=parameter_breakdown,
        total_parameters=total_parameters,
        parameter_memory_bytes=total_parameters * FLOAT32_BYTES,
        forward_flop_breakdown=forward_flop_breakdown,
        total_forward_flops=total_forward_flops,
        forward_flop_proportions=proportions,
    )


def activation_elements_per_batch(shape: TransformerShape) -> int:
    """Count the float32 activations explicitly listed in PDF problem adamw_accounting."""
    sequence_length = shape.context_length
    per_block = (
        8 * sequence_length * shape.d_model
        + 4 * sequence_length * shape.d_ff
        + 2 * shape.num_heads * sequence_length * sequence_length
    )
    final_rmsnorm = sequence_length * shape.d_model
    output_embedding_and_cross_entropy = 2 * sequence_length * shape.vocab_size
    return shape.num_layers * per_block + final_rmsnorm + output_embedding_and_cross_entropy


def adamw_memory_accounting(shape: TransformerShape, batch_size: int) -> AdamWMemoryAccounting:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    total_parameters = transformer_accounting(shape).total_parameters
    parameter_bytes = total_parameters * FLOAT32_BYTES
    gradient_bytes = total_parameters * FLOAT32_BYTES
    optimizer_state_bytes = 2 * total_parameters * FLOAT32_BYTES
    activation_bytes = batch_size * activation_elements_per_batch(shape) * FLOAT32_BYTES
    return AdamWMemoryAccounting(
        batch_size=batch_size,
        parameter_bytes=parameter_bytes,
        activation_bytes=activation_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        total_bytes=parameter_bytes + activation_bytes + gradient_bytes + optimizer_state_bytes,
    )


def maximum_batch_size(shape: TransformerShape, memory_bytes: int) -> int:
    if memory_bytes <= 0:
        raise ValueError("memory_bytes must be positive.")
    fixed_bytes = adamw_memory_accounting(shape, batch_size=1)
    non_activation_bytes = fixed_bytes.parameter_bytes + fixed_bytes.gradient_bytes + fixed_bytes.optimizer_state_bytes
    activation_bytes_per_batch = fixed_bytes.activation_bytes
    return max(0, (memory_bytes - non_activation_bytes) // activation_bytes_per_batch)


def adamw_update_flops(shape: TransformerShape) -> int:
    """Approximate scalar FLOPs for one AdamW update as 14 operations per parameter."""
    return 14 * transformer_accounting(shape).total_parameters


def training_step_flops(shape: TransformerShape, batch_size: int) -> int:
    forward_flops = transformer_accounting(shape, batch_size=batch_size).total_forward_flops
    return 3 * forward_flops + adamw_update_flops(shape)


def estimated_training_hours(
    shape: TransformerShape,
    batch_size: int,
    num_steps: int,
    peak_flops_per_second: float = H100_TF32_FLOPS_PER_SECOND,
    model_flops_utilization: float = 0.5,
) -> float:
    if num_steps <= 0 or peak_flops_per_second <= 0 or not 0 < model_flops_utilization <= 1:
        raise ValueError("Invalid training-time accounting arguments.")
    total_flops = training_step_flops(shape, batch_size) * num_steps
    effective_flops_per_second = peak_flops_per_second * model_flops_utilization
    return total_flops / effective_flops_per_second / 3600


def gpt2_assignment_shapes(context_length: int = 1024) -> list[TransformerShape]:
    specifications = [
        ("small", 12, 768, 12),
        ("medium", 24, 1024, 16),
        ("large", 36, 1280, 20),
        ("xl", 48, 1600, 25),
    ]
    return [
        TransformerShape(
            name=name,
            vocab_size=50_257,
            context_length=context_length,
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=assignment_d_ff(d_model),
        )
        for name, num_layers, d_model, num_heads in specifications
    ]


def accounting_as_dict(accounting: TransformerAccounting) -> dict[str, object]:
    return asdict(accounting)
