#!/usr/bin/env python3
"""Compute Transformer parameter and matrix-multiply FLOP accounting tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelShape:
    name: str
    vocab_size: int
    context_length: int
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int


GPT2_SHAPES = [
    ModelShape("gpt2-small", 50257, 1024, 12, 768, 12, 2048),
    ModelShape("gpt2-medium", 50257, 1024, 24, 1024, 16, 2752),
    ModelShape("gpt2-large", 50257, 1024, 36, 1280, 20, 3392),
    ModelShape("gpt2-xl", 50257, 1024, 48, 1600, 25, 4288),
    ModelShape("gpt2-xl-16k", 50257, 16384, 48, 1600, 25, 4288),
]


def account(shape: ModelShape) -> dict[str, int | float | str]:
    vocab, sequence, layers, width, feedforward = (
        shape.vocab_size,
        shape.context_length,
        shape.num_layers,
        shape.d_model,
        shape.d_ff,
    )
    embedding_parameters = vocab * width
    attention_parameters = layers * 4 * width * width
    feedforward_parameters = layers * 3 * width * feedforward
    norm_parameters = layers * 2 * width + width
    lm_head_parameters = vocab * width
    total_parameters = (
        embedding_parameters + attention_parameters + feedforward_parameters + norm_parameters + lm_head_parameters
    )

    attention_projection_flops = layers * 8 * sequence * width * width
    attention_matrix_flops = layers * 4 * sequence * sequence * width
    feedforward_flops = layers * 6 * sequence * width * feedforward
    lm_head_flops = 2 * sequence * width * vocab
    total_flops = attention_projection_flops + attention_matrix_flops + feedforward_flops + lm_head_flops
    return {
        **asdict(shape),
        "embedding_parameters": embedding_parameters,
        "attention_parameters": attention_parameters,
        "feedforward_parameters": feedforward_parameters,
        "norm_parameters": norm_parameters,
        "lm_head_parameters": lm_head_parameters,
        "total_parameters": total_parameters,
        "fp32_parameter_bytes": total_parameters * 4,
        "attention_projection_flops": attention_projection_flops,
        "attention_matrix_flops": attention_matrix_flops,
        "feedforward_flops": feedforward_flops,
        "lm_head_flops": lm_head_flops,
        "total_forward_matmul_flops": total_flops,
        "attention_projection_fraction": attention_projection_flops / total_flops,
        "attention_matrix_fraction": attention_matrix_flops / total_flops,
        "feedforward_fraction": feedforward_flops / total_flops,
        "lm_head_fraction": lm_head_flops / total_flops,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [account(shape) for shape in GPT2_SHAPES]
    rendered = json.dumps(rows, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
