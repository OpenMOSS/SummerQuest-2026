"""Quantify tensors retained for one XL TransformerBlock backward."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from profiling.common import MIB, configure_gpu, gpu_metadata, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    allocator = configure_gpu()
    gpu = gpu_metadata()
    torch.manual_seed(2026)
    positional_encoder = RotaryEmbedding(args.context_length, 80).cuda()
    block = TransformerBlock(
        d_model=2560,
        num_heads=32,
        d_ff=10240,
        positional_encoder=positional_encoder,
    ).cuda()
    inputs = torch.randn(
        1,
        args.context_length,
        2560,
        device="cuda",
        requires_grad=True,
    )
    parameter_storages = {parameter.untyped_storage().data_ptr() for parameter in block.parameters()}
    records: list[dict] = []

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        storage = tensor.untyped_storage()
        storage_pointer = storage.data_ptr()
        if storage_pointer not in parameter_storages:
            producer = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else "leaf_or_detached"
            records.append(
                {
                    "producer": producer,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "referenced_bytes": tensor.numel() * tensor.element_size(),
                    "storage_bytes": storage.nbytes(),
                    "storage_key": str(storage_pointer),
                }
            )
        return tensor

    def unpack(tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    torch.cuda.reset_peak_memory_stats()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        output = block(inputs)
        output.float().square().mean().backward()
    torch.cuda.synchronize()

    by_producer: dict[str, int] = defaultdict(int)
    unique_storages: dict[str, tuple[str, int]] = {}
    for record in records:
        producer = record["producer"]
        by_producer[producer] += record["referenced_bytes"]
        unique_storages.setdefault(
            record["storage_key"],
            (producer, record["storage_bytes"]),
        )
    unique_by_producer: dict[str, int] = defaultdict(int)
    for producer, storage_bytes in unique_storages.values():
        unique_by_producer[producer] += storage_bytes
    total_unique = sum(unique_by_producer.values())
    contributors = sorted(
        (
            {
                "producer": producer,
                "referenced_mib": by_producer[producer] / MIB,
                "unique_storage_mib": storage_bytes / MIB,
                "percentage_of_unique_saved_bytes": (100 * storage_bytes / total_unique if total_unique else 0.0),
            }
            for producer, storage_bytes in unique_by_producer.items()
        ),
        key=lambda row: row["unique_storage_mib"],
        reverse=True,
    )
    write_json(
        args.output,
        {
            "configuration": {
                "model_size": "xl",
                "scope": "one TransformerBlock",
                "batch_size": 1,
                "context_length": args.context_length,
                "dtype": "fp32",
            },
            "saved_tensor_references": len(records),
            "unique_saved_storage_mib": total_unique / MIB,
            "top_five_contributors": contributors[:5],
            "all_contributors": contributors,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
            "allocator": allocator,
            "gpu": gpu,
            "method": "torch.autograd.graph.saved_tensors_hooks",
            "command": ("python -m profiling.saved_tensor_analysis --context-length 128 --output results/memory/saved_tensors.json"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
