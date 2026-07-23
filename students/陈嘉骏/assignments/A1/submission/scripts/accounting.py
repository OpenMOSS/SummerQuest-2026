from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace

from cs336_basics.accounting import (
    adamw_memory_accounting,
    adamw_update_flops,
    estimated_training_hours,
    gpt2_assignment_shapes,
    maximum_batch_size,
    training_step_flops,
    transformer_accounting,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the PDF Transformer and AdamW resource accounting.")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--adamw-batch-size", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    reports: list[dict[str, object]] = []
    for shape in gpt2_assignment_shapes(args.context_length):
        model_report = transformer_accounting(shape)
        report: dict[str, object] = asdict(model_report)
        if shape.name == "xl":
            report["adamw_memory"] = asdict(adamw_memory_accounting(shape, args.adamw_batch_size))
            report["maximum_batch_size_80gb"] = maximum_batch_size(shape, 80_000_000_000)
            report["adamw_update_flops"] = adamw_update_flops(shape)
            report["training_step_flops_batch_1024"] = training_step_flops(shape, 1024)
            report["training_hours_400k_steps_h100_50pct_mfu"] = estimated_training_hours(
                shape,
                batch_size=1024,
                num_steps=400_000,
            )
            long_context = replace(shape, context_length=16_384, name="xl_context_16384")
            report["context_16384"] = asdict(transformer_accounting(long_context))
        reports.append(report)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
