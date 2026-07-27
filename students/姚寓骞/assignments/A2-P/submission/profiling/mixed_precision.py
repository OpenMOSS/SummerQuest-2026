"""Run the four accumulation snippets specified by the fixed assignment PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def run_experiments() -> dict:
    experiments = []

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    experiments.append(("fp32_accumulator_fp32_addend", s))

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    experiments.append(("fp16_accumulator_fp16_addend", s))

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    experiments.append(("fp32_accumulator_fp16_addend", s))

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    experiments.append(("fp32_accumulator_explicit_fp16_to_fp32", s))

    expected = 10.0
    return {
        "operation": "sequentially add 0.01 1000 times",
        "expected": expected,
        "experiments": [
            {
                "name": name,
                "result": value.item(),
                "result_dtype": str(value.dtype),
                "absolute_error": abs(value.item() - expected),
            }
            for name, value in experiments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_experiments()
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
