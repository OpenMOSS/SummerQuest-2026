from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from profiling.common import ensure_parent_dir, require_cuda, seed_everything, write_json


@dataclass
class AccumulationCase:
    case_id: str
    input_dtype: str
    accumulator_dtype: str
    output_dtype: str
    result: float
    absolute_error: float


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed precision helper experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    accumulation = subparsers.add_parser("accumulation", help="Run the four accumulation precision cases.")
    accumulation.add_argument("--output", required=True)
    accumulation.add_argument("--iters", type=int, default=1000)
    accumulation.add_argument("--disable-progress", action="store_true")

    toy = subparsers.add_parser("toy_inspect", help="Inspect dtypes under autocast on a toy model.")
    toy.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    toy.add_argument("--seed", type=int, default=42)
    toy.add_argument("--batch-size", type=int, default=8)
    toy.add_argument("--in-features", type=int, default=16)
    toy.add_argument("--out-features", type=int, default=8)
    toy.add_argument("--disable-progress", action="store_true")
    toy.add_argument("--output", required=True)
    return parser.parse_args()


def run_accumulation(args: argparse.Namespace) -> None:
    device = require_cuda()
    exact_value = 10.0
    cases: list[AccumulationCase] = []
    show_progress = not args.disable_progress

    def iter_range(desc: str):
        return tqdm(
            range(args.iters),
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )

    s = torch.tensor(0, dtype=torch.float32, device=device)
    for _ in iter_range("mp-accum:case1"):
        s += torch.tensor(0.01, dtype=torch.float32, device=device)
    cases.append(
        AccumulationCase(
            case_id="case_1",
            input_dtype="fp32",
            accumulator_dtype="fp32",
            output_dtype=str(s.dtype).replace("torch.", ""),
            result=float(s.item()),
            absolute_error=abs(float(s.item()) - exact_value),
        )
    )

    s = torch.tensor(0, dtype=torch.float16, device=device)
    for _ in iter_range("mp-accum:case2"):
        s += torch.tensor(0.01, dtype=torch.float16, device=device)
    cases.append(
        AccumulationCase(
            case_id="case_2",
            input_dtype="fp16",
            accumulator_dtype="fp16",
            output_dtype=str(s.dtype).replace("torch.", ""),
            result=float(s.item()),
            absolute_error=abs(float(s.item()) - exact_value),
        )
    )

    s = torch.tensor(0, dtype=torch.float32, device=device)
    for _ in iter_range("mp-accum:case3"):
        s += torch.tensor(0.01, dtype=torch.float16, device=device)
    cases.append(
        AccumulationCase(
            case_id="case_3",
            input_dtype="fp16",
            accumulator_dtype="fp32",
            output_dtype=str(s.dtype).replace("torch.", ""),
            result=float(s.item()),
            absolute_error=abs(float(s.item()) - exact_value),
        )
    )

    s = torch.tensor(0, dtype=torch.float32, device=device)
    for _ in iter_range("mp-accum:case4"):
        x = torch.tensor(0.01, dtype=torch.float16, device=device)
        s += x.to(torch.float32)
    cases.append(
        AccumulationCase(
            case_id="case_4",
            input_dtype="fp16_then_cast_to_fp32",
            accumulator_dtype="fp32",
            output_dtype=str(s.dtype).replace("torch.", ""),
            result=float(s.item()),
            absolute_error=abs(float(s.item()) - exact_value),
        )
    )

    payload = {
        "artifact_type": "mixed_precision_accumulation",
        "device": str(device),
        "exact_value": exact_value,
        "iters": args.iters,
        "cases": [asdict(case) for case in cases],
    }
    write_json(args.output, payload)
    print(f"Saved accumulation results to {args.output}")


def run_toy_inspect(args: argparse.Namespace) -> None:
    device = require_cuda()
    seed_everything(args.seed)
    autocast_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    model = ToyModel(args.in_features, args.out_features).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(args.batch_size, args.in_features, device=device, dtype=torch.float32)
    targets = torch.randint(0, args.out_features, (args.batch_size,), device=device)

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(
        total=6,
        desc=f"toy-inspect:{args.dtype}",
        leave=False,
        dynamic_ncols=True,
        disable=args.disable_progress,
    )
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        parameter_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
        progress.update(1)
        fc1_out = model.fc1(x)
        progress.update(1)
        relu_out = model.relu(fc1_out)
        ln_out = model.ln(relu_out)
        progress.update(1)
        logits = model.fc2(ln_out)
        progress.update(1)
        loss = F.cross_entropy(logits, targets)
        progress.update(1)
    loss.backward()
    progress.update(1)
    progress.close()

    payload: dict[str, Any] = {
        "artifact_type": "toy_autocast_dtype_inspection",
        "autocast_dtype": args.dtype,
        "device": str(device),
        "parameter_dtype": parameter_dtype,
        "fc1_output_dtype": str(fc1_out.dtype).replace("torch.", ""),
        "layernorm_output_dtype": str(ln_out.dtype).replace("torch.", ""),
        "logits_dtype": str(logits.dtype).replace("torch.", ""),
        "loss_dtype": str(loss.dtype).replace("torch.", ""),
        "gradient_dtype": str(next(model.parameters()).grad.dtype).replace("torch.", ""),
    }

    write_json(args.output, payload)
    print(f"Saved toy autocast inspection to {args.output}")


def main() -> None:
    args = parse_args()
    ensure_parent_dir(args.output)
    if args.command == "accumulation":
        run_accumulation(args)
        return
    if args.command == "toy_inspect":
        run_toy_inspect(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
