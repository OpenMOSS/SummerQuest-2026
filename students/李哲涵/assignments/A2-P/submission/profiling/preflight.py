from __future__ import annotations

import argparse
import json

import torch

from profiling.config import public_environment, resolve_device


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the GPU for A2-P runs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-free-mib", type=float, default=22_000)
    args = parser.parse_args()
    device = resolve_device(args.device)
    result = {
        "device": str(device),
        "environment": public_environment(device),
        "free_mib": None,
        "meets_min_free": True,
    }
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result["free_mib"] = free / (1024**2)
        result["total_mib"] = total / (1024**2)
        result["meets_min_free"] = result["free_mib"] >= args.min_free_mib
    print(json.dumps(result, indent=2))
    return 0 if result["meets_min_free"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
