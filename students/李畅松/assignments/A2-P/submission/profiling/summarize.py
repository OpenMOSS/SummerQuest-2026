from __future__ import annotations

import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    args = p.parse_args()
    print("| model | context | mode | dtype | mean (ms) | std (ms) | CV |")
    print("|---|---:|---|---|---:|---:|---:|")
    for filename in args.inputs:
        x = json.loads(Path(filename).read_text())
        c = x["config"]
        print(f'| {c["model_size"]} | {c["context_length"]} | {c["mode"]} | {c["dtype"]} | {x["mean_ms"]:.3f} | {x["std_ms"]:.3f} | {x["cv"]:.3f} |')
