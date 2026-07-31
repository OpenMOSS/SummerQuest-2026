from __future__ import annotations

from pathlib import Path

import torch

from profiling.benchmark import parser, run


if __name__ == "__main__":
    p = parser()
    p.add_argument("--trace", default="results/torch/trace.json")
    args = p.parse_args()
    trace = Path(args.trace)
    trace.parent.mkdir(parents=True, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        run(args)
    prof.export_chrome_trace(str(trace))
