from __future__ import annotations

from pathlib import Path

import torch

from profiling.benchmark import parser, run


def main():
    p = parser()
    p.add_argument("--snapshot", default="results/memory/snapshot.pickle")
    args = p.parse_args()
    if not torch.cuda.is_available():
        p.error("memory snapshots require a CUDA device")
    path = Path(args.snapshot)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    try:
        run(args)
        torch.cuda.memory._dump_snapshot(str(path))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    main()
