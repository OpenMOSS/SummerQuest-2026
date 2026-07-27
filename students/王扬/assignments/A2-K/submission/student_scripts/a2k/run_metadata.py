from __future__ import annotations

import argparse

from student_scripts.a2k.common import add_common_args, cuda_metadata, ensure_dirs, set_allocator_limit, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()

    ensure_dirs()
    allocator = set_allocator_limit()
    metadata = cuda_metadata()
    metadata.update(
        {
            "seed": args.seed,
            "commands": args.command,
            "allocator": allocator,
            "benchmark_timer": "triton.testing.do_bench or CUDA events",
            "warmup": "attention do_bench warmup=100 ms; checkpoint warmup_steps=3",
            "measurement": "attention do_bench rep=300 ms; checkpoint measurement_steps=5",
            "compile_config": {
                "backend": "inductor",
                "mode": None,
                "fullgraph": None,
                "dynamic": None,
                "cache_policy": "cold compile measured separately from steady-state",
            },
        }
    )
    write_json(args.output_dir / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()
