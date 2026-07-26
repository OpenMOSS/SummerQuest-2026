"""把 A2-P JSONL 实验记录压平成轻量 CSV。"""

from __future__ import annotations

import argparse
from pathlib import Path

from profiling.result_tables import main as result_tables_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize A2-P JSONL records.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # result_tables 的 CLI 已经过测试；这里将稳定提交入口映射到同一实现。
    import sys

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *(str(path) for path in args.inputs), "--output-dir", str(args.output_dir)]
        return result_tables_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
