# 单变量 sweep，老配置复制一份，只拧一个旋钮。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.training import train  # noqa: E402


def parse_value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    summaries = []
    for raw_value in args.values:
        value = parse_value(raw_value)
        config = dict(base_config)
        # 就改这一项，其他照抄，免得 sweep 变成玄学大乱炖。
        config[args.parameter] = value
        suffix = str(value).replace(".", "p").replace("-", "m")
        config["run_name"] = f"{base_config.get('run_name', 'run')}_{args.parameter}_{suffix}"
        config["output_dir"] = str(args.output_root / config["run_name"])
        summaries.append(train(config))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "sweep_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
