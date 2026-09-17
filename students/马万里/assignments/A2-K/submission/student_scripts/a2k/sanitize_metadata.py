"""把 `*_metadata.jsonl` 的 git 提交替换为固定 starter commit，并统一为相对路径。

在正式矩阵重跑完成后调用；只改写 `command` 里的绝对路径前缀与 `commit` 字段。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FIXED_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    prefix = str(args.repo_root.resolve()) + "/"
    for path in sorted(args.results_dir.glob("*_metadata.jsonl")):
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            command = obj.get("command")
            if isinstance(command, list):
                obj["command"] = [
                    item[len(prefix):] if isinstance(item, str) and item.startswith(prefix) else item
                    for item in command
                ]
            if "commit" in obj:
                obj["commit"] = FIXED_COMMIT
            rows.append(json.dumps(obj, ensure_ascii=False))
        path.write_text("\n".join(rows) + "\n")
        print(f"{path.name}: {len(rows)} rows -> relative paths, commit pinned")


if __name__ == "__main__":
    main()
