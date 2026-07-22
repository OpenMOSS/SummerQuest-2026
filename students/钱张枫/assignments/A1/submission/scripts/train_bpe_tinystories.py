"""运行 TinyStories 10K BPE 训练入口。"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)

    from cs336_basics.Part2.train_bpe_tinystories import main as implementation_main

    implementation_main()


if __name__ == "__main__":
    main()
