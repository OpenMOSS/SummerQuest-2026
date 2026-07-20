from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(paths: list[Path]) -> list[dict]:
    records = []
    for path in paths:
        source = path.parent.name if path.stem in {"train", "eval", "log"} else path.stem
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    record["_source"] = source
                    records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/validation loss curves from JSONL logs.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    records = load_records([Path(path) for path in args.inputs])
    fig, ax = plt.subplots(figsize=(8, 5))
    for source in sorted({record["_source"] for record in records}):
        subset = [record for record in records if record["_source"] == source]
        steps = [record["step"] for record in subset]
        train = [record["train_loss"] for record in subset]
        ax.plot(steps, train, label=f"{source} train", linewidth=1.4)
        val_points = [(record["step"], record["val_loss"]) for record in subset if record.get("val_loss") is not None]
        if val_points:
            ax.scatter([p[0] for p in val_points], [p[1] for p in val_points], label=f"{source} val", s=16)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    if args.title:
        ax.set_title(args.title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=160)


if __name__ == "__main__":
    main()
