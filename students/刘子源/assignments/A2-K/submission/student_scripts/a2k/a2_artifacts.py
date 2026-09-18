from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def records(root: Path) -> list[dict]:
    out = []
    for path in sorted(root.rglob("*.json")):
        if "smoke" in path.name:
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(item, dict) and "kind" in item:
            item["result_file"] = str(path.relative_to(root))
            out.append(item)
    return out


def flatten(item: dict) -> dict:
    row = {}
    for key, value in item.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                row[f"{key}_{child_key}"] = child_value
        elif isinstance(value, list):
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = value
    return row


def write_summary(items: list[dict], target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    (target / "all_results.json").write_text(json.dumps(items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = [flatten(item) for item in items]
    fields = sorted({key for row in rows for key in row})
    with (target / "all_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plots(items: list[dict], target: Path, kind: str) -> None:
    import matplotlib.pyplot as plt

    target.mkdir(parents=True, exist_ok=True)
    if kind == "a2p":
        base = [x for x in items if x["kind"] == "model_benchmark" and x.get("model_size") == "small" and x.get("dtype") == "float32" and x.get("status") == "ok"]
        if base:
            labels = [f"{x['mode']}\nw{x['warmup']}" for x in base]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.bar(labels, [x["mean_seconds"] for x in base], yerr=[x.get("sample_stdev_seconds", 0) for x in base], capsize=4)
            ax.set(ylabel="seconds / step", title="A2-P: CUDA-synchronized baseline timing")
            fig.tight_layout(); fig.savefig(target / "benchmark_baseline.png", dpi=180); plt.close(fig)
        precision = [x for x in items if x["kind"] == "model_benchmark" and x.get("dtype") == "bfloat16" and x.get("status") == "ok"]
        if precision:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for mode in sorted({x["mode"] for x in precision}):
                data = [x for x in precision if x["mode"] == mode]
                ax.plot([x["model_size"] for x in data], [x["mean_seconds"] for x in data], marker="o", label=mode)
            ax.set(ylabel="seconds / step", title="A2-P: BF16 autocast timing")
            ax.legend(); fig.tight_layout(); fig.savefig(target / "mixed_precision.png", dpi=180); plt.close(fig)
        memory = [x for x in items if x["kind"] == "memory_profile" and x.get("memory") and x.get("status") == "ok"]
        if memory:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            labels = [f"{x['model_size']} L={x['context_length']}\n{x['mode']}" for x in memory]
            vals = [x["memory"]["allocated_peak_bytes"] / 2**30 for x in memory]
            ax.bar(labels, vals)
            ax.set(ylabel="peak allocated GiB", title="A2-P: peak allocated memory")
            fig.tight_layout(); fig.savefig(target / "memory_peak.png", dpi=180); plt.close(fig)
        for item in memory:
            stages = item.get("stages", [])
            if not stages:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.5))
            names = [row["stage"] for row in stages]
            for field, label in (("allocated_bytes", "allocated"), ("reserved_bytes", "reserved"), ("active_bytes", "active")):
                ax.plot(names, [row[field] / 2**30 for row in stages], marker="o", label=label)
            ax.set(ylabel="GiB", title=f"A2-P: {item['model_size']} L={item['context_length']} {item['mode']} memory stages")
            ax.legend(); fig.tight_layout()
            fig.savefig(target / f"memory_timeline_{item['model_size']}_l{item['context_length']}_{item['mode']}.png", dpi=180)
            plt.close(fig)
    else:
        ckpt = [x for x in items if x["kind"] == "checkpoint" and x.get("context_length") == 1024 and x.get("status") == "ok"]
        if ckpt:
            ckpt.sort(key=lambda x: x["block_size"])
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot([x["block_size"] for x in ckpt], [x["memory"]["allocated_peak_bytes"] / 2**30 for x in ckpt], marker="o", label="peak allocated")
            ax2 = ax.twinx(); ax2.plot([x["block_size"] for x in ckpt], [x["mean_seconds"] for x in ckpt], marker="s", color="tab:orange", label="step time")
            ax.set(xlabel="checkpoint block size (0 = none)", ylabel="peak allocated GiB", title="A2-K: checkpoint trade-off")
            ax2.set_ylabel("seconds / train step")
            fig.tight_layout(); fig.savefig(target / "checkpoint_tradeoff.png", dpi=180); plt.close(fig)
        attn = [x for x in items if x["kind"] == "attention" and x.get("sequence_length") in (512, 2048, 8192) and x.get("dimension") == 128 and x.get("phase") == "forward_backward" and x.get("status") == "ok"]
        if attn:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for impl in sorted({x["implementation"] for x in attn}):
                data = sorted((x for x in attn if x["implementation"] == impl), key=lambda x: x["sequence_length"])
                ax.plot([x["sequence_length"] for x in data], [x["p50_seconds"] for x in data], marker="o", label=impl)
            ax.set(xlabel="sequence length", ylabel="p50 seconds", xscale="log", title="A2-K: attention forward + backward")
            ax.legend(); fig.tight_layout(); fig.savefig(target / "attention_latency.png", dpi=180); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--kind", choices=["a2p", "a2k"], required=True)
    args = parser.parse_args()
    items = records(Path(args.source))
    write_summary(items, Path(args.results))
    plots(items, Path(args.assets), args.kind)
    print(f"wrote {len(items)} rows")


if __name__ == "__main__":
    main()
