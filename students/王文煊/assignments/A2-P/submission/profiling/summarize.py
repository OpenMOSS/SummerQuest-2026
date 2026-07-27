"""Aggregate local raw results into lightweight CSV/JSON for submission.

Reads work-repo results/ and writes sanitized summaries (no host/user/paths
beyond relative result paths) to an output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsons(pattern, root):
    for p in sorted(root.glob(pattern)):
        with open(p) as f:
            yield p, json.load(f)


def summarize_benchmark(src: Path, dst: Path):
    rows = []
    for p, d in load_jsons("bench/*.json", src):
        cfg = d["metadata"]["config"]
        rows.append({
            "run": p.stem,
            "model_size": cfg["model_size"],
            "batch_size": cfg["batch_size"],
            "context_length": cfg["context_length"],
            "mode": cfg["mode"],
            "dtype": cfg["dtype"],
            "warmup": cfg["warmup"],
            "steps": cfg["steps"],
            "mean_ms": round(d["mean_s"] * 1e3, 3),
            "stdev_ms": round(d["stdev_s"] * 1e3, 3),
            "cv": round(d["cv"], 4),
            "peak_memory_gib": round(d["peak_memory_bytes"] / 2**30, 4),
            "raw_timings_ms": ";".join(f"{t*1e3:.3f}" for t in d["timings_s"]),
            "losses": ";".join(f"{l:.5f}" for l in d["losses"]) if d.get("losses") else "",
        })
    dst.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(dst / "benchmark.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def summarize_profile(src: Path, dst: Path):
    rows = []
    meta_rows = {}
    parsed = {p.stem.replace("parsed_", ""): d for p, d in load_jsons("profile/parsed_*.json", src)}
    for p, d in load_jsons("profile/summary_*.json", src):
        cfg = d["metadata"]["config"]
        run = p.stem.replace("summary_", "")
        pr = parsed.get(run, {})
        meta_rows[run] = {
            "config": {k: cfg[k] for k in ("model_size", "batch_size", "context_length", "warmup", "dtype", "seed")},
            "tool": d.get("tool"),
            "trace_file": f"trace_{run}.json",
            "loss_of_measured_step": d.get("loss_of_measured_step"),
            "measure_span_us": pr.get("measure_span_us"),
            "stage_summary": pr.get("stage_summary"),
            "total_gpu_kernel_us_in_measure": pr.get("total_gpu_kernel_us_in_measure"),
        }
        for group, entries in (("KERNEL", pr.get("top_gpu_kernels") or []),
                               ("CPU_OP", pr.get("top_cpu_ops") or [])):
            for op in entries:
                rows.append({
                    "run": run,
                    "model_size": cfg["model_size"],
                    "batch_size": cfg["batch_size"],
                    "context_length": cfg["context_length"],
                    "dtype": cfg["dtype"],
                    "kind": group,
                    "op_or_kernel": op["key"],
                    "calls": op["calls"],
                    "cpu_time_total_us": "",
                    "cuda_time_total_us": op["total_us"],
                })
        for stage, s in (pr.get("stage_summary") or {}).items():
            rows.append({
                "run": run,
                "model_size": cfg["model_size"],
                "batch_size": cfg["batch_size"],
                "context_length": cfg["context_length"],
                "dtype": cfg["dtype"],
                "kind": "STAGE",
                "op_or_kernel": stage,
                "calls": s["calls"],
                "cpu_time_total_us": s["cpu_wall_us"],
                "cuda_time_total_us": s["cuda_time_total_us"],
            })
    dst.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(dst / "trace_summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if meta_rows:
        with open(dst / "run_metadata.json", "w") as f:
            json.dump({"runs": meta_rows}, f, indent=2, ensure_ascii=False)


def summarize_memory(src: Path, dst: Path):
    rows = []
    meta_rows = {}
    for p, d in load_jsons("memory/memsummary_*.json", src):
        cfg = d["metadata"]["config"]
        run = p.stem.replace("memsummary_", "")
        rows.append({
            "run": run,
            "model_size": cfg["model_size"],
            "batch_size": cfg["batch_size"],
            "context_length": cfg["context_length"],
            "mode": cfg["mode"],
            "dtype": cfg["dtype"],
            "active_gib_now": round(d["active_bytes_now"] / 2**30, 4),
            "peak_allocated_gib": round(d["peak_allocated_bytes"] / 2**30, 4),
            "reserved_gib_now": round(d["reserved_bytes_now"] / 2**30, 4),
            "peak_reserved_gib": round(d["peak_reserved_bytes"] / 2**30, 4),
        })
        meta_rows[run] = {
            "config": {k: cfg[k] for k in ("model_size", "batch_size", "context_length", "mode", "warmup", "steps", "dtype", "seed")},
            "snapshot_file": f"snapshot_{run}.pickle",
            "stats": {k: d[k] for k in ("active_bytes_now", "peak_allocated_bytes", "reserved_bytes_now", "peak_reserved_bytes", "num_alloc_retries")},
        }
    dst.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(dst / "peaks.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if meta_rows:
        with open(dst / "run_metadata.json", "w") as f:
            json.dump({"runs": meta_rows}, f, indent=2, ensure_ascii=False)


def summarize_mixed(src: Path, dst: Path):
    raw = src / "mixed" / "mixed_precision_raw.json"
    if not raw.exists():
        return
    with open(raw) as f:
        d = json.load(f)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "mixed_precision.json", "w") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results")
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    summarize_benchmark(src, dst)
    summarize_profile(src, dst / "profile")
    summarize_memory(src, dst / "memory")
    summarize_mixed(src, dst)
    print("summaries written to", dst)
