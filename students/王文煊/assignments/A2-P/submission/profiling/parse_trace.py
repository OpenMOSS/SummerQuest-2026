"""Parse exported Chrome traces from nvtx_ranges.py into reliable summaries.

For each trace_*.json:
- locate the "profile/measure" CPU range of the captured step;
- stage wall times = CPU durations of forward/backward/optimizer/attention/*
  record_function ranges inside the measurement span;
- CUDA time per stage = sum of GPU kernel/memcpy events overlapping the
  stage span (on GPU streams);
- top ops = aggregated CPU ops (cpu_op) and top kernels = aggregated GPU
  kernel events (kernel) inside the measurement span.
Writes parsed_<tag>.json next to the trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = ["forward", "backward", "optimizer",
          "attention/scores", "attention/softmax", "attention/value"]


def parse_trace(path: Path):
    with open(path) as f:
        trace = json.load(f)
    events = trace["traceEvents"]

    measure = [e for e in events if e.get("name") == "profile/measure"
               and e.get("ph") == "X" and e.get("cat") == "user_annotation"]
    if not measure:
        raise RuntimeError(f"no profile/measure range in {path}")
    m = max(measure, key=lambda e: e["dur"])
    t0, t1 = m["ts"], m["ts"] + m["dur"]

    in_span = lambda e: e.get("ph") == "X" and e.get("ts", 0) >= t0 - 1 and e.get("ts", 0) + e.get("dur", 0) <= t1 + 1

    # stage CPU wall-time; attention sub-stages are summed over all per-layer calls
    stage_summary = {}
    for stage in ["profile/measure"] + STAGES:
        matches = [e for e in events if e.get("name") == stage
                   and e.get("cat") == "user_annotation" and in_span(e)]
        if not matches:
            continue
        cpu_us = 0.0
        cuda_us = 0.0
        kernels = {}
        # CUDA overlap uses the CPU range span: backward's gpu_user_annotation
        # is degenerate (~0us), while kernels provably overlap the CPU span.
        spans = [(e["ts"], e["ts"] + e["dur"]) for e in matches]
        for e in matches:
            cpu_us += e["dur"]
        for s0, s1 in spans:
            for g in events:
                if g.get("ph") != "X" or g.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
                    continue
                g0, g1 = g["ts"], g["ts"] + g["dur"]
                ov = max(0.0, min(g1, s1) - max(g0, s0))
                if ov > 0:
                    cuda_us += ov
                    k = kernels.setdefault(g["name"], [0, 0.0])
                    k[0] += 1
                    k[1] += g["dur"]
        stage_summary[stage] = {
            "calls": len(matches),
            "cpu_wall_us": round(cpu_us, 1),
            "cuda_time_total_us": round(cuda_us, 1),
        }
        if stage.startswith("attention/"):
            stage_summary[stage]["top_kernels"] = [
                {"name": n, "calls": c, "cuda_us": round(d, 1)}
                for n, (c, d) in sorted(kernels.items(), key=lambda kv: -kv[1][1])[:5]
            ]

    # aggregated CPU ops and GPU kernels over the measurement span
    cpu_ops, gpu_kernels = {}, {}
    for e in events:
        if not in_span(e):
            continue
        cat = e.get("cat")
        if cat == "cpu_op":
            k = cpu_ops.setdefault(e["name"], [0, 0.0])
            k[0] += 1
            k[1] += e["dur"]
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            k = gpu_kernels.setdefault(e["name"], [0, 0.0])
            k[0] += 1
            k[1] += e["dur"]

    def top(d, n):
        return [{"key": name, "calls": c, "total_us": round(v, 1)}
                for name, (c, v) in sorted(d.items(), key=lambda kv: -kv[1][1])[:n]]

    return {
        "trace_file": path.name,
        "measure_span_us": round(m["dur"], 1),
        "stage_summary": stage_summary,
        "top_cpu_ops": top(cpu_ops, 25),
        "top_gpu_kernels": top(gpu_kernels, 25),
        "total_gpu_kernel_us_in_measure": round(sum(v[1] for v in gpu_kernels.values()), 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", default="results/profile")
    a = ap.parse_args()
    d = Path(a.profile_dir)
    for tr in sorted(d.glob("trace_*.json")):
        tag = tr.stem.replace("trace_", "")
        out = parse_trace(tr)
        with open(d / f"parsed_{tag}.json", "w") as f:
            json.dump(out, f, indent=2)
        ss = out["stage_summary"]
        print(tag, f"measure={out['measure_span_us']/1e3:.1f}ms",
              " ".join(f"{s}={ss[s]['cuda_time_total_us']/1e3:.1f}ms" for s in STAGES if s in ss))
