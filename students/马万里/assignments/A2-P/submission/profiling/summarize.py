"""Generate lightweight, machine-readable profiling summaries for A2-P.

Outputs (all under results/profile/):
  trace_summary.csv   one table with a `kind` column covering
                        kind=kernel  : GPU kernel rows  (Calls, cumulative CUDA time, phase)
                        kind=stage   : NVTX stage ranges (forward/backward/optimizer/attention/*)
                        kind=cpu_api : CUDA runtime API rows (Calls, cumulative CPU time)
                        kind=cpu_api_total : total CPU API time for the config
  run_metadata.json   per-run config/command/status (success|failed) + failure reason
"""
import csv
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

PROFILE_DIR = Path("results/profile")
OUTPUT_METADATA = PROFILE_DIR / "run_metadata.json"
OUTPUT_TRACE_SUMMARY = PROFILE_DIR / "trace_summary.csv"
FAILURES_FILE = PROFILE_DIR / "failures.jsonl"

STAGE_TEXTS = ["profile/measure", "forward", "backward", "optimizer",
               "scaled_dot_product_attention",
               "attention/scores", "attention/softmax", "attention/value"]

PROFILE_DEFAULT_BATCH = 1


# ---------------- failures / batch helpers ----------------
def load_failures() -> dict:
    failures = {}
    if not FAILURES_FILE.exists():
        return failures
    with open(FAILURES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("name"):
                failures[rec["name"]] = rec
    return failures


def load_batch(model: str, ctx: int) -> int:
    hits = sorted(PROFILE_DIR.glob(f"run_{model}_ctx{ctx}_b*_env.json"))
    for hit in reversed(hits):
        try:
            rec = json.loads(hit.read_text(encoding="utf-8"))
            return int(rec.get("config", {}).get("batch_size", PROFILE_DEFAULT_BATCH))
        except Exception:
            continue
    return PROFILE_DEFAULT_BATCH


# ---------------- metadata ----------------
def generate_metadata():
    failures = load_failures()
    names = set(failures.keys())
    for rep_file in PROFILE_DIR.glob("run_*.nsys-rep"):
        m = re.match(r"run_(small|medium|large|xl)_ctx(\d+)", rep_file.name)
        if m:
            names.add(m.group(0))
    records = []
    for name in sorted(names):
        m = re.match(r"run_(small|medium|large|xl)_ctx(\d+)", name)
        if not m:
            continue
        model, ctx = m.group(1), int(m.group(2))
        batch = load_batch(model, ctx)
        stats_file = PROFILE_DIR / f"{name}_stats.csv"
        base = {
            "model_size": model, "context_length": ctx, "batch_size": batch,
            "dtype": "fp32", "tool": "nsys", "mode": "train_step",
            "command": (f"nsys profile --output results/profile/{name} --force-overwrite true "
                        f"python profiling/profile_one_step.py --model-size {model} "
                        f"--batch-size {batch} --context-length {ctx} --dtype fp32 --warmup 5"),
        }
        if name in failures:
            rec = failures[name]
            record = dict(base)
            record.update({
                "status": "failed", "trace_file": None,
                "stage": rec.get("stage"), "exception": rec.get("exception"),
                "reason": rec.get("message"), "failure_timestamp": rec.get("timestamp"),
            })
        elif stats_file.exists():
            record = dict(base)
            record.update({"status": "success",
                           "trace_file": f"results/profile/{name}.nsys-rep"})
        else:
            record = dict(base)
            record.update({"status": "failed", "trace_file": None,
                           "reason": "no nsys trace or stats produced"})
        records.append(record)
    OUTPUT_METADATA.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    n_ok = sum(1 for r in records if r["status"] == "success")
    print(f"Metadata written to {OUTPUT_METADATA} ({n_ok} success, {len(records) - n_ok} failed)")


# ---------------- stats csv parsing ----------------
def _first_header(rows, must_have):
    for i, row in enumerate(rows):
        if all(any(m in c for c in row) for m in must_have):
            return i
    return None


def parse_kernel_rows(stats_file: Path):
    """GPU kernel table (first table; header contains 'Instances')."""
    with open(stats_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hi = _first_header(rows, ["Name", "Instances", "Total Time"])
    if hi is None:
        return []
    header = rows[hi]
    name_col = next(j for j, h in enumerate(header) if "Name" in h)
    inst_col = next(j for j, h in enumerate(header) if "Instances" in h)
    time_col = next(j for j, h in enumerate(header) if "Total Time" in h)
    out = []
    for row in rows[hi + 1:]:
        if not row or all(c.strip() == "" for c in row):
            continue
        if any(c.strip() == "Name" for c in row):   # start of the 2nd (CUDA API) table
            break
        try:
            out.append((row[name_col].strip(),
                        int(row[inst_col].strip().replace(",", "")),
                        float(row[time_col].strip().replace(",", ""))))
        except (ValueError, IndexError):
            continue
    return sorted(out, key=lambda r: r[2], reverse=True)


def parse_cpu_api_rows(stats_file: Path):
    """CUDA runtime API table (second table; header contains 'Num Calls')."""
    with open(stats_file, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    hi = _first_header(rows, ["Name", "Num Calls", "Total Time"])
    if hi is None:
        return []
    header = rows[hi]
    name_col = next(j for j, h in enumerate(header) if "Name" in h)
    call_col = next(j for j, h in enumerate(header) if "Num Calls" in h or "Calls" in h)
    time_col = next(j for j, h in enumerate(header) if "Total Time" in h)
    out = []
    for row in rows[hi + 1:]:
        if not row or all(c.strip() == "" for c in row):
            continue
        try:
            out.append((row[name_col].strip(),
                        int(row[call_col].strip().replace(",", "")),
                        float(row[time_col].strip().replace(",", ""))))
        except (ValueError, IndexError):
            continue
    return sorted(out, key=lambda r: r[2], reverse=True)


# ---------------- NVTX stage ranges from the sqlite export ----------------
def parse_stage_rows(sqlite_file: Path):
    """Return {stage: (n_ranges, duration_ns)} from NVTX_EVENTS inside profile/measure."""
    if not sqlite_file.exists():
        return {}
    con = sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)
    cur = con.cursor()
    marks = ",".join("?" * len(STAGE_TEXTS))
    nv = {}
    try:
        for start, end, text in cur.execute(
                f"SELECT start, end, text FROM NVTX_EVENTS WHERE text IN ({marks})", STAGE_TEXTS):
            if end >= start:
                nv.setdefault(text, []).append((start, end))
    except sqlite3.Error:
        return {}
    finally:
        con.close()
    if "profile/measure" not in nv:
        return {}
    m0 = min(s for s, _ in nv["profile/measure"])
    m1 = max(e for _, e in nv["profile/measure"])
    out = {}
    for text, spans in nv.items():
        inside = [(s, e) for s, e in spans if s >= m0 and e <= m1]
        if inside:
            # 多区间（如每层一次 attention/*）取各区间时长之和；单区间时和=跨度。
            total = sum(e - s for s, e in inside)
            out[text] = (len(inside), total)
    return out


def generate_trace_summary():
    rows = []
    def add(model, ctx, kind, name, calls, cpu_us, cuda_us, phase):
        rows.append({"model_size": model, "context_length": ctx, "kind": kind, "name": name,
                     "calls": calls, "cpu_time_us": cpu_us, "cuda_time_us": cuda_us, "phase": phase})

    for stats_file in sorted(PROFILE_DIR.glob("run_*_stats.csv")):
        m = re.match(r"run_(small|medium|large|xl)_ctx(\d+)_stats", stats_file.stem)
        if not m:
            continue
        model, ctx = m.group(1), int(m.group(2))
        print(f"Processing {stats_file.name} ...")

        for name, calls, ns in parse_kernel_rows(stats_file)[:5]:
            add(model, ctx, "kernel", name, calls, "", round(ns / 1000.0, 2), infer_phase(name))

        cpu_rows = parse_cpu_api_rows(stats_file)
        for name, calls, ns in cpu_rows[:5]:
            add(model, ctx, "cpu_api", name, calls, round(ns / 1000.0, 2), "", "cpu_api")
        if cpu_rows:
            total_ns = sum(ns for _, _, ns in cpu_rows)
            total_calls = sum(c for _, c, _ in cpu_rows)
            add(model, ctx, "cpu_api_total", "cuda_api_total", total_calls,
                round(total_ns / 1000.0, 2), "", "cpu_api")

        stage_rows = parse_stage_rows(PROFILE_DIR / f"run_{model}_ctx{ctx}.sqlite")
        for stage in ["forward", "backward", "optimizer",
                      "attention/scores", "attention/softmax", "attention/value"]:
            if stage in stage_rows:
                n, ns = stage_rows[stage]
                add(model, ctx, "stage", stage, n, "", round(ns / 1000.0, 2), stage)

    result_df = pd.DataFrame(rows, columns=["model_size", "context_length", "kind", "name",
                                            "calls", "cpu_time_us", "cuda_time_us", "phase"])
    result_df.to_csv(OUTPUT_TRACE_SUMMARY, index=False)
    print(f"trace_summary.csv written to {OUTPUT_TRACE_SUMMARY} ({len(result_df)} rows)")


def infer_phase(kernel_name: str) -> str:
    n = kernel_name.lower()
    if any(k in n for k in ["gemm", "matmul", "sgemm", "cublas"]):
        return "matmul"
    if "softmax" in n:
        return "attention/softmax"
    if "flash" in n or "attn" in n:
        return "attention"
    if "norm" in n:
        return "layernorm"
    if "adam" in n or "optimizer" in n:
        return "optimizer"
    if any(k in n for k in ["elementwise", "add", "mul", "div"]):
        return "elementwise"
    if any(k in n for k in ["reduce", "sum", "mean"]):
        return "reduce"
    if "embed" in n:
        return "embedding"
    if "dropout" in n:
        return "dropout"
    return "other"


if __name__ == "__main__":
    generate_metadata()
    generate_trace_summary()
