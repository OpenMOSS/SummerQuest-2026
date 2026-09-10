"""Generate lightweight, machine-readable profiling summaries for A2-P.

统计口径统一为**单个预热后的 measurement step**：nsys 会把整个进程写进 trace
(5 个 warm-up step + 1 个 measure step)，所以这里不读整份 `nsys stats` 总表，
而是按 NVTX `profile/measure` 区间的时间边界过滤 SQLite 里的 kernel / CUDA API 行，
让 kernel、CUDA API 与 NVTX stage 三类行口径一致。

Outputs (all under results/profile/):
  trace_summary.csv   one table with a `kind` column covering
                        kind=kernel  : GPU kernel rows (Calls, cumulative cuda_time_us, phase)
                        kind=kernel_total : all GPU kernels inside the measure window
                        kind=stage   : NVTX stage ranges (wall/host duration -> stage_time_us)
                        kind=cpu_api : CUDA runtime API rows (Calls, cumulative cpu_time_us)
                        kind=cpu_api_total : total CPU API time for the config
  run_metadata.json   per-run config/command/status (success|failed) + summary scope
"""
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

SUMMARY_SCOPE = ("kernel / CUDA API / stage rows are all filtered to the NVTX "
                 "`profile/measure` window via SQLite timestamps")
CAPTURE_SCOPE = ("nsys captures the whole process (5 warm-up steps + 1 measure step); "
                 "the reported numbers only use the single measurement step")

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
                           "trace_file": f"results/profile/{name}.nsys-rep",
                           "capture_scope": CAPTURE_SCOPE,
                           "summary_scope": SUMMARY_SCOPE})
            window = measure_window(PROFILE_DIR / f"{name}.sqlite")
            if window is not None:
                m0, m1 = window
                record.update({"measure_window_ns": [m0, m1],
                               "measure_window_us": round((m1 - m0) / 1000.0, 3)})
        else:
            record = dict(base)
            record.update({"status": "failed", "trace_file": None,
                           "reason": "no nsys trace or stats produced"})
        records.append(record)
    OUTPUT_METADATA.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    n_ok = sum(1 for r in records if r["status"] == "success")
    print(f"Metadata written to {OUTPUT_METADATA} ({n_ok} success, {len(records) - n_ok} failed)")


# ---------------- measure window / sqlite aggregation ----------------
def _connect(sqlite_file: Path):
    return sqlite3.connect(f"file:{sqlite_file}?mode=ro", uri=True)


def measure_window(sqlite_file: Path):
    """Return (start_ns, end_ns) of the `profile/measure` NVTX range, or None."""
    if not sqlite_file.exists():
        return None
    con = _connect(sqlite_file)
    try:
        rows = list(con.execute(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE text = 'profile/measure' AND end >= start"))
    except sqlite3.Error:
        return None
    finally:
        con.close()
    if not rows:
        return None
    return min(s for s, _ in rows), max(e for _, e in rows)


def _normalize_api_name(name: str) -> str:
    """nsys 的 sqlite 里 CUDA API 名带版本后缀（cudaLaunchKernel_v7000）。"""
    return re.sub(r"_v\d+$", "", name)


def aggregate_activity_rows(sqlite_file: Path, table: str, name_column: str,
                            m0: int, m1: int):
    """Group kernel / CUDA API rows by name inside [m0, m1).

    Returns (name, calls, duration_ns) sorted by duration descending. The name is
    resolved through StringIds when the column is an id; API version suffixes are
    stripped before grouping so that rows split by version still merge.
    """
    sql = (f"SELECT COALESCE(s.value, CAST(t.{name_column} AS TEXT)), COUNT(*), "
           f"SUM(t.end - t.start) FROM {table} AS t "
           f"LEFT JOIN StringIds AS s ON s.id = t.{name_column} "
           f"WHERE t.start >= ? AND t.start < ? GROUP BY 1")
    merged: dict = {}
    con = _connect(sqlite_file)
    try:
        rows = list(con.execute(sql, (m0, m1)))
    except sqlite3.Error:
        return []
    finally:
        con.close()
    for raw, calls, ns in rows:
        key = _normalize_api_name(raw if raw else "[unknown]")
        acc = merged.setdefault(key, [0, 0])
        acc[0] += calls
        acc[1] += ns or 0
    return sorted(((k, c, d) for k, (c, d) in merged.items()),
                  key=lambda r: r[2], reverse=True)


# ---------------- NVTX stage ranges from the sqlite export ----------------
def parse_stage_rows(sqlite_file: Path):
    """Return {stage: (n_ranges, duration_ns)} from NVTX_EVENTS inside profile/measure."""
    if not sqlite_file.exists():
        return {}
    con = _connect(sqlite_file)
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

    def add(model, ctx, kind, name, calls, cpu_us, cuda_us, stage_us, phase):
        rows.append({"model_size": model, "context_length": ctx, "kind": kind, "name": name,
                     "calls": calls, "cpu_time_us": cpu_us, "cuda_time_us": cuda_us,
                     "stage_time_us": stage_us, "phase": phase})

    for sqlite_file in sorted(PROFILE_DIR.glob("run_*_ctx*.sqlite")):
        m = re.match(r"run_(small|medium|large|xl)_ctx(\d+)\.sqlite", sqlite_file.name)
        if not m:
            continue
        model, ctx = m.group(1), int(m.group(2))
        window = measure_window(sqlite_file)
        if window is None:
            print(f"[skip] {sqlite_file.name}: no profile/measure NVTX range")
            continue
        m0, m1 = window
        print(f"Processing {sqlite_file.name} (measure window {(m1 - m0) / 1e3:.1f} us) ...")

        kernel_rows = aggregate_activity_rows(
            sqlite_file, "CUPTI_ACTIVITY_KIND_KERNEL", "demangledName", m0, m1)
        for name, calls, ns in kernel_rows[:5]:
            add(model, ctx, "kernel", name, calls, "", round(ns / 1000.0, 2), "",
                infer_phase(name))
        if kernel_rows:
            add(model, ctx, "kernel_total", "gpu_kernel_total",
                sum(c for _, c, _ in kernel_rows),
                "", round(sum(ns for _, _, ns in kernel_rows) / 1000.0, 2), "", "kernel")

        cpu_rows = aggregate_activity_rows(
            sqlite_file, "CUPTI_ACTIVITY_KIND_RUNTIME", "nameId", m0, m1)
        for name, calls, ns in cpu_rows[:5]:
            add(model, ctx, "cpu_api", name, calls, round(ns / 1000.0, 2), "", "", "cpu_api")
        if cpu_rows:
            add(model, ctx, "cpu_api_total", "cuda_api_total",
                sum(c for _, c, _ in cpu_rows),
                round(sum(ns for _, _, ns in cpu_rows) / 1000.0, 2), "", "", "cpu_api")

        stage_rows = parse_stage_rows(sqlite_file)
        for stage in ["profile/measure", "forward", "backward", "optimizer",
                      "attention/scores", "attention/softmax", "attention/value"]:
            if stage in stage_rows:
                n, ns = stage_rows[stage]
                add(model, ctx, "stage", stage, n, "", "", round(ns / 1000.0, 2), stage)

    result_df = pd.DataFrame(rows, columns=["model_size", "context_length", "kind", "name",
                                            "calls", "cpu_time_us", "cuda_time_us",
                                            "stage_time_us", "phase"])
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
