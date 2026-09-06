import csv
import json
import re
from pathlib import Path
import pandas as pd

PROFILE_DIR = Path("results/profile")
OUTPUT_METADATA = PROFILE_DIR / "run_metadata.json"
OUTPUT_TRACE_SUMMARY = PROFILE_DIR / "trace_summary.csv"
FAILURES_FILE = PROFILE_DIR / "failures.jsonl"

# ---------- Failure records (crash jsonl) ----------
def load_failures() -> dict:
    """Return {run_name: record} parsed from failures.jsonl (if present)."""
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
            name = rec.get("name")
            if name:
                failures[name] = rec
    return failures


PROFILE_DEFAULT_BATCH = 1

def load_batch(model: str, ctx: int) -> int:
    """Read the batch_size actually used from the run's env json (profile_one_step writes it).

    Falls back to PROFILE_DEFAULT_BATCH if no env json is present.
    """
    hits = sorted(PROFILE_DIR.glob(f"run_{model}_ctx{ctx}_b*_env.json"))
    for hit in reversed(hits):
        try:
            rec = json.loads(hit.read_text(encoding="utf-8"))
            return int(rec.get("config", {}).get("batch_size", PROFILE_DEFAULT_BATCH))
        except Exception:
            continue
    return PROFILE_DEFAULT_BATCH

# ---------- Metadata generation ----------
def generate_metadata():
    failures = load_failures()
    # 需要覆盖的 run 名 = 成功产出 .nsys-rep 的 + 记录在 failures.jsonl 的（失败配置）。
    names = set(failures.keys())
    for rep_file in PROFILE_DIR.glob("run_*.nsys-rep"):
        match = re.match(r"run_(small|medium|large|xl)_ctx(\d+)", rep_file.name)
        if match:
            names.add(match.group(0))
    records = []
    for name in sorted(names):
        match = re.match(r"run_(small|medium|large|xl)_ctx(\d+)", name)
        if not match:
            continue
        model = match.group(1)
        ctx = int(match.group(2))
        batch = load_batch(model, ctx)
        stats_file = PROFILE_DIR / f"{name}_stats.csv"
        base = {
            "model_size": model,
            "context_length": ctx,
            "batch_size": batch,
            "dtype": "fp32",
            "tool": "nsys",
            "command": (f"nsys profile --output {PROFILE_DIR / name} --force-overwrite true "
                        f"python profiling/profile_one_step.py --model-size {model} "
                        f"--batch-size {batch} --context-length {ctx} --dtype fp32 --warmup 5"),
        }
        if name in failures:
            rec = failures[name]
            cfg = rec.get("config", {})
            record = dict(base)
            record.update({
                "batch_size": cfg.get("batch_size", batch),
                "trace_file": None,
                "status": "failed",
                "stage": rec.get("stage"),
                "exception": rec.get("exception"),
                "reason": rec.get("message"),
                "failure_timestamp": rec.get("timestamp"),
            })
        elif stats_file.exists():
            record = dict(base)
            record.update({
                "trace_file": str(PROFILE_DIR / f"{name}.nsys-rep"),
                "status": "success",
            })
        else:
            record = dict(base)
            record.update({
                "trace_file": None,
                "status": "failed",
                "reason": "no nsys trace or stats produced",
            })
        records.append(record)
    with open(OUTPUT_METADATA, "w") as f:
        json.dump(records, f, indent=2)
    n_succ = sum(1 for r in records if r["status"] == "success")
    n_fail = len(records) - n_succ
    print(f"Metadata written to {OUTPUT_METADATA} ({n_succ} success, {n_fail} failed)")

# ---------- Trace summary generation ----------
def infer_phase(kernel_name: str) -> str:
    name = kernel_name.lower()
    if any(k in name for k in ["gemm", "matmul", "sgemm", "cublas"]):
        return "matmul"
    if "softmax" in name:
        return "attention/softmax"
    if "flash" in name or "attn" in name:
        return "attention"
    if "norm" in name:
        return "layernorm"
    if "adam" in name or "optimizer" in name:
        return "optimizer"
    if any(k in name for k in ["elementwise", "add", "mul", "div"]):
        return "elementwise"
    if any(k in name for k in ["reduce", "sum", "mean"]):
        return "reduce"
    if "embed" in name:
        return "embedding"
    if "dropout" in name:
        return "dropout"
    return "other"

def parse_kernel_stats(stats_file: Path):
    """Extract top kernels from nsys stats CSV using csv module for robustness."""
    with open(stats_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    header_idx = None
    for i, row in enumerate(rows):
        if any("Name" in cell for cell in row) and any("Instances" in cell for cell in row):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Kernel table header not found in {stats_file}")

    header = rows[header_idx]
    name_col = next((j for j, h in enumerate(header) if "Name" in h), None)
    inst_col = next((j for j, h in enumerate(header) if "Instances" in h or "Calls" in h), None)
    time_col = next((j for j, h in enumerate(header) if "Total Time" in h), None)
    if name_col is None or inst_col is None or time_col is None:
        raise ValueError(f"Required columns missing in {stats_file}")

    data = []
    for row in rows[header_idx + 1:]:
        if not row or all(cell.strip() == "" for cell in row):
            continue
        # 碰到第二张表（cuda_api_sum）的表头就停，避免把 CUDA API 调用当成 kernel。
        if any(cell.strip() == "Name" for cell in row):
            break
        if any("CUDA API" in cell or cell.startswith("---") for cell in row):
            break
        try:
            name = row[name_col].strip()
            calls = int(row[inst_col].strip().replace(',', ''))
            total_ns = float(row[time_col].strip().replace(',', ''))
            data.append((name, calls, total_ns))
        except (ValueError, IndexError):
            continue
    if not data:
        raise ValueError(f"No kernel data extracted from {stats_file}")
    df = pd.DataFrame(data, columns=["kernel_name", "calls", "total_time_ns"])
    df = df.sort_values("total_time_ns", ascending=False)
    return df

def generate_trace_summary():
    all_rows = []
    for stats_file in sorted(PROFILE_DIR.glob("*_stats.csv")):
        match = re.match(r"run_(small|medium|large|xl)_ctx(\d+)_stats", stats_file.stem)
        if not match:
            continue
        model = match.group(1)
        ctx = int(match.group(2))
        print(f"Processing {stats_file.name} ...")
        df = parse_kernel_stats(stats_file)
        for _, row in df.head(5).iterrows():
            kernel_name = row["kernel_name"]
            calls = int(row["calls"])
            total_time_us = row["total_time_ns"] / 1000.0
            phase = infer_phase(kernel_name)
            all_rows.append({
                "model_size": model,
                "context_length": ctx,
                "kernel_name": kernel_name,
                "calls": calls,
                "total_time_us": round(total_time_us, 2),
                "phase": phase
            })
    result_df = pd.DataFrame(all_rows, columns=["model_size", "context_length", "kernel_name", "calls", "total_time_us", "phase"])
    result_df.to_csv(OUTPUT_TRACE_SUMMARY, index=False)
    print(f"trace_summary.csv written to {OUTPUT_TRACE_SUMMARY}")

if __name__ == "__main__":
    generate_metadata()
    generate_trace_summary()