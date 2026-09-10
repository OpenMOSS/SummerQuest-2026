import argparse
import json
import sys
import time
import pickle
from pathlib import Path

import torch
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from env_utils import public_env

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}
VOCAB_SIZE = 10000

MEM_DIR = Path("results/memory")
RUNS_DIR = MEM_DIR / "runs"
SNAPSHOT_DIR = MEM_DIR / "snapshots"

AMP_DTYPES = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}


def get_model(model_size: str, context_length: int, dtype: torch.dtype) -> nn.Module:
    cfg = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE, context_length=context_length,
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=10000.0,
    )
    return model.to(dtype=dtype).cuda()


def run_step(mode, model, batch, optimizer=None, amp_dtype=None):
    """一个 step；amp_dtype 非 None 时在 autocast 下做 forward/loss（真正的混合精度）。"""
    input_ids, labels = batch

    def forward_loss():
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_ids)
                loss = cross_entropy(logits, labels)
        else:
            logits = model(input_ids)
            loss = cross_entropy(logits, labels)
        return logits, loss

    if mode == "forward":
        if amp_dtype is not None:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
                model(input_ids)
        else:
            with torch.no_grad():
                model(input_ids)
        return
    model.zero_grad(set_to_none=True)
    _logits, loss = forward_loss()
    loss.backward()
    if mode == "train_step":
        optimizer.step()


def residual_stream_bytes(batch_size, context_length, d_model, bytes_per=4):
    return batch_size * context_length * d_model * bytes_per


def traced_delta_mib(snapshot_path) -> float:
    with open(snapshot_path, "rb") as f:
        snap = pickle.load(f)
    events = []
    for tr in snap.get("device_traces", []):
        events += [e for e in tr if e.get("action") in ("alloc", "free_requested")]
    cur, peak = 0, 0
    for e in sorted(events, key=lambda e: e.get("time_us", 0)):
        if e["action"] == "alloc":
            cur += e["size"]
        elif e["action"] == "free_requested":
            cur -= e["size"]
        peak = max(peak, cur)
    return peak / 1024 ** 2


def collect(args) -> dict:
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    amp_dtype = AMP_DTYPES[args.amp]
    cfg = MODEL_CONFIGS[args.model_size]
    d_model = cfg["d_model"]

    args.stage = "model_init"
    model = get_model(args.model_size, args.context_length, dtype)
    optimizer = AdamW(model.parameters(), lr=1e-3) if args.mode == "train_step" else None
    input_ids = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (args.batch_size, args.context_length), device=device)
    batch = (input_ids, labels)

    # 预热（不计入峰值统计），让分配器/kernel 选择稳定
    args.stage = "warmup"
    for _ in range(args.warmup):
        run_step(args.mode, model, batch, optimizer, amp_dtype)
        torch.cuda.synchronize()

    # 只对测量段统计峰值并记录内存历史
    args.stage = "measure"
    torch.cuda.reset_peak_memory_stats()
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    hist_kwargs = {"max_entries": args.max_entries}
    if args.context_all:
        hist_kwargs["context"] = "all"
    if args.stacks_all:
        hist_kwargs["stacks"] = "all"
    try:
        torch.cuda.memory._record_memory_history(**hist_kwargs)
    except TypeError:
        torch.cuda.memory._record_memory_history(max_entries=args.max_entries)
        print("warn: context/stacks not supported by this torch; recorded with defaults",
              file=sys.stderr)

    t0 = time.perf_counter()
    run_step(args.mode, model, batch, optimizer, amp_dtype)
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0

    args.stage = "snapshot"
    snap_path = SNAPSHOT_DIR / f"{args.tag}_snapshot.pickle"
    torch.cuda.memory._dump_snapshot(str(snap_path))
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_active = torch.cuda.max_memory_allocated() / 1024 ** 2
    peak_reserved = torch.cuda.max_memory_reserved() / 1024 ** 2
    delta = traced_delta_mib(str(snap_path))
    residual_mib = residual_stream_bytes(args.batch_size, args.context_length, d_model) / 1024 ** 2

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag,
        "model_size": args.model_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "amp": args.amp,
        "mode": args.mode,
        "warmup": args.warmup,
        "precision_profile": ("mixed-precision (autocast %s)" % args.amp) if amp_dtype is not None
                             else ("pure %s weights" % args.dtype),
        "command": ("python profiling/memory_profiler.py "
                    f"--model-size {args.model_size} --context-length {args.context_length} "
                    f"--batch-size {args.batch_size} --dtype {args.dtype} --amp {args.amp} "
                    f"--mode {args.mode} --warmup {args.warmup} --tag {args.tag}"),
        "memory_history": {
            "context": "all" if args.context_all else "default",
            "stacks": "all" if args.stacks_all else "default",
        },
        "wall_time_s": wall_s,
        "peak_active_mib": round(peak_active, 3),
        "peak_reserved_mib": round(peak_reserved, 3),
        "traced_delta_mib": round(delta, 3),
        "residual_stream_theory_mib": round(residual_mib, 3),
        "snapshot_file": str(snap_path),
        "env": public_env(),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{args.tag}.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(json.dumps(record, indent=2, default=str))
    return record


def write_failure(args, exc, stage="unknown") -> None:
    """Record a failed config (e.g. OOM) so finalize() still reports it honestly."""
    peak_active = peak_reserved = None
    try:
        peak_active = max(0, torch.cuda.max_memory_allocated()) / 1024 ** 2
        peak_reserved = max(0, torch.cuda.max_memory_reserved()) / 1024 ** 2
    except Exception:
        pass
    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tag": args.tag,
        "model_size": args.model_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "amp": getattr(args, "amp", "none"),
        "mode": args.mode,
        "warmup": args.warmup,
        "status": "failed",
        "stage": stage,
        "exception": f"{type(exc).__module__}.{type(exc).__name__}",
        "reason": str(exc),
        "peak_active_mib": round(peak_active, 3) if peak_active is not None else None,
        "peak_reserved_mib": round(peak_reserved, 3) if peak_reserved is not None else None,
        "env": public_env(),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    (RUNS_DIR / f"{args.tag}.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
    (MEM_DIR / "failures.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open(MEM_DIR / "failures.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    print(f"FAILED {args.tag} (stage={stage}): {type(exc).__name__}: {exc}", file=sys.stderr)


def finalize():
    import csv
    rows = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    out_csv = MEM_DIR / "peaks.csv"
    fieldnames = [
        "model_size", "context_length", "batch_size", "dtype", "amp", "mode",
        "peak_active_mib", "peak_reserved_mib", "traced_delta_mib",
        "residual_stream_theory_mib", "wall_time_s", "snapshot_file"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    meta = MEM_DIR / "run_metadata.json"
    meta.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"peaks.csv -> {out_csv}; run_metadata.json -> {meta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-size", choices=MODEL_CONFIGS.keys())
    ap.add_argument("--context-length", type=int)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"],
                    help="把模型参数整体转成该精度（纯低精度，不是混合精度）")
    ap.add_argument("--amp", default="none", choices=["none", "bf16", "fp16"],
                    help="真正的混合精度：参数保持 fp32，forward/loss 在 autocast 下做")
    ap.add_argument("--mode", choices=["forward", "train_step"])
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag")
    ap.add_argument("--max-entries", type=int, default=1000000)
    ap.add_argument("--context-all", action="store_true",
                    help="record memory history with context='all'")
    ap.add_argument("--stacks-all", action="store_true",
                    help="record C+++Python stack for every allocation (memory_viz 逐层归因)")
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()

    if args.finalize:
        finalize()
        return
    missing = [a for a in ("model_size", "context_length", "mode", "tag")
               if not getattr(args, a)]
    if missing:
        raise SystemExit("--finalize 之外需要提供: " + ", ".join("--" + a.replace("_", "-") for a in missing))
    if not torch.cuda.is_available():
        exc = RuntimeError(
            "CUDA not available (no GPU allocated; run via `srun --gres=gpu:1` and check "
            "`python -c \"import torch;print(torch.cuda.is_available())\"` first).")
        write_failure(args, exc, "cuda_check")
        sys.exit(1)
    try:
        collect(args)
    except Exception as exc:  # 例如 OOM：如实记录失败配置后以非零码退出
        write_failure(args, exc, getattr(args, "stage", "unknown"))
        sys.exit(1)


if __name__ == "__main__":
    main()
