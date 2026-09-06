import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SNAP_DIR = Path("results/memory/snapshots")
RESULT_DIR = Path("results/figures")


def load_events(snapshot_path: Path):
    with open(snapshot_path, "rb") as f:
        snap = pickle.load(f)
    events = []
    for tr in snap.get("device_traces", []):
        events += [e for e in tr if e.get("action") in ("alloc", "free_requested")]
    events.sort(key=lambda e: e.get("time_us", 0))
    return events, snap


def frame_phase(frames):
    names = " | ".join(f.get("name", "") for f in frames)
    for key in ("backward", "cross_entropy", "optimizer"):
        if key in names:
            return key
    for key in ("forward", "linear", "layernorm", "attention", "relu", "embedding"):
        if key in names:
            return key
    return "other"


def caller(frames):
    interesting = [f.get("name", "") for f in frames if f.get("filename", "").endswith(".py")]
    if interesting:
        return interesting[-1]
    return frames[0].get("name", "?") if frames else "?"


def to_mib(b):
    return b / 1024 ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="run tag -> <SNAP_DIR>/<tag>_snapshot.pickle")
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.snapshot:
        snap_path = args.snapshot
    elif args.tag:
        snap_path = SNAP_DIR / f"{args.tag}_snapshot.pickle"
    else:
        raise SystemExit("provide --tag or --snapshot")
    if not snap_path.exists():
        raise SystemExit(f"{snap_path} not found")

    events, snap = load_events(snap_path)
    if not events:
        raise SystemExit(f"{snap_path}: no alloc/free events in device_traces")

    ts, act = [], []
    cur, peak, peak_t = 0, 0, events[0]["time_us"]
    t0 = events[0]["time_us"]
    for e in events:
        t = e["time_us"]
        if e["action"] == "alloc":
            cur += e["size"]
        elif e["action"] == "free_requested":
            cur -= e["size"]
        ts.append((t - t0) / 1000.0)   # ms
        act.append(cur)
        if cur > peak:
            peak, peak_t = cur, t

    # 基线：模型权重等在记录开始前就已分配、不在事件里；用 segments 的 active_size 作为起点，
    # 使时间轴反映“真实在用的总显存”（权重 ~13 GiB + 激活增量），而不是只画事件增量。
    baseline_mib = sum(seg.get("active_size", 0) for seg in snap.get("segments", [])) / 1024 ** 2
    peak_total_mib = baseline_mib + to_mib(peak)

    allocs = [e for e in events if e["action"] == "alloc"]
    allocs.sort(key=lambda e: e["size"], reverse=True)
    print(f"{snap_path.name}: events={len(events)} "
          f"delta_peak={to_mib(peak):.2f} MiB (baseline={baseline_mib:.2f} MiB, "
          f"peak_total={peak_total_mib:.2f} MiB) @ t={(peak_t - t0) / 1000:.2f} ms")
    print("Top single allocations (MiB | source):")
    seen_phases = defaultdict(float)
    for e in allocs[:10]:
        ph = frame_phase(e["frames"])
        seen_phases[ph] += e["size"]
        print(f"  {to_mib(e['size']):9.3f} MiB  phase~{ph:9s} caller={caller(e['frames'])}")
    print("Cumulative alloc bytes by phase ~ " + ", ".join(
        f"{k}={to_mib(v):.1f} MiB" for k, v in sorted(seen_phases.items(), key=lambda kv: -kv[1])))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step([t for t in ts], [baseline_mib + to_mib(a) for a in act], where="post", lw=1.2)
    ax.set_xlabel("time (ms), measured step")
    ax.set_ylabel("active memory (MiB), incl. weights")
    ax.set_title(f"{snap_path.stem}  Active Memory Timeline  ·  peak={peak_total_mib:.2f} MiB "
                 f"(baseline weights={baseline_mib:.2f} MiB)")
    ax.grid(alpha=0.3)
    out = args.out or RESULT_DIR / f"{snap_path.stem.replace('_snapshot', '')}_mem.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
