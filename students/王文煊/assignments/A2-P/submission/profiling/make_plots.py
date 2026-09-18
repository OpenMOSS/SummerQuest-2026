"""Generate submission figures from local raw results (matplotlib, no screenshots)."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("assets")
DST.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(DST / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    kb = (DST / name).stat().st_size / 1024
    print(name, f"{kb:.0f} KiB")


def load(p):
    with open(p) as f:
        return json.load(f)


# ---------- 1. benchmark 3 modes ----------
modes = ["forward", "forward_backward", "train_step"]
means, stds = [], []
for m in modes:
    d = load(SRC / f"bench/small_bs4_ctx512_fp32_{m}_w5.json")
    means.append(d["mean_s"] * 1e3)
    stds.append(d["stdev_s"] * 1e3)
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.bar(modes, means, yerr=stds, capsize=5, color=["#4c9be0", "#e08b4c", "#68b06b"])
ax.set_ylabel("time per step (ms)")
ax.set_title("small / bs4 / ctx512 / FP32 — mean ± stdev of 10 steps (warmup=5)")
save(fig, "benchmark_modes.png")

# ---------- 2. warmup comparison ----------
ws, wmeans, wstds = [], [], []
for w in [0, 1, 2, 5]:
    d = load(SRC / f"bench/small_bs4_ctx512_fp32_train_step_w{w}.json")
    ws.append(str(w))
    wmeans.append(d["mean_s"] * 1e3)
    wstds.append(d["stdev_s"] * 1e3)
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.bar(ws, wmeans, yerr=wstds, capsize=5, color="#9b6bb0")
ax.set_xlabel("warmup steps")
ax.set_ylabel("train_step mean (ms)")
ax.set_title("Effect of warmup on train_step timing (small/bs4/ctx512/FP32)")
save(fig, "warmup_effect.png")

# ---------- 3. profiler stage breakdown (representative: large ctx512) ----------
rep = load(SRC / "profile/parsed_large_ctx512_bs4_fp32.json")
ss = rep["stage_summary"]
fig, ax = plt.subplots(figsize=(6.5, 3.8))
names = ["forward", "backward", "optimizer",
         "attention/scores", "attention/softmax", "attention/value"]
vals = [ss[n]["cuda_time_total_us"] / 1e3 for n in names]
colors = ["#4c9be0", "#e08b4c", "#68b06b", "#c0504d", "#8064a2", "#f0a030"]
ax.bar(names, vals, color=colors)
ax.set_ylabel("CUDA time in one train_step (ms)")
ax.set_title("torch.profiler stage attribution — large / bs4 / ctx512 / FP32")
plt.xticks(rotation=20)
save(fig, "profile_stages.png")

# ---------- 4. fp32 vs bf16 time & peak memory ----------
labels, t32, t16 = [], [], []
for m in modes:
    labels.append(m)
    t32.append(load(SRC / f"bench/small_bs4_ctx512_fp32_{m}_w5.json")["mean_s"] * 1e3)
    t16.append(load(SRC / f"bench/small_bs4_ctx512_bf16_{m}_w5.json")["mean_s"] * 1e3)
p32 = [load(SRC / f"bench/small_bs4_ctx512_fp32_{m}_w5.json")["peak_memory_bytes"] / 2**30 for m in modes]
p16 = [load(SRC / f"bench/small_bs4_ctx512_bf16_{m}_w5.json")["peak_memory_bytes"] / 2**30 for m in modes]
import numpy as np
xp = np.arange(len(modes))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.5))
a1.bar(xp - 0.18, t32, 0.36, label="FP32")
a1.bar(xp + 0.18, t16, 0.36, label="BF16 autocast")
a1.set_xticks(xp); a1.set_xticklabels(labels, rotation=15)
a1.set_ylabel("mean time (ms)"); a1.legend(); a1.set_title("Time: FP32 vs BF16 (small/bs4/ctx512)")
a2.bar(xp - 0.18, p32, 0.36, label="FP32")
a2.bar(xp + 0.18, p16, 0.36, label="BF16 autocast")
a2.set_xticks(xp); a2.set_xticklabels(labels, rotation=15)
a2.set_ylabel("peak memory (GiB)"); a2.legend(); a2.set_title("Peak memory: FP32 vs BF16")
save(fig, "fp32_vs_bf16.png")

# ---------- 5/6. active memory timelines from snapshots ----------
def timeline(snap_path):
    with open(snap_path, "rb") as f:
        snap = pickle.load(f)
    traces = snap.get("device_traces", [])
    events = []
    for tr in traces:
        for e in tr:
            action = e["action"]
            if action in ("alloc", "free_requested", "free_completed", "segment_alloc", "segment_free"):
                events.append(e)
    events.sort(key=lambda e: e.get("time_us", 0))
    t, mem, cur = [], [], 0
    for e in events:
        a = e["action"]
        if a == "alloc":
            cur += e["size"]
        elif a in ("free_requested", "free_completed"):
            cur -= e["size"]
        t.append(e.get("time_us", 0) / 1e3)
        mem.append(cur / 2**30)
    return t, mem

for ctx, mode, mdl, dt, bs in [(128, "forward", "xl", "fp32", 4),
                               (2048, "forward", "xl", "fp32", 4),
                               (2048, "train_step", "large", "fp32", 1)]:
    p = SRC / f"memory/snapshot_{mdl}_ctx{ctx}_bs{bs}_{mode}_{dt}.pickle"
    if not p.exists():
        print("missing", p)
        continue
    t, mem = timeline(p)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(t, mem, lw=0.8)
    ax.set_xlabel("time (ms, relative)")
    ax.set_ylabel("active memory (GiB)")
    ax.set_title(f"Active memory timeline — {mdl} / bs{bs} / ctx{ctx} / {mode} / {dt.upper()}")
    save(fig, f"mem_timeline_{mdl}_ctx{ctx}_{mode}.png")

print("done")
