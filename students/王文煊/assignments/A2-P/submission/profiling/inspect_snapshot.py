"""Inspect a memory snapshot: largest allocations and their stack origins."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

snap_path = Path(sys.argv[1])
with open(snap_path, "rb") as f:
    snap = pickle.load(f)

segs = snap.get("segments", [])
blocks = []
for seg in segs:
    for b in seg.get("blocks", []):
        if b.get("state") == "active_allocated":
            blocks.append(b)
blocks.sort(key=lambda b: -b["size"])
out = []
for b in blocks[:15]:
    frames = []
    for fr in b.get("frames", [])[:6]:
        frames.append(f"{fr.get('filename','?').split('/')[-1]}:{fr.get('line','?')} {fr.get('name','?')}")
    out.append({"size_mib": round(b["size"] / 2**20, 2), "frames": frames})
print(json.dumps(out[:8], indent=1))
total = sum(b["size"] for b in blocks)
print("total active:", round(total / 2**30, 3), "GiB; blocks:", len(blocks))
