import argparse
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PROFILE_DIR = Path("results/profile")
RESULTS_FIG_DIR = Path("results/figures")

STAGE_TEXTS = ["profile/measure", "forward", "backward", "optimizer",
               "scaled_dot_product_attention", "attention/scores",
               "attention/softmax", "attention/value"]

STAGE_BAND_COLOR = {"forward": "#4C72B0", "backward": "#DD8452", "optimizer": "#55A868",
                    "profile/measure": "#cbcbcb"}

FAMILY_COLORS = {
    "matmul":      "#4C72B0",
    "softmax":     "#C44E52",
    "layernorm":   "#8172B2",
    "optimizer":   "#55A868",
    "elementwise": "#DD8452",
    "reduce":      "#937860",
    "embedding":   "#64B5CD",
    "dropout":     "#9B59B6",
    "other":       "#7f7f7f",
}


def classify(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["gemm", "matmul", "cublas", "sgemm", "hgemm", "tgv_"]):
        return "matmul"
    if "softmax" in n:
        return "softmax"
    if "norm" in n:
        return "layernorm"
    if any(k in n for k in ["adam", "momentum"]):
        return "optimizer"
    if any(k in n for k in ["elementwise", "unary", "binary"]):
        return "elementwise"
    if any(k in n for k in ["reduce", "segment", "sum", "mean"]):
        return "reduce"
    if any(k in n for k in ["gather", "embed"]):
        return "embedding"
    if "dropout" in n:
        return "dropout"
    return "other"


def resolve_names(cur, ids):
    """Return {string_id: value} for the given ids."""
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = cur.execute(f"SELECT id, value FROM StringIds WHERE id IN ({marks})", list(ids))
    return {i: v for i, v in rows}


def load(db: Path):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()

    # NVTX stage ranges (some tables store a range as one row with start+end).
    nv = defaultdict(list)
    marks = ",".join("?" * len(STAGE_TEXTS))
    for start, end, text in cur.execute(
        f"SELECT start, end, text FROM NVTX_EVENTS WHERE text IN ({marks})", STAGE_TEXTS):
        if end >= start:
            nv[text].append((start, end))

    if "profile/measure" not in nv:
        raise RuntimeError(
            f"{db.name}: no 'profile/measure' NVTX range. Was this traced with "
            "profile_one_step.py?")
    m0 = min(s for s, e in nv["profile/measure"])
    m1 = max(e for s, e in nv["profile/measure"])
    ms = m0

    # Stage spans within the measure window (first contiguous occurrence each).
    spans = {}
    for text in ["forward", "backward", "optimizer"]:
        inside = [(s, e) for s, e in nv[text] if s >= m0 and e <= m1]
        if inside:
            spans[text] = (min(s for s, _ in inside), max(e for _, e in inside))
    # Optional attention sub-ranges (only for nicer annotation; not required).
    attn = {}
    for text in ["attention/scores", "attention/softmax", "attention/value"]:
        inside = [(s, e) for s, e in nv[text] if s >= m0 and e <= m1]
        if inside:
            attn[text] = (min(s for s, _ in inside), max(e for _, e in inside))

    # Kernels inside the measure window (start >= m0, start < m1).
    ker_rows = list(cur.execute(
        "SELECT start, end, shortName, streamId FROM CUPTI_ACTIVITY_KIND_KERNEL "
        "WHERE start >= ? AND start < ?", (m0, m1)))
    name_ids = {r[2] for r in ker_rows}
    names = resolve_names(cur, list(name_ids))
    streams = sorted({r[3] for r in ker_rows})

    kernels = [{"s": (st - ms) / 1e6, "e": (en - ms) / 1e6,
                "name": names.get(shid, str(shid)), "stream": stid}
               for st, en, shid, stid in ker_rows]
    return {"ms": ms, "t0": (m0 - ms) / 1e6, "t1": (m1 - ms) / 1e6,
            "spans": spans, "attn": attn, "kernels": kernels, "streams": streams,
            "measure_us": (m1 - m0) / 1e3}


def draw(data, name: str, out: Path):
    spans = data["spans"]
    attn = data["attn"]
    kernels = data["kernels"]
    streams = data["streams"]
    x0, x1 = data["t0"], data["t1"]

    stream_row = {s: i for i, s in enumerate(streams)}
    n_streams = max(1, len(streams))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 5.5), sharex=True,
        gridspec_kw={"height_ratios": [1, max(2, n_streams)]})

    # --- top: stage bands (forward/backward/optimizer) ---
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_ylabel("stage")
    for lbl, (s, e) in spans.items():
        ax_top.barh(0.5, (e - s) / 1e6, left=(s - data["ms"]) / 1e6,
                    height=0.9, color=STAGE_BAND_COLOR.get(lbl, "#888888"))
        ax_top.text((s + e) / 2 / 1e6 - data["t0"], 0.5, lbl,
                    ha="center", va="center", color="white", fontsize=10, weight="bold")
    # attention sub-ranges as tick markers on the stage band.
    for lbl, (s, e) in attn.items():
        ax_top.annotate(lbl.split("/")[-1], xy=((s - data["ms"]) / 1e6, 0.12),
                        xytext=((s - data["ms"]) / 1e6, 0.28),
                        ha="center", fontsize=7, color="#555555",
                        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))

    # --- bottom: per-kernel GPU timeline colored by family ---
    ax_bot.set_yticks(range(n_streams))
    ax_bot.set_yticklabels([f"stream {s}" for s in streams] if n_streams > 1 else ["GPU kernel"])
    ax_bot.set_xlabel("time (ms), measure step")
    ax_bot.set_ylabel("CUDA kernel")
    ax_bot.set_ylim(-0.5, n_streams - 0.5)

    by_fam = defaultdict(list)
    for k in kernels:
        by_fam[classify(k["name"])].append(k)
    for fam, ks in by_fam.items():
        color = FAMILY_COLORS.get(fam, FAMILY_COLORS["other"])
        # group per stream for fewer artists
        per_stream = defaultdict(list)
        for k in ks:
            per_stream[k["stream"]].append((k["s"], k["e"] - k["s"]))
        for st, segs in per_stream.items():
            ax_bot.broken_barh(segs, (stream_row[st] - 0.35, 0.7),
                               facecolors=color, edgecolors="none")

    # vertical separators for stage boundaries
    for s, e in spans.values():
        ax_bot.axvline((s - data["ms"]) / 1e6, color="#999999", lw=0.6, ls=":")

    ax_bot.set_xlim(x0, x1)
    handles = [Patch(facecolor=c, label=f) for f, c in FAMILY_COLORS.items() if any(
        classify(k["name"]) == f for k in kernels)]
    ax_bot.legend(handles=handles, loc="upper right", fontsize=8, ncol=3,
                  framealpha=0.9)

    title = (f"{name}  GPU timeline (measure step, "
             f"{data['measure_us']/1000:.2f} ms)  ·  n_kernels={len(kernels)}")
    fig.suptitle(title, fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    except Exception:
        pass
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def print_summary(data, name: str):
    def ms(s, e):
        return (e - s) / 1e6
    print(f"\n[{name}] measure window: {data['measure_us']/1000:.3f} ms, "
          f"kernels in window: {len(data['kernels'])}")
    for lbl in ["forward", "backward", "optimizer"]:
        if lbl in data["spans"]:
            s, e = data["spans"][lbl]
            print(f"  stage {lbl:9s} = {ms(s, e):8.3f} ms  "
                  f"(start {ms(s, data['ms']):8.3f} ms)")
    from collections import Counter
    fam = Counter(classify(k["name"]) for k in data["kernels"])
    print("  kernel families: " + ", ".join(f"{f}: {n}" for f, n in fam.most_common()))
    if data["attn"]:
        parts = ", ".join(f"{k.split('/')[-1]} {ms(s, e):.3f} ms"
                          for k, (s, e) in sorted(data["attn"].items()))
        print(f"  attention sub-steps: {parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True,
                    help="run basename, e.g. run_medium_ctx512 (reads results/profile/<name>.sqlite)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    db = PROFILE_DIR / f"{args.name}.sqlite"
    if not db.exists():
        raise SystemExit(
            f"{db} not found. Create it from the matching report first:\n"
            f"  nsys stats --force-export true results/profile/{args.name}.nsys-rep")
    data = load(db)
    print_summary(data, args.name)
    out = args.out or RESULTS_FIG_DIR / f"{args.name}_timeline.png"
    draw(data, args.name, out)


if __name__ == "__main__":
    main()
