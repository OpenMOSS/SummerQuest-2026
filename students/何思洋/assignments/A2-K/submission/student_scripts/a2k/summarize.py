from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flash", type=Path, required=True)
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def svg_bar(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    left, top, bottom = 80, 55, 90
    plot_w, plot_h = width - left - 30, height - top - bottom
    max_v = max(values) if values else 1
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values, strict=False)):
        slot = plot_w / max(1, len(values))
        bar_w = slot * 0.6
        x = left + i * slot + slot * 0.2
        h = 0 if max_v == 0 else value / max_v * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#497b5a"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.1f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 24}" transform="rotate(35 {x + bar_w / 2:.1f} {top + plot_h + 24})" font-family="Arial" font-size="10">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = [r for r in read_csv(args.flash) if r.get("status") == "ok" and r.get("p50_ms") and r.get("phase") == "forward"]
    rows = rows[:12]
    if rows:
        svg_bar(
            args.assets_dir / "flash_forward_latency.svg",
            "FlashAttention Forward p50 Latency",
            [f"{r['implementation']}-{r['sequence_length']}-{r['head_dim']}" for r in rows],
            [float(r["p50_ms"]) for r in rows],
        )
    mem = json.loads(args.memory.read_text(encoding="utf-8")) if args.memory.is_file() else {}
    peaks = mem.get("peaks", [])
    if peaks:
        svg_bar(
            args.assets_dir / "a2k_memory_evidence.svg",
            "A2-K Peak Reserved Memory",
            [p["source"] for p in peaks],
            [float(p["peak_reserved_mib"]) for p in peaks],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
