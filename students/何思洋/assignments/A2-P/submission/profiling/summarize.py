from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight summaries and SVG charts for A2-P.")
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--profile-json", type=Path, action="append", default=[])
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path)
    return parser.parse_args()


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def simple_bar_svg(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 760, 420
    left, top, bottom = 90, 60, 70
    plot_w, plot_h = width - left - 30, height - top - bottom
    max_value = max(values) if values else 1.0
    bar_w = plot_w / max(1, len(values)) * 0.62
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<text x="22" y="{top + plot_h/2}" transform="rotate(-90 22 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="13">{ylabel}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for i, (label, value) in enumerate(zip(labels, values, strict=False)):
        x = left + (i + 0.2) * (plot_w / max(1, len(values)))
        h = 0 if max_value == 0 else value / max_value * plot_h
        y = top + plot_h - h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="#2f7f91"/>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.1f}</text>')
        parts.append(f'<text x="{x + bar_w/2:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="11">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def profile_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        for event in payload.get("events", []):
            rows.append(
                {
                    "run": path.stem,
                    "model_size": config.get("model_size"),
                    "context_length": config.get("context_length"),
                    "dtype": config.get("dtype"),
                    "tool": config.get("tool"),
                    "name": event.get("name"),
                    "calls": event.get("calls"),
                    "cpu_time_total_us": event.get("cpu_time_total_us"),
                    "cuda_time_total_us": event.get("cuda_time_total_us"),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    bench = read_csv(args.benchmark)
    ok_bench = [row for row in bench if row.get("status") == "ok" and row.get("mean_ms")]
    if ok_bench:
        simple_bar_svg(
            args.assets_dir / "benchmark_modes.svg",
            "A2-P Benchmark Latency",
            [row["mode"] + "\\nwu" + row["warmup"] for row in ok_bench[:8]],
            [float(row["mean_ms"]) for row in ok_bench[:8]],
            "mean ms",
        )
    memory = read_csv(args.memory)
    ok_mem = [row for row in memory if row.get("peak_reserved_mib")]
    if ok_mem:
        simple_bar_svg(
            args.assets_dir / "memory_peaks.svg",
            "A2-P Peak Reserved Memory",
            [row["model_size"] + "-" + row["context_length"] + "-" + row["mode"] for row in ok_mem[:8]],
            [float(row["peak_reserved_mib"]) for row in ok_mem[:8]],
            "MiB",
        )
    rows = profile_rows(args.profile_json)
    if rows and args.profile_summary:
        write_csv(args.profile_summary, rows)
        top = rows[:8]
        simple_bar_svg(
            args.assets_dir / "profile_top_events.svg",
            "A2-P Profile Top CUDA Events",
            [str(row["name"])[:18] for row in top],
            [float(row["cuda_time_total_us"] or 0.0) / 1000.0 for row in top],
            "CUDA ms",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
