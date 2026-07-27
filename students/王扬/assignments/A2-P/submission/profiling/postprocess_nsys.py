from __future__ import annotations

import argparse
import json
import re
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from profiling.common import ensure_dir

TIME_COL_CANDIDATES = [
    "Total Time (ns)",
    "Total Time (us)",
    "Total Time (ms)",
    "Total Time",
    "Time",
    "Average",
]
CALL_COL_CANDIDATES = ["Calls", "Instances", "Count"]
NAME_COL_CANDIDATES = ["Name", "Operation", "Kernel Name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process nsys stats CSV exports into summary tables.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/summary")
    return parser.parse_args()


def split_csv_sections(text: str) -> list[pd.DataFrame]:
    sections: list[pd.DataFrame] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if len(current) < 2:
            current = []
            return
        block = "\n".join(current).strip()
        current = []
        try:
            df = pd.read_csv(StringIO(block))
        except Exception:
            return
        if not df.empty and len(df.columns) >= 2:
            sections.append(df)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if "," not in line:
            flush()
            continue
        current.append(raw_line)
    flush()
    return sections


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.strip().lower() in lowered:
            return str(lowered[candidate.strip().lower()])
    return None


def parse_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce").fillna(0)


def kernel_kind(name: str) -> str:
    lowered = name.lower()
    if re.search(r"softmax|scaledmaskedsoftmax|masksoftmax", lowered):
        return "softmax"
    if re.search(r"gemm|matmul|mma|wmma|cublas|cutlass", lowered):
        return "matmul"
    if re.search(r"adam|optimizer|multi_tensor_apply|ampere_scudnn", lowered):
        return "optimizer_or_fused"
    return "other"


def pick_best_table(sections: list[pd.DataFrame], expect_kernel: bool) -> pd.DataFrame | None:
    best_df: pd.DataFrame | None = None
    best_score = -1
    for df in sections:
        name_col = find_column(df, NAME_COL_CANDIDATES)
        time_col = find_column(df, TIME_COL_CANDIDATES)
        if not name_col or not time_col:
            continue
        score = len(df)
        if expect_kernel and any("Kernel" in str(col) or str(col) == "Name" for col in df.columns):
            score += 1000
        if not expect_kernel and any("Operation" in str(col) for col in df.columns):
            score += 1000
        if score > best_score:
            best_score = score
            best_df = df.copy()
    return best_df


def normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    name_col = find_column(df, NAME_COL_CANDIDATES)
    time_col = find_column(df, TIME_COL_CANDIDATES)
    calls_col = find_column(df, CALL_COL_CANDIDATES)
    normalized = pd.DataFrame()
    normalized["name"] = df[name_col].astype(str) if name_col else ""
    normalized["time_value"] = parse_numeric(df[time_col]) if time_col else 0
    normalized["calls"] = parse_numeric(df[calls_col]) if calls_col else 0
    normalized = normalized[normalized["name"].str.strip() != ""].copy()
    normalized["kind"] = normalized["name"].map(kernel_kind)
    normalized.sort_values("time_value", ascending=False, inplace=True)
    return normalized


def summarize_kernel_table(path: Path, kernel_df: pd.DataFrame) -> dict[str, Any]:
    top_kernel = kernel_df.iloc[0] if not kernel_df.empty else None
    non_matmul = kernel_df[kernel_df["kind"] != "matmul"]
    top_non_matmul = non_matmul.iloc[0] if not non_matmul.empty else None
    matmul_time = float(kernel_df.loc[kernel_df["kind"] == "matmul", "time_value"].sum())
    softmax_time = float(kernel_df.loc[kernel_df["kind"] == "softmax", "time_value"].sum())
    return {
        "source_stats_file": str(path),
        "top_kernel": None if top_kernel is None else str(top_kernel["name"]),
        "top_kernel_time": None if top_kernel is None else float(top_kernel["time_value"]),
        "top_kernel_calls": None if top_kernel is None else int(top_kernel["calls"]),
        "top_non_matmul_kernel": None if top_non_matmul is None else str(top_non_matmul["name"]),
        "top_non_matmul_time": None if top_non_matmul is None else float(top_non_matmul["time_value"]),
        "top_non_matmul_calls": None if top_non_matmul is None else int(top_non_matmul["calls"]),
        "matmul_time": matmul_time,
        "softmax_time": softmax_time,
        "softmax_to_matmul_ratio": (softmax_time / matmul_time) if matmul_time else None,
    }


def summarize_api_table(path: Path, api_df: pd.DataFrame) -> dict[str, Any]:
    top_api = api_df.iloc[0] if not api_df.empty else None
    return {
        "source_stats_file": str(path),
        "top_cuda_api": None if top_api is None else str(top_api["name"]),
        "top_cuda_api_time": None if top_api is None else float(top_api["time_value"]),
        "top_cuda_api_calls": None if top_api is None else int(top_api["calls"]),
    }


def save_table(df: pd.DataFrame, output_dir: Path, stem: str) -> list[str]:
    if df.empty:
        return []
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    return [str(csv_path), str(md_path)]


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    kernel_summary_rows: list[dict[str, Any]] = []
    api_summary_rows: list[dict[str, Any]] = []
    top5_rows: list[dict[str, Any]] = []

    for path in sorted((results_root / "nsys").glob("*_stats.csv")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        sections = split_csv_sections(text)
        if not sections:
            continue

        kernel_raw = pick_best_table(sections, expect_kernel=True)
        api_raw = pick_best_table(sections, expect_kernel=False)

        if kernel_raw is not None:
            kernel_df = normalize_table(kernel_raw)
            kernel_summary_rows.append(summarize_kernel_table(path, kernel_df))
            for rank, (_, row) in enumerate(kernel_df.head(5).iterrows(), start=1):
                top5_rows.append(
                    {
                        "source_stats_file": str(path),
                        "rank": rank,
                        "kernel_name": str(row["name"]),
                        "kernel_kind": str(row["kind"]),
                        "time_value": float(row["time_value"]),
                        "calls": int(row["calls"]),
                    }
                )

        if api_raw is not None:
            api_df = normalize_table(api_raw)
            api_summary_rows.append(summarize_api_table(path, api_df))

    generated_files: list[str] = []
    generated_files.extend(save_table(pd.DataFrame(kernel_summary_rows), output_dir, "nsys_kernel_summary"))
    generated_files.extend(save_table(pd.DataFrame(api_summary_rows), output_dir, "nsys_api_summary"))
    generated_files.extend(save_table(pd.DataFrame(top5_rows), output_dir, "nsys_top5_kernels"))

    index_payload = {
        "artifact_type": "nsys_postprocess_index",
        "generated_files": generated_files,
        "counts": {
            "kernel_summary": len(kernel_summary_rows),
            "api_summary": len(api_summary_rows),
            "top5_rows": len(top5_rows),
        },
    }
    index_path = output_dir / "nsys_postprocess_index.json"
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved nsys post-processed summaries under {output_dir}")


if __name__ == "__main__":
    main()
