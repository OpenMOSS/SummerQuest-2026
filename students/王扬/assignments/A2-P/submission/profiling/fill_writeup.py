from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from profiling.common import ensure_dir, to_repo_relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate writeup-ready markdown snippets from profiling outputs.")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/summary")
    return parser.parse_args()


def load_tsv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t")
    return None if df.empty else df


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return None if df.empty else df


def command_block(df: pd.DataFrame, categories: list[str]) -> str:
    subset = df[df["category"].isin(categories)]
    if subset.empty:
        return "```bash\n# No commands recorded yet.\n```\n"
    lines = subset["command"].astype(str).tolist()
    return "```bash\n" + "\n".join(lines) + "\n```\n"


def artifact_bullets(paths: list[Path]) -> str:
    if not paths:
        return "- 暂无生成文件\n"
    return "".join(f"- `{to_repo_relative(path.resolve())}`\n" for path in paths)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    commands_df = load_tsv(output_dir / "commands.tsv")
    benchmark_df = load_csv(output_dir / "benchmark_summary.csv")
    memory_df = load_csv(output_dir / "memory_summary.csv")
    nsys_df = load_csv(output_dir / "nsys_kernel_summary.csv")
    api_df = load_csv(output_dir / "nsys_api_summary.csv")

    figure_index = results_root / "figures" / "generated_figures.txt"
    figure_paths = []
    if figure_index.exists():
        figure_paths = [Path(line.strip()) for line in figure_index.read_text(encoding="utf-8").splitlines() if line.strip()]

    parts: list[str] = []
    parts.append("# Auto-Generated Writeup Snippets\n")

    parts.append("## Benchmark Raw Commands\n")
    if commands_df is not None:
        parts.append(command_block(commands_df, ["benchmark"]))

    parts.append("## NSYS Commands\n")
    if commands_df is not None:
        parts.append(command_block(commands_df, ["nsys_profile", "nsys_stats", "nsys_memory_trace"]))

    parts.append("## Torch Profiler Command\n")
    if commands_df is not None:
        parts.append(command_block(commands_df, ["torch_profiler"]))

    parts.append("## Memory Snapshot Commands\n")
    if commands_df is not None:
        parts.append(command_block(commands_df, ["memory_snapshot"]))

    parts.append("## Generated Artifact Paths\n")
    benchmark_artifacts = sorted((results_root / "benchmark").rglob("*.json"))
    nsys_artifacts = sorted((results_root / "nsys").glob("*"))
    torch_artifacts = sorted((results_root / "torch").glob("*"))
    memory_artifacts = sorted((results_root / "memory").glob("*"))
    parts.append("### Benchmark\n")
    parts.append(artifact_bullets(benchmark_artifacts))
    parts.append("### NSYS\n")
    parts.append(artifact_bullets(nsys_artifacts))
    parts.append("### Torch Profiler\n")
    parts.append(artifact_bullets(torch_artifacts))
    parts.append("### Memory\n")
    parts.append(artifact_bullets(memory_artifacts))
    parts.append("### Figures\n")
    parts.append(artifact_bullets(figure_paths))

    if benchmark_df is not None:
        parts.append("## Benchmark Summary Table\n")
        parts.append(benchmark_df.to_markdown(index=False) + "\n")

    if nsys_df is not None:
        parts.append("## NSYS Kernel Summary Table\n")
        parts.append(nsys_df.to_markdown(index=False) + "\n")

    if api_df is not None:
        parts.append("## NSYS API Summary Table\n")
        parts.append(api_df.to_markdown(index=False) + "\n")

    if memory_df is not None:
        parts.append("## Memory Summary Table\n")
        parts.append(memory_df.to_markdown(index=False) + "\n")

    skipped_path = output_dir / "skipped_commands.tsv"
    skipped_df = load_tsv(skipped_path)
    if skipped_df is not None:
        parts.append("## Skipped Commands\n")
        parts.append(skipped_df.to_markdown(index=False) + "\n")

    output_path = output_dir / "writeup_snippets.md"
    output_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Saved writeup snippets to {output_path}")


if __name__ == "__main__":
    main()
