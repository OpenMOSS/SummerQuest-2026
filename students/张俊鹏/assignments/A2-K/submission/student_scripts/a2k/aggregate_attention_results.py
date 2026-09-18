from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "local_results" / "a2k" / "attention"
RESULTS = REPO_ROOT.parent / "SummerQuest-2026" / "students" / "张俊鹏" / "assignments" / "A2-K" / "results"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def tile_metadata(row: dict[str, str]) -> dict[str, str]:
    if row["implementation"] != "triton":
        return {"query_tile": "", "key_tile": "", "num_warps": "", "num_stages": ""}
    # Forward uses 64x64 for D=64 and 32x32 for D=128; all backward kernels
    # use the validated 32x32, one-stage configuration.
    if row["phase"] == "forward" and row["head_dim"] == "64":
        tile, stages = "64x64", "2"
    else:
        tile, stages = "32x32", "1"
    return {"query_tile": tile.split("x")[0], "key_tile": tile.split("x")[1], "num_warps": "4", "num_stages": stages}


def key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["sequence_length"], row["head_dim"], row["dtype"], row["causal"], row["phase"])


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    core: list[dict[str, str]] = []
    for implementation in ("eager", "compiled", "triton"):
        for sequence_length in (512, 2048, 8192):
            for head_dim in (64, 128):
                core.extend(read_rows(RAW / f"core_{implementation}_s{sequence_length}_d{head_dim}.csv"))
    boundary: list[dict[str, str]] = []
    for implementation in ("eager", "triton"):
        for head_dim in (64, 128):
            boundary.extend(read_rows(RAW / f"boundary_{implementation}_s16384_d{head_dim}.csv"))

    eager = {key(row): row for row in core + boundary if row["implementation"] == "eager" and row["status"] == "ok"}
    flash_rows: list[dict[str, str]] = []
    for row in core + boundary:
        result = dict(row)
        reference = eager.get(key(row))
        if row["status"] == "ok" and reference and reference["status"] == "ok":
            result["speedup_vs_eager"] = str(float(reference["p50_ms"]) / float(row["p50_ms"]))
        else:
            result["speedup_vs_eager"] = ""
        result.update(tile_metadata(row))
        result["matrix"] = "boundary_16384" if row in boundary else "core"
        flash_rows.append(result)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / "attention_baseline.csv", [row for row in core if row["implementation"] == "eager"])
    write_csv(RESULTS / "flash_benchmark.csv", flash_rows)
    print(f"attention_baseline_rows={sum(row['implementation'] == 'eager' for row in core)}")
    print(f"flash_benchmark_rows={len(flash_rows)}")


if __name__ == "__main__":
    main()
