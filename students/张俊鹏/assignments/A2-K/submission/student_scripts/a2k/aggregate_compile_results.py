from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = REPO_ROOT / "local_results" / "a2k"
SUBMISSION_RESULTS = REPO_ROOT.parent / "SummerQuest-2026" / "students" / "张俊鹏" / "assignments" / "A2-K" / "results"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    rows: list[dict[str, str]] = []
    for implementation in ("eager", "compiled"):
        for sequence_length, head_dim in ((512, 64), (2048, 128), (8192, 128)):
            source = RAW / "attention" / f"core_{implementation}_s{sequence_length}_d{head_dim}.csv"
            for row in read_rows(source):
                rows.append({"experiment": "explicit_attention", **row})
        for phase in ("forward", "forward_backward", "training_step"):
            source = RAW / "compile" / f"small_{implementation}_{phase}.csv"
            for row in read_rows(source):
                rows.append({"experiment": "small_transformer", **row})

    fields = sorted({field for row in rows for field in row})
    SUBMISSION_RESULTS.mkdir(parents=True, exist_ok=True)
    destination = SUBMISSION_RESULTS / "compile_comparison.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {destination}")


if __name__ == "__main__":
    main()
