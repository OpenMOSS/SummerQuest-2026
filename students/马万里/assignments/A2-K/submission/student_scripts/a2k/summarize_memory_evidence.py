from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ALLOCATOR_LIMIT_MIB = 23_552
HARD_LIMIT_MIB = 24_576
RTX_4090_TOTAL_MIB = 24_106.0

SOURCES: tuple[tuple[str, str, tuple[str, str]], ...] = (
    ("local_results/a2k/checkpointing.csv", "任务一 activation checkpointing 矩阵",
     ("peak_allocated_mib", "peak_reserved_mib")),
    ("local_results/a2k/task5/flash_benchmark.csv", "任务五 FlashAttention 性能矩阵",
     ("peak_allocated_mib", "peak_reserved_mib")),
    ("local_results/a2k/attention_baseline.csv", "任务二 显式 PyTorch attention 基线",
     ("peak_allocated_mib", "peak_reserved_mib")),
)


def resolve_allocator_fraction(root: Path) -> float | None:
    metadata = root / "local_results/a2k/task5/run_metadata.json"
    if metadata.is_file():
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        limit_mib = payload.get("allocator_limit_mib")
        total_mib = payload.get("total_memory_mib") or payload.get("cuda_reported_total_mib")
        if limit_mib and total_mib:
            return min(1.0, float(limit_mib) / float(total_mib))
    try:
        import torch

        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            return min(1.0, ALLOCATOR_LIMIT_MIB * 2**20 / total_bytes)
    except Exception:
        pass
    return min(1.0, ALLOCATOR_LIMIT_MIB / RTX_4090_TOTAL_MIB)


def to_float(value) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def scan_source(path: Path, label: str, columns: tuple[str, str], display: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        return {"source": display, "label": label, "present": False,
                "successful_rows": 0, "peak_allocated_mib": None, "peak_reserved_mib": None}
    allocated, reserved, ok_rows = [], [], 0
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if (record.get("status") or "ok") != "ok":
                continue
            ok_rows += 1
            value_a, value_r = to_float(record.get(columns[0])), to_float(record.get(columns[1]))
            if value_a is not None:
                allocated.append(value_a)
            if value_r is not None:
                reserved.append(value_r)
    return {
        "source": display,
        "label": label,
        "present": True,
        "successful_rows": ok_rows,
        "peak_allocated_mib": round(max(allocated), 3) if allocated else None,
        "peak_reserved_mib": round(max(reserved), 3) if reserved else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="输出路径，通常为 local_results/a2k/memory_evidence.json")
    parser.add_argument("--repo", default=".", help="仓库根目录")
    args = parser.parse_args()
    root = Path(args.repo).resolve()

    per_source = [
        scan_source(root / relative, label, columns, relative)
        for relative, label, columns in SOURCES
    ]
    present = [item for item in per_source if item["present"]]

    allocated = [item["peak_allocated_mib"] for item in present if item["peak_allocated_mib"] is not None]
    reserved = [item["peak_reserved_mib"] for item in present if item["peak_reserved_mib"] is not None]
    peak_allocated = max(allocated) if allocated else 0.0
    peak_reserved = max(reserved) if reserved else 0.0

    fraction = resolve_allocator_fraction(root)
    payload = {
        "allocator": {
            "allocator_fraction": fraction,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": round(peak_allocated, 3),
        "pytorch_peak_reserved_mib": round(peak_reserved, 3),
        "within_24gib": bool(peak_reserved <= ALLOCATOR_LIMIT_MIB),
        "measured_scope": "A2-K 全部正式进程（任务一 checkpointing + 任务五性能矩阵等）",
        "successful_configurations": sum(item["successful_rows"] for item in present),
        "sources": per_source,
    }
    if not present:
        payload["status"] = "no_data"
        payload["note"] = "尚未找到任何正式结果 CSV；请先跑对应任务的矩阵脚本"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"已写入 {args.output}")
    for item in per_source:
        if item["present"]:
            print(f"  {item['label']}: ok 行 {item['successful_rows']}，"
                  f"peak_alloc={item['peak_allocated_mib']} MiB，peak_reserved={item['peak_reserved_mib']} MiB")
        else:
            print(f"  {item['label']}: 未找到（{item['source']}），跳过")
    print(f"  合计最高：peak_allocated={payload['pytorch_peak_allocated_mib']} MiB，"
          f"peak_reserved={payload['pytorch_peak_reserved_mib']} MiB，"
          f"within_24gib={payload['within_24gib']}")


if __name__ == "__main__":
    main()
