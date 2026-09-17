"""为性能 CSV 计算仅基于等价成功 eager 行的 speedup。

题面要求：只有 implementation 之外的所有条件相同且两行都成功时才能计算 speedup；
不得跨 GPU、跨 shape、跨 dtype、跨 causal 设置或使用 OOM 行计算。
因此匹配键取 (sequence_length, head_dim, phase)，并要求 dtype / is_causal 一致、
且参照行的 status == "ok"。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MATCH_KEYS = ("sequence_length", "head_dim", "phase")
EQUIVALENCE_KEYS = ("dtype", "is_causal")


def key_of(record: dict) -> tuple:
    return tuple(record.get(name, "") for name in MATCH_KEYS)


def equivalent(left: dict, right: dict) -> bool:
    return all((left.get(name) or "") == (right.get(name) or "") for name in EQUIVALENCE_KEYS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.exists() or args.input.stat().st_size == 0:
        print(f"{args.input} 不存在或为空，跳过")
        return

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        rows = list(reader)

    eager = {}
    for record in rows:
        if record.get("implementation") == "eager" and record.get("status") == "ok" and record.get("p50_ms"):
            eager[key_of(record)] = record

    computed = 0
    no_reference = set()
    for record in rows:
        reference = eager.get(key_of(record))
        if (reference is not None and record.get("status") == "ok"
                and record.get("p50_ms") and equivalent(record, reference)):
            record["speedup_vs_eager"] = f"{float(reference['p50_ms']) / float(record['p50_ms']):.6f}"
            computed += 1
        else:
            record["speedup_vs_eager"] = ""
            if record.get("status") == "ok" and reference is None:
                no_reference.add(key_of(record))

    with args.input.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    ok = sum(1 for record in rows if record.get("status") == "ok")
    print(f"{args.input}: 共 {total} 行，status=ok 的 {ok} 行，其中 {computed} 行算出了 speedup")
    if no_reference:
        print(f"  提示: 以下 (seq, head_dim, phase) 没有成功的 eager 参照行，speedup 留空：")
        for item in sorted(no_reference):
            print(f"    {item}")


if __name__ == "__main__":
    main()
