
import sqlite3
import argparse
from pathlib import Path


def analyze_nsys_trace(sqlite_path):
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()

    # 1. NVTX 区间
    cursor.execute("""
        SELECT text, start, end FROM NVTX_EVENTS
        WHERE text IN ('profile/warmup','profile/measure','forward','backward','optimizer')
        ORDER BY start
    """)
    nvtx_rows = cursor.fetchall()

    # 2. 所有内存事件
    cursor.execute("""
        SELECT start, bytes, memoryOperationType
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        ORDER BY start
    """)
    events = cursor.fetchall()

    # 3. 重建时间线
    timeline = []  # (time_ns, total_bytes)
    current = 0
    peak = 0
    peak_time = 0
    for t, b, op in events:
        if op == 0:
            current += b
        else:
            current -= b
        timeline.append((t, current))
        if current > peak:
            peak = current
            peak_time = t

    print("=" * 60)
    print("Nsight Systems Memory Trace 分析")
    print("=" * 60)
    print(f"\n峰值内存: {peak/1024**3:.2f} GiB @ t={peak_time} ns")
    print(f"最终内存: {timeline[-1][1]/1024**3:.2f} GiB")
    print(f"总事件数: {len(events)} (ALLOC: {sum(1 for _,_,op in events if op==0)}, FREE: {sum(1 for _,_,op in events if op==1)})")

    # 4. 测量 step 内是否有任何事件
    measure_start = min((r[1] for r in nvtx_rows if r[0] == 'profile/measure'), default=0)
    measure_events = [e for e in events if e[0] >= measure_start]
    print(f"\nprofile/measure 内的事件数: {len(measure_events)}")
    if len(measure_events) == 0:
        print("→ 确认：PyTorch CUDA allocator 在 warmup 后已缓存所有内存,")
        print("  measurement step 无 cudaMalloc/cudaFree 调用。")

    # 5. NVTX 阶段峰值分析
    free_count = 0
    alloc_count = 0
    for t, b, op in events:
        if op == 0:
            alloc_count += 1
        else:
            free_count += 1

    print(f"\n5 个 warmup step + 1 measurement step 总体情况:")
    print(f"  总 ALLOC: {alloc_count} 次, 总 FREE: {free_count} 次")
    print(f"  Net 分配: {alloc_count - free_count} 次 (对应最终常驻显存)")

    # 6. NVTX 阶段的内存分配
    print(f"\n--- 按 NVTX 阶段分析 ---")
    for nvtx in nvtx_rows:
        name, t_start, t_end = nvtx
        alloc_in_stage = [(t, b) for t, b, op in events if t >= t_start and t < t_end and op == 0]
        free_in_stage = [(t, b) for t, b, op in events if t >= t_start and t < t_end and op == 1]
        total_alloc = sum(b for _, b in alloc_in_stage)
        total_free = sum(b for _, b in free_in_stage)
        if total_alloc > 0 or total_free > 0:
            print(f"  {name:20s}: ALLOC={total_alloc/1024**2:7.1f} MiB ({len(alloc_in_stage):4d}次), FREE={total_free/1024**2:7.1f} MiB ({len(free_in_stage):4d}次), Net={((total_alloc-total_free)/1024**2):7.1f} MiB")

    # 7. 最大的 Top-10 ALLOC 事件
    print(f"\n--- Top 10 最大 ALLOC 事件 ---")
    allocs = sorted([(b, t) for t, b, op in events if op == 0], reverse=True)[:10]
    for i, (b, t) in enumerate(allocs, 1):
        # 查找所属的 NVTX 阶段
        stage = "unknown"
        for nvtx in nvtx_rows:
            _, ts, te = nvtx
            if ts <= t < te:
                stage = nvtx[0]
                break
        print(f"  {i:2d}. {b/1024**2:7.1f} MiB @ {t} ns [{stage}]")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description='分析 nsys memory trace')
    parser.add_argument('sqlite', type=str, help='SQLite 路径')
    args = parser.parse_args()

    p = Path(args.sqlite)
    if not p.exists():
        # 尝试 .nsys-rep 同名的 .sqlite
        p = p.with_suffix('.sqlite')
    if not p.exists():
        print(f"❌ 未找到: {args.sqlite}")
        return

    analyze_nsys_trace(str(p))


if __name__ == '__main__':
    main()
