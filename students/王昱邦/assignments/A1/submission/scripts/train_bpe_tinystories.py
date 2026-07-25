from pathlib import Path
import argparse
import time
import pickle
import json
import resource

import multiprocessing
import os

import psutil

from cs336_basics.bpe import train_bpe

VOCAB_SIZE=10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" /"TinyStoriesV2-GPT4-train.txt"
VALID_PATH = PROJECT_ROOT / "data" /"TinyStoriesV2-GPT4-valid.txt"
OWT_TRAIN_PATH = PROJECT_ROOT / "data" / "owt_train.txt"
OWT_VALID_PATH = PROJECT_ROOT / "data" / "owt_valid.txt"

TINYSTORIES_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "tinystories_bpe"
OWT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "owt_bpe"

DATASET_CONFIGS = {
    "valid": (VALID_PATH, TINYSTORIES_OUTPUT_DIR),
    "train": (TRAIN_PATH, TINYSTORIES_OUTPUT_DIR),
    "owt-valid": (OWT_VALID_PATH, OWT_OUTPUT_DIR),
    "owt-train": (OWT_TRAIN_PATH, OWT_OUTPUT_DIR),
}

def monitor_process_tree_memory(
    target_pid: int,
    stop_event,
    ready_event,
    peak_rss_bytes,
    sample_interval_seconds: float=0.05,
) -> None:
    monitor_pid = os.getpid()
    target_process = psutil.Process(target_pid)
    ready_event.set()

    while not stop_event.is_set():
        try:
            processes = [
                target_process,
                *target_process.children(recursive=True)
            ]
        except(
            psutil.NoSuchProcess,
            psutil.AccessDenied,
            psutil.ZombieProcess
        ):
            break

        total_rss_bytes = 0

        for process in processes:
            if process.pid == monitor_pid:
                continue

            try:
                total_rss_bytes+= process.memory_info().rss
            except(
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ):
                continue

        with peak_rss_bytes.get_lock():
            if total_rss_bytes > peak_rss_bytes.value:
                peak_rss_bytes.value=total_rss_bytes

        stop_event.wait(sample_interval_seconds)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 TinyStories 或 OpenWebText byte-level BPE tokenizer"
    )

    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIGS),
        required=True,
        help="选择 TinyStories 或 OpenWebText 的训练集/验证集",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help="词表大小"
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="本次实验的唯一名称；若同名输出目录已存在则拒绝覆盖",
    )

    return parser.parse_args()

def main() -> None:
    args=parse_args()

    input_path, output_dir = DATASET_CONFIGS[args.dataset]

    if not input_path.exists():
        raise FileNotFoundError(f"找不到训练数据:{input_path}")

    target_vocab_size=args.vocab_size
    minimum_vocab_size = 256 + len(SPECIAL_TOKENS)
    if target_vocab_size < minimum_vocab_size:
        raise ValueError(
            f"词表大小不能小于 {minimum_vocab_size}，"
            "因为必须容纳 256 个字节和所有特殊 token"
        )

    run_output_dir = (
        output_dir
        / args.dataset
        / f"vocab_{target_vocab_size}"
        / args.run_name
    )
    if run_output_dir.exists():
        raise FileExistsError(f"实验输出目录已存在，拒绝覆盖：{run_output_dir}")

    print(f"数据集：{args.dataset}")
    print(f"训练文件：{input_path}")
    print(f"词表大小：{target_vocab_size}")
    print(f"特殊token: {SPECIAL_TOKENS}")
    print(f"输出目录：{output_dir}")

    print("开始训练 BPE......")
    # start_time = time.perf_counter()

    # vocab,merges = train_bpe(
    #     input_path=input_path,
    #     vocab_size=target_vocab_size,
    #     special_tokens=SPECIAL_TOKENS,
    # )

    # elapsed_seconds=time.perf_counter()-start_time
    stop_event = multiprocessing.Event()
    ready_event=multiprocessing.Event()
    peak_tree_rss_bytes=multiprocessing.Value("Q",0)

    memory_monitor = multiprocessing.Process(
        target=monitor_process_tree_memory,
        args=(
            os.getpid(),
            stop_event,
            ready_event,
            peak_tree_rss_bytes,
        ),
        daemon=True,
    )
    memory_monitor.start()

    if not ready_event.wait(timeout=5):
        stop_event.set()
        memory_monitor.terminate()
        memory_monitor.join()
        raise RuntimeError("内存监控进程启动失败")

    start_time = time.perf_counter()

    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=target_vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
    finally:
        stop_event.set()
        memory_monitor.join(timeout=5)

        if memory_monitor.is_alive():
            memory_monitor.terminate()
            memory_monitor.join()

    if memory_monitor.exitcode != 0:
        raise RuntimeError(
            "内存监控进程异常退出："
            f"exitcode={memory_monitor.exitcode}"
        )

    elapsed_seconds = time.perf_counter() - start_time

    peak_memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    main_process_peak_memory_mb = peak_memory_kb / 1024

    process_tree_peak_memory_mb = (
        peak_tree_rss_bytes.value / 1024**2
    )

    print("BPE 训练完成")

    longest_token = max(vocab.values(), key=len)

    print(f"最长token:{longest_token!r}")
    print(f"最长token字节数:{len(longest_token)}")

    run_output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path=run_output_dir/"vocab.pkl"
    merges_path=run_output_dir/"merges.pkl"

    with vocab_path.open("wb") as vocab_file:
        pickle.dump(vocab,vocab_file)

    with merges_path.open("wb") as merges_file:
        pickle.dump(merges, merges_file)

    metadata ={
        "dataset": args.dataset,
        "run_name": args.run_name,
        "input_path": str(input_path),
        "target_vocab_size": target_vocab_size,
        "final_vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
        "longest_token_repr": repr(longest_token),
        "longest_token_hex": longest_token.hex(),
        "longest_token_num_bytes": len(longest_token),
        "main_process_peak_memory_mib": main_process_peak_memory_mb,
        "process_tree_peak_memory_mib": process_tree_peak_memory_mb,
        "memory_measurement": "aggregate RSS sampled every 50 ms",
    }

    metadata_path = run_output_dir/"metadata.json"
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata,metadata_file, ensure_ascii=False, indent=2)

    print(f"训练耗时：{elapsed_seconds:.2f} 秒")
    print(
        f"主进程峰值内存："
        f"{main_process_peak_memory_mb:.2f} MiB"
    )
    print(
        f"进程树峰值内存："
        f"{process_tree_peak_memory_mb:.2f} MiB"
    )
    print(f"词表已保存：{vocab_path}")
    print(f"merges 已保存：{merges_path}")
    print(f"实验信息已保存：{metadata_path}")
    print(f"最终词表大小：{len(vocab)}")
    print(f"merge 数量：{len(merges)}")

if __name__ == "__main__":
    main()
