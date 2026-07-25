from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from profiling.benchmark import MODEL_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="A2-P Transformer memory profiling"
    )
    
    parser.add_argument(
        "--model-size",
        choices=["xl", "large"],
        default="xl",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        required=True,
        choices=[128, 1024, 2048],
    )

    parser.add_argument(
        "--mode",
        choices=["forward", "train_step"],
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    return parser.parse_args()


def memory_recording_start() -> bool:
    """
    开启 PyTorch CUDA memory history。

    不同 PyTorch 版本的参数略有差异，
    因此先尝试 enabled='all'，失败后使用默认参数。
    """

    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)

    if recorder is None:
        return False

    try:
        recorder(enabled="all")
        return True
    except TypeError:
        try:
            recorder()
            return True
        except Exception:
            return False
    except Exception:
        return False


def memory_recording_stop() -> None:
    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)

    if recorder is None:
        return

    try:
        recorder(enabled=None)
    except Exception:
        pass


def dump_memory_snapshot(path: Path) -> str | None:
    """
    导出 PyTorch memory snapshot。

    这个文件用于本地 Memory Visualizer，
    不要提交到公开 GitHub。
    """

    dumper = getattr(torch.cuda.memory, "_dump_snapshot", None)

    if dumper is None:
        return "torch.cuda.memory._dump_snapshot is unavailable"

    try:
        dumper(str(path))
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def cuda_memory_stats() -> dict[str, float]:
    torch.cuda.synchronize()

    return {
        "allocated_mib": round(
            torch.cuda.memory_allocated() / 1024**2,
            3,
        ),
        "reserved_mib": round(
            torch.cuda.memory_reserved() / 1024**2,
            3,
        ),
        "max_allocated_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2,
            3,
        ),
        "max_reserved_mib": round(
            torch.cuda.max_memory_reserved() / 1024**2,
            3,
        ),
    }


def build_model(args: argparse.Namespace) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        rope_theta=10_000.0,
    )

    return model.to(
        device="cuda",
        dtype=torch.float32,
    )


def run_step(
    model: BasicsTransformerLM,
    tokens: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    optimizer: AdamW | None,
) -> float | None:
    if mode == "forward":
        model.eval()

        with torch.inference_mode():
            model(tokens)

        return None

    model.train()
    model.zero_grad(set_to_none=True)

    with torch.autograd.profiler.record_function("forward"):
        logits = model(tokens)
        loss = cross_entropy(logits, targets)

    with torch.autograd.profiler.record_function("backward"):
        loss.backward()

    assert optimizer is not None

    with torch.autograd.profiler.record_function("optimizer"):
        optimizer.step()

    return float(loss.detach().float().cpu())


def write_metadata(
    path: Path,
    metadata: dict,
) -> None:
    path.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def write_peaks_csv(
    path: Path,
    metadata: dict,
) -> None:
    fieldnames = [
        "model_size",
        "context_length",
        "batch_size",
        "mode",
        "status",
        "allocated_mib",
        "reserved_mib",
        "max_allocated_mib",
        "max_reserved_mib",
        "loss",
    ]

    row = {
        "model_size": metadata.get("model_size"),
        "context_length": metadata.get("context_length"),
        "batch_size": metadata.get("batch_size"),
        "mode": metadata.get("mode"),
        "status": metadata.get("status"),
        "allocated_mib": metadata.get("memory", {}).get(
            "allocated_mib"
        ),
        "reserved_mib": metadata.get("memory", {}).get(
            "reserved_mib"
        ),
        "max_allocated_mib": metadata.get("memory", {}).get(
            "max_allocated_mib"
        ),
        "max_reserved_mib": metadata.get("memory", {}).get(
            "max_reserved_mib"
        ),
        "loss": metadata.get("loss"),
    }

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [sys.executable, *sys.argv]
        ),
        "model_size": args.model_size,
        "model_config": MODEL_CONFIGS[args.model_size],
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "vocab_size": args.vocab_size,
        "mode": args.mode,
        "warmup": args.warmup,
        "dtype": "float32",
        "status": "started",
        "device": None,
        "torch_version": torch.__version__,
        "memory_history_enabled": False,
    }

    if not torch.cuda.is_available():
        metadata["status"] = "failed"
        metadata["error_type"] = "CUDAUnavailable"
        metadata["error"] = "CUDA is not available"

        write_metadata(
            args.output_dir / "run_metadata.json",
            metadata,
        )
        write_peaks_csv(
            args.output_dir / "peaks.csv",
            metadata,
        )
        return 1

    metadata["device"] = torch.cuda.get_device_name()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = None
    optimizer = None

    try:
        print("Building XL model...")
        model = build_model(args)

        tokens = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            device="cuda",
            dtype=torch.long,
        )

        targets = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            device="cuda",
            dtype=torch.long,
        )

        if args.mode == "train_step":
            optimizer = AdamW(
                model.parameters(),
                lr=1e-3,
            )

        print("Running warmup...")

        for _ in range(args.warmup):
            run_step(
                model=model,
                tokens=tokens,
                targets=targets,
                mode=args.mode,
                optimizer=optimizer,
            )
            torch.cuda.synchronize()

        print("Warmup completed.")

        # 官方要求：warm-up 完成后再开启 memory history
        history_enabled = memory_recording_start()
        metadata["memory_history_enabled"] = history_enabled

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        print("Running one measured step...")

        loss = run_step(
            model=model,
            tokens=tokens,
            targets=targets,
            mode=args.mode,
            optimizer=optimizer,
        )

        torch.cuda.synchronize()

        memory = cuda_memory_stats()

        metadata["status"] = "success"
        metadata["loss"] = loss
        metadata["memory"] = memory

        snapshot_error = dump_memory_snapshot(
            args.output_dir / "memory_snapshot.pickle"
        )

        if snapshot_error is not None:
            metadata["snapshot_error"] = snapshot_error

        print("\n===== Result =====")
        print(json.dumps(metadata, indent=2))

        write_metadata(
            args.output_dir / "run_metadata.json",
            metadata,
        )

        write_peaks_csv(
            args.output_dir / "peaks.csv",
            metadata,
        )

        return 0

    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.synchronize()

        metadata["status"] = "oom"
        metadata["error_type"] = "torch.cuda.OutOfMemoryError"
        metadata["error"] = str(exc)
        metadata["memory"] = cuda_memory_stats()

        print("\n===== OOM =====")
        print(json.dumps(metadata, indent=2))

        write_metadata(
            args.output_dir / "run_metadata.json",
            metadata,
        )

        write_peaks_csv(
            args.output_dir / "peaks.csv",
            metadata,
        )

        return 2

    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        metadata["traceback"] = traceback.format_exc()

        print("\n===== Failed =====")
        print(json.dumps(metadata, indent=2))

        write_metadata(
            args.output_dir / "run_metadata.json",
            metadata,
        )

        write_peaks_csv(
            args.output_dir / "peaks.csv",
            metadata,
        )

        return 1

    finally:
        memory_recording_stop()


if __name__ == "__main__":
    raise SystemExit(main())