from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from profiling.benchmark import (
    add_common_arguments,
    autocast_context,
    build_experiment,
    configure_runtime,
    perform_warmup,
    run_step,
)
from profiling.config import (
    MODEL_CONFIGS,
    base_metadata,
    classify_error,
    environment_metadata,
    make_run_name,
    model_config_dict,
    public_relative_path,
    safe_error_summary,
    utc_now,
    write_json,
)
from profiling.nvtx_ranges import patched_attention_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P CUDA memory profiler")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    snapshot = subparsers.add_parser("snapshot")
    add_common_arguments(snapshot)
    snapshot.set_defaults(steps=1, warmup=1, annotate_attention=True)
    snapshot.add_argument("--output-dir", type=Path, required=True)
    snapshot.add_argument("--max-entries", type=int, default=1_000_000)
    snapshot.add_argument("--requested-model", choices=tuple(MODEL_CONFIGS))
    snapshot.add_argument("--requested-context", type=int)
    snapshot.add_argument("--requested-batch", type=int)
    snapshot.add_argument("--fallback-reason")
    snapshot.add_argument("--fallback-parent-run-id")
    snapshot.add_argument("--fallback-attempt", type=int, default=0)

    saved = subparsers.add_parser("saved-tensors")
    saved.add_argument("--run-id", default="MEM-BLOCK")
    saved.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="xl")
    saved.add_argument("--batch-size", type=int, default=4)
    saved.add_argument("--context-length", type=int, default=128)
    saved.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    saved.add_argument("--seed", type=int, default=42)
    saved.add_argument("--device", default="cuda")
    saved.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _record_memory_history(max_entries: int) -> None:
    recorder = cast(Any, torch.cuda.memory._record_memory_history)
    try:
        recorder(enabled="all", context="all", stacks="python", max_entries=max_entries)
    except TypeError:
        try:
            recorder(max_entries=max_entries)
        except TypeError:
            recorder(True, max_entries=max_entries)


def _stop_memory_history() -> None:
    recorder = cast(Any, torch.cuda.memory._record_memory_history)
    try:
        recorder(enabled=None)
    except TypeError:
        recorder(None)


def _mib(value: int | float | None) -> float | None:
    return round(float(value) / (1024**2), 4) if value is not None else None


def _memory_peaks() -> dict[str, float | None]:
    stats = torch.cuda.memory_stats()
    return {
        "peak_active_mib": _mib(stats.get("active_bytes.all.peak")),
        "peak_allocated_mib": _mib(stats.get("allocated_bytes.all.peak")),
        "peak_reserved_mib": _mib(stats.get("reserved_bytes.all.peak")),
        "ending_allocated_mib": _mib(stats.get("allocated_bytes.all.current")),
        "ending_reserved_mib": _mib(stats.get("reserved_bytes.all.current")),
    }


def _largest_allocator_events(snapshot: dict[str, Any]) -> dict[str, Any]:
    largest_alloc: dict[str, Any] | None = None
    largest_segment: dict[str, Any] | None = None
    for device_trace in snapshot.get("device_traces", []):
        for event in device_trace:
            action = event.get("action")
            size = int(event.get("size", 0) or 0)
            if action == "alloc" and (largest_alloc is None or size > int(largest_alloc.get("size", 0))):
                largest_alloc = event
            elif action == "segment_alloc" and (largest_segment is None or size > int(largest_segment.get("size", 0))):
                largest_segment = event

    def summarize(event: dict[str, Any] | None) -> dict[str, Any] | None:
        if event is None:
            return None
        frames = []
        for frame in event.get("frames", [])[:12]:
            filename = str(frame.get("filename", ""))
            frames.append(
                {
                    "file": public_relative_path(filename) if filename else None,
                    "line": frame.get("line"),
                    "name": frame.get("name"),
                }
            )
        return {"size_bytes": int(event.get("size", 0) or 0), "size_mib": _mib(event.get("size")), "frames": frames}

    return {
        "largest_tensor_allocation": summarize(largest_alloc),
        "largest_allocator_segment": summarize(largest_segment),
    }


def run_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    if args.steps != 1:
        raise ValueError("memory history captures exactly one target step")
    if not args.device.startswith("cuda"):
        raise ValueError("memory snapshot experiments require CUDA")

    run_name = make_run_name(
        f"memory-{args.run_id}",
        args.model_size,
        args.batch_size,
        args.context_length,
        args.mode,
        args.dtype,
        "history",
    )
    output_dir = args.output_dir
    snapshot_path = output_dir / f"{run_name}.snapshot.pickle"
    metadata_path = output_dir / f"{run_name}.metadata.json"
    payload = base_metadata(args.run_id, run_name, "torch.cuda.memory_history")
    payload.update(
        {
            "kind": "memory_profile_result",
            "requested_model": args.requested_model or args.model_size,
            "requested_context": args.requested_context or args.context_length,
            "requested_batch": args.requested_batch or args.batch_size,
            "actual_model": args.model_size,
            "actual_context": args.context_length,
            "actual_batch": args.batch_size,
            "model_config": model_config_dict(args.model_size),
            "mode": args.mode,
            "dtype": args.dtype,
            "warmup_steps": args.warmup,
            "measurement_steps": 1,
            "seed": args.seed,
            "snapshot_file": public_relative_path(snapshot_path),
            "max_entries": args.max_entries,
            "fallback_reason": args.fallback_reason,
            "fallback_parent_run_id": args.fallback_parent_run_id,
            "fallback_attempt": args.fallback_attempt,
        }
    )
    failure_stage = "initialization"
    history_enabled = False
    try:
        state = build_experiment(
            model_size=args.model_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            mode=args.mode,
            dtype=args.dtype,
            seed=args.seed,
            device_name=args.device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        payload.update(
            {
                "parameter_count": state.parameter_count,
                "model_fingerprint": state.model_fingerprint,
                "environment": environment_metadata(torch),
            }
        )

        attention_context = patched_attention_ranges() if args.annotate_attention else nullcontext()
        with attention_context:
            failure_stage = "warmup"
            perform_warmup(
                state,
                args.warmup,
                stage_callback=lambda stage: payload.update(failure_stage=stage),
            )
            torch.cuda.synchronize(state.device)
            torch.cuda.reset_peak_memory_stats(state.device)

            failure_stage = "memory_history_start"
            _record_memory_history(args.max_entries)
            history_enabled = True

            failure_stage = args.mode
            run_step(
                state,
                outer_label="profile/measure",
                synchronize_at_end=True,
                stage_callback=lambda stage: payload.update(failure_stage=stage),
            )

            failure_stage = "snapshot_dump"
            allocator_snapshot = torch.cuda.memory._snapshot()
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            payload.update(_memory_peaks())
            allocator_events = _largest_allocator_events(allocator_snapshot)
            payload.update(allocator_events)
            largest_tensor = allocator_events["largest_tensor_allocation"]
            largest_segment = allocator_events["largest_allocator_segment"]
            payload["largest_allocation_mib"] = largest_tensor["size_mib"] if largest_tensor else None
            payload["largest_segment_allocation_mib"] = largest_segment["size_mib"] if largest_segment else None
            payload["status"] = "success"
            payload["failure_stage"] = None
    except Exception as exc:
        if torch.cuda.is_available():
            try:
                payload.update(_memory_peaks())
            except Exception:
                pass
        payload.update(
            {
                "status": classify_error(exc),
                "failure_stage": payload.get("failure_stage") or failure_stage,
                "error_type": exc.__class__.__name__,
                "error_summary": safe_error_summary(exc),
                "environment": environment_metadata(torch),
            }
        )
        raise
    finally:
        if history_enabled:
            try:
                _stop_memory_history()
            except Exception:
                pass
        payload["finished_at"] = utc_now()
        write_json(metadata_path, payload)
    return payload


def run_saved_tensors(args: argparse.Namespace) -> dict[str, Any]:
    payload = base_metadata(args.run_id, f"saved_tensors_{args.model_size}_ctx{args.context_length}", "autograd_hooks")
    payload.update(
        {
            "kind": "saved_tensor_diagnostic",
            "model_size": args.model_size,
            "model_config": model_config_dict(args.model_size),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "dtype": args.dtype,
            "seed": args.seed,
            "output_file": public_relative_path(args.output),
        }
    )
    failure_stage = "initialization"
    try:
        configure_runtime(args.seed)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("saved-tensor diagnostic requires CUDA")
        model_config = MODEL_CONFIGS[args.model_size]
        with torch.device(device):
            positional_encoder = RotaryEmbedding(
                context_length=args.context_length,
                dim=model_config.d_model // model_config.num_heads,
            )
            block = TransformerBlock(
                d_model=model_config.d_model,
                num_heads=model_config.num_heads,
                d_ff=model_config.d_ff,
                positional_encoder=positional_encoder,
            )
            inputs = torch.randn(
                args.batch_size,
                args.context_length,
                model_config.d_model,
                requires_grad=True,
            )

        parameter_storages = {parameter.untyped_storage().data_ptr() for parameter in block.parameters()}
        storage_ids: dict[int, int] = {}
        records: list[dict[str, Any]] = []

        def pack_hook(tensor: torch.Tensor) -> torch.Tensor:
            storage = tensor.untyped_storage()
            storage_pointer = storage.data_ptr()
            storage_id = storage_ids.setdefault(storage_pointer, len(storage_ids) + 1)
            records.append(
                {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "logical_bytes": tensor.numel() * tensor.element_size(),
                    "storage_bytes": storage.nbytes(),
                    "storage_id": storage_id,
                    "storage_offset": tensor.storage_offset(),
                    "is_parameter_storage": storage_pointer in parameter_storages,
                    "grad_fn": tensor.grad_fn.__class__.__name__ if tensor.grad_fn is not None else None,
                }
            )
            return tensor

        def unpack_hook(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        failure_stage = "forward_backward"
        autocast = autocast_context(args.dtype, device)
        with patched_attention_ranges(), torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            with autocast:
                output = block(inputs)
                loss = output.float().sum()
            loss.backward()
        torch.cuda.synchronize(device)

        grouped: dict[tuple[tuple[int, ...], str, str | None], dict[str, Any]] = defaultdict(lambda: {"calls": 0, "logical_bytes": 0})
        for record in records:
            if record["is_parameter_storage"]:
                continue
            key = (tuple(record["shape"]), record["dtype"], record["grad_fn"])
            target = grouped[key]
            target.update(
                {
                    "shape": list(key[0]),
                    "dtype": key[1],
                    "grad_fn": key[2],
                }
            )
            target["calls"] += 1
            target["logical_bytes"] += record["logical_bytes"]

        activation_records = [record for record in records if not record["is_parameter_storage"]]
        unique_activation_storages: dict[int, dict[str, Any]] = {}
        for record in activation_records:
            unique_activation_storages.setdefault(record["storage_id"], record)

        groups = sorted(grouped.values(), key=lambda item: -item["logical_bytes"])
        total_logical_saved = sum(record["logical_bytes"] for record in activation_records)
        for group in groups:
            group["logical_mib"] = _mib(group["logical_bytes"])
            group["logical_percentage"] = 100.0 * group["logical_bytes"] / total_logical_saved if total_logical_saved else 0.0

        unique_groups = []
        total_unique_saved = sum(record["storage_bytes"] for record in unique_activation_storages.values())
        for record in sorted(unique_activation_storages.values(), key=lambda item: -item["storage_bytes"]):
            unique_groups.append(
                {
                    "shape": record["shape"],
                    "dtype": record["dtype"],
                    "grad_fn": record["grad_fn"],
                    "storage_mib": _mib(record["storage_bytes"]),
                    "percentage": (100.0 * record["storage_bytes"] / total_unique_saved if total_unique_saved else 0.0),
                }
            )

        parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in block.parameters())
        parameter_gradient_bytes = sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in block.parameters() if parameter.grad is not None)
        missing_parameter_gradients = sum(parameter.grad is None for parameter in block.parameters())
        input_gradient_bytes = inputs.grad.numel() * inputs.grad.element_size() if inputs.grad is not None else 0

        payload.update(
            {
                "status": "success",
                "saved_tensor_count": len(records),
                "parameter_storage_record_count": sum(record["is_parameter_storage"] for record in records),
                "activation_saved_tensor_count": len(activation_records),
                "unique_activation_storage_count": len(unique_activation_storages),
                "logical_activation_saved_bytes": total_logical_saved,
                "logical_activation_saved_mib": _mib(total_logical_saved),
                "unique_activation_saved_bytes": total_unique_saved,
                "unique_activation_saved_mib": _mib(total_unique_saved),
                "top_activation_logical_groups": groups[:20],
                "top_unique_activation_storages": unique_groups[:20],
                "parameter_bytes": parameter_bytes,
                "parameter_mib": _mib(parameter_bytes),
                "parameter_gradient_bytes": parameter_gradient_bytes,
                "parameter_gradient_mib": _mib(parameter_gradient_bytes),
                "missing_parameter_gradients": missing_parameter_gradients,
                "input_gradient_bytes": input_gradient_bytes,
                "input_gradient_mib": _mib(input_gradient_bytes),
                "total_produced_gradient_bytes": parameter_gradient_bytes + input_gradient_bytes,
                "total_produced_gradient_mib": _mib(parameter_gradient_bytes + input_gradient_bytes),
                "environment": environment_metadata(torch),
            }
        )
    except Exception as exc:
        payload.update(
            {
                "status": classify_error(exc),
                "failure_stage": failure_stage,
                "error_type": exc.__class__.__name__,
                "error_summary": safe_error_summary(exc),
                "environment": environment_metadata(torch),
            }
        )
        raise
    finally:
        payload["finished_at"] = utc_now()
        write_json(args.output, payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.experiment == "snapshot":
        run_snapshot(args)
    else:
        run_saved_tensors(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
