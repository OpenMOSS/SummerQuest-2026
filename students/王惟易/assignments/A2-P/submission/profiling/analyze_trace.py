import argparse
import json
from collections import Counter
from pathlib import Path


MEASUREMENT_LABEL = "profile/measure"

PHASE_LABELS = (
    "zero_grad",
    "forward",
    "loss",
    "backward",
    "optimizer",
)

ATTENTION_LABELS = (
    "attention/scores",
    "attention/softmax",
    "attention/value",
)

MATRIX_MULTIPLY_KERNEL_MARKERS = (
    "gemm",
    "gemv",
    "cutlass",
    "cublas",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one measured train step in a torch.profiler trace.")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def duration_events(events: list[dict[str, object]]) -> list[dict[str, object]]:
    return [event for event in events if event.get("ph") == "X" and "ts" in event and "dur" in event]


def is_inside(event: dict[str, object], scope: dict[str, object]) -> bool:
    event_start = float(event["ts"])
    event_end = event_start + float(event["dur"])
    scope_start = float(scope["ts"])
    scope_end = scope_start + float(scope["dur"])
    return event_start >= scope_start and event_end <= scope_end


def events_inside(
    events: list[dict[str, object]],
    scopes: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [event for event in events if any(is_inside(event, scope) for scope in scopes)]


def matching_ranges(
    events: list[dict[str, object]],
    *,
    category: str,
    name: str,
) -> list[dict[str, object]]:
    return [event for event in events if event.get("cat") == category and event.get("name") == name]


def summarize_ranges(ranges: list[dict[str, object]]) -> dict[str, object]:
    durations_ms = [float(event["dur"]) / 1000 for event in ranges]
    return {
        "calls": len(ranges),
        "total_ms": sum(durations_ms),
        "mean_ms": sum(durations_ms) / len(durations_ms) if durations_ms else 0.0,
    }


def summarize_kernels(
    kernels: list[dict[str, object]],
    *,
    top_k: int,
) -> dict[str, object]:
    cumulative_us: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    for kernel in kernels:
        name = str(kernel["name"])
        cumulative_us[name] += float(kernel["dur"])
        calls[name] += 1

    ranked = sorted(
        cumulative_us,
        key=lambda name: cumulative_us[name],
        reverse=True,
    )
    matrix_multiply_names = [name for name in ranked if is_matrix_multiply_kernel(name)]
    other_names = [name for name in ranked if not is_matrix_multiply_kernel(name)]
    total_us = sum(cumulative_us.values())

    def category_summary(names: list[str]) -> dict[str, object]:
        category_us = sum(cumulative_us[name] for name in names)
        return {
            "calls": sum(calls[name] for name in names),
            "cumulative_ms": category_us / 1000,
            "fraction": category_us / total_us if total_us else 0.0,
            "top": [
                {
                    "name": name,
                    "calls": calls[name],
                    "cumulative_ms": cumulative_us[name] / 1000,
                }
                for name in names[:top_k]
            ],
        }

    if kernels:
        first_start_us = min(float(kernel["ts"]) for kernel in kernels)
        last_end_us = max(float(kernel["ts"]) + float(kernel["dur"]) for kernel in kernels)
        span_ms = (last_end_us - first_start_us) / 1000
    else:
        span_ms = 0.0
    return {
        "calls": len(kernels),
        "span_ms": span_ms,
        "cumulative_ms": total_us / 1000,
        "matrix_multiply": category_summary(matrix_multiply_names),
        "other": category_summary(other_names),
        "top": [
            {
                "name": name,
                "calls": calls[name],
                "cumulative_ms": cumulative_us[name] / 1000,
            }
            for name in ranked[:top_k]
        ],
    }


def is_matrix_multiply_kernel(name: str) -> bool:
    normalized_name = name.lower()
    return any(marker in normalized_name for marker in MATRIX_MULTIPLY_KERNEL_MARKERS)


def summarize_gpu_scope(
    events: list[dict[str, object]],
    kernels: list[dict[str, object]],
    ranges: list[dict[str, object]],
    *,
    top_k: int,
) -> dict[str, object]:
    return {
        "gpu_ranges": summarize_ranges(ranges),
        "kernels": summarize_kernels(
            events_inside(kernels, ranges),
            top_k=top_k,
        ),
    }


def event_external_id(event: dict[str, object]) -> int | None:
    arguments = event.get("args")
    if not isinstance(arguments, dict):
        return None
    external_id = arguments.get("External id")
    return int(external_id) if external_id is not None else None


def kernels_launched_by_scope(
    events: list[dict[str, object]],
    kernels: list[dict[str, object]],
    scope: dict[str, object],
) -> list[dict[str, object]]:
    cpu_events = [event for event in events if event.get("pid") == scope.get("pid") and event.get("cat") not in {"kernel", "gpu_user_annotation"} and is_inside(event, scope)]
    external_ids = {external_id for event in cpu_events if (external_id := event_external_id(event)) is not None}
    return [kernel for kernel in kernels if event_external_id(kernel) in external_ids]


def summarize_phase(
    events: list[dict[str, object]],
    kernels: list[dict[str, object]],
    *,
    label: str,
    top_k: int,
) -> dict[str, object]:
    ranges = matching_ranges(
        events,
        category="user_annotation",
        name=label,
    )
    if len(ranges) != 1:
        raise ValueError(f"expected one {label} range, found {len(ranges)}")
    scope = ranges[0]
    phase_kernels = kernels_launched_by_scope(events, kernels, scope)
    return {
        "scope": label,
        "cpu_span_ms": float(scope["dur"]) / 1000,
        "kernels": summarize_kernels(phase_kernels, top_k=top_k),
    }


def analyze_measurement(
    events: list[dict[str, object]],
    *,
    top_k: int,
) -> dict[str, object]:
    measurement_ranges = matching_ranges(
        events,
        category="user_annotation",
        name=MEASUREMENT_LABEL,
    )
    if len(measurement_ranges) != 1:
        raise ValueError(f"expected one {MEASUREMENT_LABEL} range, found {len(measurement_ranges)}")

    scoped_events = events_inside(events, measurement_ranges)
    kernels = [event for event in scoped_events if event.get("cat") == "kernel"]
    logical_calls = Counter(str(event["name"]) for event in scoped_events if event.get("cat") == "user_annotation")

    attention = {}
    for attention_label in ATTENTION_LABELS:
        ranges = matching_ranges(
            scoped_events,
            category="gpu_user_annotation",
            name=attention_label,
        )
        attention[attention_label] = summarize_gpu_scope(
            scoped_events,
            kernels,
            ranges,
            top_k=top_k,
        )

    return {
        "scope": MEASUREMENT_LABEL,
        "cpu_span_ms": float(measurement_ranges[0]["dur"]) / 1000,
        "logical_calls": dict(sorted(logical_calls.items())),
        "kernels": summarize_kernels(kernels, top_k=top_k),
        "phases": {
            label: summarize_phase(
                scoped_events,
                kernels,
                label=label,
                top_k=top_k,
            )
            for label in PHASE_LABELS
        },
        "attention": attention,
    }


def print_scope(name: str, summary: dict[str, object]) -> None:
    gpu_ranges = summary["gpu_ranges"]
    kernels = summary["kernels"]
    top = kernels["top"]
    print(
        f"{name:24} ranges={gpu_ranges['calls']:3} gpu_span_ms={gpu_ranges['total_ms']:9.3f} kernel_calls={kernels['calls']:5} kernel_cumulative_ms={kernels['cumulative_ms']:9.3f}"
    )
    if top:
        top_kernel = top[0]
        print(f"{'top kernel':24} calls={top_kernel['calls']:5} cumulative_ms={top_kernel['cumulative_ms']:9.3f} name={top_kernel['name']}")


def print_profile(name: str, summary: dict[str, object]) -> None:
    kernels = summary["kernels"]
    matrix_multiply = kernels["matrix_multiply"]
    top = kernels["top"]
    print(
        f"{name:24} cpu_span_ms={summary['cpu_span_ms']:9.3f} kernel_span_ms={kernels['span_ms']:9.3f} kernel_calls={kernels['calls']:5} kernel_cumulative_ms={kernels['cumulative_ms']:9.3f} matmul_fraction={matrix_multiply['fraction']:7.2%}"
    )
    if top:
        top_kernel = top[0]
        print(f"{'top kernel':24} calls={top_kernel['calls']:5} cumulative_ms={top_kernel['cumulative_ms']:9.3f} name={top_kernel['name']}")


def main() -> None:
    args = parse_args()
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    events = duration_events(trace["traceEvents"])

    measurement = analyze_measurement(events, top_k=args.top_k)
    result = {
        "trace": args.trace.as_posix(),
        "matrix_multiply_kernel_markers": MATRIX_MULTIPLY_KERNEL_MARKERS,
        "measurement": measurement,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_profile("train_step", measurement)
    for name, summary in measurement["phases"].items():
        print_profile(name, summary)
    print()
    for name, summary in measurement["attention"].items():
        print_scope(name, summary)
    print(f"analysis JSON: {args.output}")


if __name__ == "__main__":
    main()
