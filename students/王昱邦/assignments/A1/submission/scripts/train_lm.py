from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, clip_gradients, get_lr_cosine_schedule


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "lm_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Assignment 1 Transformer language model."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--train-data-path", type=Path, required=True)
    data.add_argument("--valid-data-path", type=Path, required=True)
    data.add_argument(
        "--dataset-dtype",
        choices=("uint16", "uint32", "int32", "int64"),
        default="uint16",
        help="Integer dtype used by both tokenized dataset files.",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--vocab-size", type=int, default=10_000)
    model.add_argument("--context-length", type=int, default=256)
    model.add_argument("--d-model", type=int, default=512)
    model.add_argument("--num-layers", type=int, default=4)
    model.add_argument("--num-heads", type=int, default=16)
    model.add_argument("--d-ff", type=int, default=1344)
    model.add_argument("--rope-theta", type=float, default=10_000.0)
    model.add_argument(
        "--no-rmsnorm",
        action="store_true",
        help="Remove all RMSNorm layers for the 7.3 layer-norm ablation.",
    )
    model.add_argument(
        "--post-norm",
        action="store_true",
        help="Use post-norm Transformer blocks instead of the default pre-norm.",
    )
    model.add_argument(
        "--no-rope",
        action="store_true",
        help="Remove RoPE position information for the 7.3 NoPE ablation.",
    )
    model.add_argument(
        "--ffn-type",
        choices=("swiglu", "silu"),
        default="swiglu",
        help="Feed-forward variant; SiLU is used with d_ff=4*d_model in 7.3.",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument("--batch-size", type=int, default=32)
    optimization.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of micro-batches whose gradients are accumulated before "
            "one optimizer update. The effective batch size is "
            "batch_size * gradient_accumulation_steps."
        ),
    )
    optimization.add_argument("--total-iterations", type=int, default=10_000)
    optimization.add_argument("--max-learning-rate", type=float, default=3e-4)
    optimization.add_argument("--min-learning-rate", type=float, default=3e-5)
    optimization.add_argument("--warmup-iters", type=int, default=100)
    optimization.add_argument(
        "--cosine-cycle-iters",
        type=int,
        default=None,
        help="Defaults to total-iterations.",
    )
    optimization.add_argument("--beta1", type=float, default=0.9)
    optimization.add_argument("--beta2", type=float, default=0.999)
    optimization.add_argument("--epsilon", type=float, default=1e-8)
    optimization.add_argument("--weight-decay", type=float, default=0.01)
    optimization.add_argument("--max-grad-norm", type=float, default=1.0)

    runtime = parser.add_argument_group("runtime and reporting")
    runtime.add_argument("--device", default="cpu")
    runtime.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help="PyTorch float32 matrix-multiplication precision setting.",
    )
    runtime.add_argument("--seed", type=int, default=336)
    runtime.add_argument("--log-interval", type=int, default=10)
    runtime.add_argument("--validation-interval", type=int, default=100)
    runtime.add_argument("--validation-batches", type=int, default=10)
    runtime.add_argument(
        "--validation-batch-size",
        type=int,
        default=None,
        help="Defaults to --batch-size.",
    )
    runtime.add_argument(
        "--validation-seed",
        type=int,
        default=None,
        help=(
            "Use a dedicated seed to evaluate the same validation windows at "
            "every evaluation without advancing the training RNG."
        ),
    )
    runtime.add_argument("--checkpoint-interval", type=int, default=1000)
    runtime.add_argument(
        "--show-progress",
        action="store_true",
        help="Display a tqdm progress bar with percentage and estimated time remaining.",
    )
    runtime.add_argument(
        "--progress-update-interval",
        type=int,
        default=10,
        help="Write progress.json every this many optimizer iterations.",
    )
    runtime.add_argument(
        "--no-save-final-checkpoint",
        action="store_true",
        help="Skip the automatic final checkpoint, useful for short sweeps.",
    )
    runtime.add_argument(
        "--overfit-single-batch",
        action="store_true",
        help="Reuse one sampled training batch for architecture debugging.",
    )
    runtime.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    runtime.add_argument(
        "--run-name",
        required=True,
        help="Unique run name. A new run refuses to overwrite an existing directory.",
    )
    runtime.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path from which to resume.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "batch_size",
        "gradient_accumulation_steps",
        "total_iterations",
        "log_interval",
        "validation_interval",
        "validation_batches",
        "checkpoint_interval",
        "progress_update_interval",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if args.validation_batch_size is not None and args.validation_batch_size <= 0:
        raise ValueError("validation_batch_size must be positive.")

    if args.d_model % args.num_heads != 0:
        raise ValueError("d_model must be divisible by num_heads.")
    if (args.d_model // args.num_heads) % 2 != 0:
        raise ValueError("The per-head dimension must be even for RoPE.")
    if args.rope_theta <= 0:
        raise ValueError("rope_theta must be positive.")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")
    if args.warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if args.max_learning_rate < 0 or args.min_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative.")
    if args.min_learning_rate > args.max_learning_rate:
        raise ValueError("min_learning_rate cannot exceed max_learning_rate.")
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        raise ValueError("AdamW beta values must lie in [0, 1).")
    if args.epsilon < 0:
        raise ValueError("epsilon must be non-negative.")
    if args.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")

    cosine_cycle_iters = (
        args.total_iterations
        if args.cosine_cycle_iters is None
        else args.cosine_cycle_iters
    )
    if cosine_cycle_iters <= args.warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters.")

    if not args.train_data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {args.train_data_path}")
    if not args.valid_data_path.is_file():
        raise FileNotFoundError(f"Validation data not found: {args.valid_data_path}")
    if args.resume_from is not None and not args.resume_from.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot use CUDA on this machine.")


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_wall_time_offset(metrics_path: Path) -> float:
    """Recover cumulative wall time when appending to a resumed run."""
    if not metrics_path.is_file():
        return 0.0

    maximum_wall_time = 0.0
    with metrics_path.open(encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {metrics_path} on line {line_number}."
                ) from exc
            wall_time = record.get("wall_clock_sec")
            if isinstance(wall_time, int | float):
                maximum_wall_time = max(maximum_wall_time, float(wall_time))
    return maximum_wall_time


def finite_perplexity(loss: float) -> float | None:
    """Convert mean cross-entropy to perplexity without emitting infinities."""
    if not math.isfinite(loss) or loss > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        return None
    return math.exp(loss)


def format_optional_float(value: float | None) -> str:
    return "overflow" if value is None else f"{value:.3f}"


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    validation_data: np.ndarray,
    batch_size: int,
    context_length: int,
    validation_batches: int,
    device: torch.device,
    validation_seed: int | None = None,
) -> float:
    """Estimate validation loss without constructing backward graphs."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    validation_rng = (
        np.random.default_rng(validation_seed)
        if validation_seed is not None
        else None
    )

    try:
        for _ in range(validation_batches):
            inputs, targets = get_batch(
                dataset=validation_data,
                batch_size=batch_size,
                context_length=context_length,
                device=device,
                rng=validation_rng,
            )
            logits = model(inputs)
            total_loss += cross_entropy(logits, targets).item()
    finally:
        model.train(was_training)

    return total_loss / validation_batches


def save_numbered_checkpoint(
    checkpoint_dir: Path,
    model: TransformerLM,
    optimizer: AdamW,
    completed_iterations: int,
) -> Path:
    checkpoint_path = checkpoint_dir / f"checkpoint_{completed_iterations:08d}.pt"
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_path}")
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=completed_iterations,
        out=checkpoint_path,
    )
    return checkpoint_path


def train(args: argparse.Namespace) -> None:
    validate_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    run_dir = args.output_dir / args.run_name
    continuing_existing_run = args.resume_from is not None and run_dir.is_dir()
    if run_dir.exists() and not continuing_existing_run:
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=continuing_existing_run)
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.jsonl"
    summary_path = run_dir / "summary.json"
    progress_path = run_dir / "progress.json"
    if not continuing_existing_run:
        with config_path.open("x", encoding="utf-8") as config_file:
            json.dump(serialize_args(args), config_file, ensure_ascii=False, indent=2)

    wall_time_offset = load_wall_time_offset(metrics_path)

    dataset_dtype = np.dtype(args.dataset_dtype)
    train_data = np.memmap(args.train_data_path, dtype=dataset_dtype, mode="r")
    valid_data = np.memmap(args.valid_data_path, dtype=dataset_dtype, mode="r")
    if len(train_data) <= args.context_length:
        raise ValueError("Training data is too short for the requested context length.")
    if len(valid_data) <= args.context_length:
        raise ValueError("Validation data is too short for the requested context length.")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
        dtype=torch.float32,
        use_rmsnorm=not args.no_rmsnorm,
        pre_norm=not args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )

    start_iteration = 0
    if args.resume_from is not None:
        start_iteration = load_checkpoint(
            src=args.resume_from,
            model=model,
            optimizer=optimizer,
        )
    if start_iteration >= args.total_iterations:
        raise ValueError(
            "The checkpoint iteration must be smaller than total_iterations."
        )

    cosine_cycle_iters = (
        args.total_iterations
        if args.cosine_cycle_iters is None
        else args.cosine_cycle_iters
    )
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    tokens_per_iteration = effective_batch_size * args.context_length
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"run directory: {run_dir}")
    print(f"device: {device}")
    print(f"float32 matmul precision: {torch.get_float32_matmul_precision()}")
    print(f"model parameters: {parameter_count:,}")
    print(f"training tokens on disk: {len(train_data):,}")
    print(f"validation tokens on disk: {len(valid_data):,}")
    print(f"starting iteration: {start_iteration}")
    print(f"wall-time offset: {wall_time_offset:.3f} seconds")
    print(f"micro-batch size: {args.batch_size}")
    print(f"gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"effective batch size: {effective_batch_size}")
    print(f"overfit single batch: {args.overfit_single_batch}")

    model.train()
    fixed_training_batch = None
    if args.overfit_single_batch:
        fixed_training_batch = get_batch(
            dataset=train_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

    session_start_time = time.perf_counter()
    interval_start_time = time.perf_counter()
    interval_start_iteration = start_iteration
    most_recent_checkpoint_iteration: int | None = None
    latest_train_loss: float | None = None
    latest_validation_loss: float | None = None
    latest_learning_rate: float | None = None

    append_jsonl(
        metrics_path,
        {
            "type": "session_start",
            "step": start_iteration,
            "iteration": start_iteration,
            "wall_clock_sec": wall_time_offset,
            "tokens_processed": start_iteration * tokens_per_iteration,
            "resumed": args.resume_from is not None,
            "resume_from": (
                str(args.resume_from) if args.resume_from is not None else None
            ),
            "config": serialize_args(args),
        },
    )
    write_json(
        progress_path,
        {
            "status": "running",
            "run_name": args.run_name,
            "step": start_iteration,
            "total_steps": args.total_iterations,
            "percent_complete": (
                100.0 * start_iteration / args.total_iterations
            ),
            "tokens_processed": start_iteration * tokens_per_iteration,
            "total_tokens": args.total_iterations * tokens_per_iteration,
            "elapsed_sec": wall_time_offset,
            "estimated_remaining_sec": None,
            "tokens_per_second": None,
        },
    )
    progress_bar = tqdm(
        total=args.total_iterations,
        initial=start_iteration,
        desc=args.run_name,
        unit="step",
        dynamic_ncols=True,
        disable=not args.show_progress,
    )

    for iteration in range(start_iteration, args.total_iterations):
        learning_rate = get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=args.max_learning_rate,
            min_learning_rate=args.min_learning_rate,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        optimizer.zero_grad()
        accumulated_train_loss = 0.0
        for accumulation_step in range(args.gradient_accumulation_steps):
            if fixed_training_batch is None:
                inputs, targets = get_batch(
                    dataset=train_data,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device=device,
                )
            else:
                inputs, targets = fixed_training_batch

            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                wall_clock_sec = (
                    wall_time_offset + time.perf_counter() - session_start_time
                )
                append_jsonl(
                    metrics_path,
                    {
                        "type": "divergence",
                        "step": iteration,
                        "iteration": iteration,
                        "accumulation_step": accumulation_step,
                        "wall_clock_sec": wall_clock_sec,
                        "tokens_processed": iteration * tokens_per_iteration,
                        "train_loss": None,
                        "loss": None,
                        "lr": learning_rate,
                        "learning_rate": learning_rate,
                        "reason": "non-finite training loss",
                    },
                )
                write_json(
                    summary_path,
                    {
                        "status": "diverged",
                        "run_name": args.run_name,
                        "divergence_step": iteration,
                        "divergence_accumulation_step": accumulation_step,
                        "wall_clock_sec": wall_clock_sec,
                        "train_loss": None,
                        "learning_rate": learning_rate,
                        "reason": "non-finite training loss",
                        "parameter_count": parameter_count,
                        "config": serialize_args(args),
                    },
                )
                raise RuntimeError(
                    "Non-finite training loss at iteration "
                    f"{iteration}, accumulation step {accumulation_step}."
                )

            accumulated_train_loss += loss.item()
            scaled_loss = loss / args.gradient_accumulation_steps
            scaled_loss.backward()

        gradient_norm = clip_gradients(
            model.parameters(),
            max_l2_norm=args.max_grad_norm,
        )
        if not torch.isfinite(gradient_norm):
            wall_clock_sec = (
                wall_time_offset + time.perf_counter() - session_start_time
            )
            append_jsonl(
                metrics_path,
                {
                    "type": "divergence",
                    "step": iteration,
                    "iteration": iteration,
                    "wall_clock_sec": wall_clock_sec,
                    "tokens_processed": iteration * tokens_per_iteration,
                    "train_loss": (
                        accumulated_train_loss
                        / args.gradient_accumulation_steps
                    ),
                    "loss": (
                        accumulated_train_loss
                        / args.gradient_accumulation_steps
                    ),
                    "lr": learning_rate,
                    "learning_rate": learning_rate,
                    "gradient_norm_before_clipping": None,
                    "reason": "non-finite gradient norm",
                },
            )
            write_json(
                summary_path,
                {
                    "status": "diverged",
                    "run_name": args.run_name,
                    "divergence_step": iteration,
                    "wall_clock_sec": wall_clock_sec,
                    "train_loss": (
                        accumulated_train_loss
                        / args.gradient_accumulation_steps
                    ),
                    "learning_rate": learning_rate,
                    "reason": "non-finite gradient norm",
                    "parameter_count": parameter_count,
                    "config": serialize_args(args),
                },
            )
            raise RuntimeError(
                f"Non-finite gradient norm at iteration {iteration}."
            )
        optimizer.step()

        completed_iterations = iteration + 1
        latest_train_loss = (
            accumulated_train_loss / args.gradient_accumulation_steps
        )
        latest_learning_rate = learning_rate
        progress_bar.update(1)

        if (
            completed_iterations % args.progress_update_interval == 0
            or completed_iterations == args.total_iterations
        ):
            session_elapsed_sec = time.perf_counter() - session_start_time
            completed_this_session = completed_iterations - start_iteration
            steps_per_second = completed_this_session / session_elapsed_sec
            remaining_steps = args.total_iterations - completed_iterations
            estimated_remaining_sec = remaining_steps / steps_per_second
            current_tokens_per_second = (
                completed_this_session * tokens_per_iteration / session_elapsed_sec
            )
            write_json(
                progress_path,
                {
                    "status": "running",
                    "run_name": args.run_name,
                    "step": completed_iterations,
                    "total_steps": args.total_iterations,
                    "percent_complete": (
                        100.0 * completed_iterations / args.total_iterations
                    ),
                    "tokens_processed": (
                        completed_iterations * tokens_per_iteration
                    ),
                    "total_tokens": (
                        args.total_iterations * tokens_per_iteration
                    ),
                    "elapsed_sec": wall_time_offset + session_elapsed_sec,
                    "estimated_remaining_sec": estimated_remaining_sec,
                    "tokens_per_second": current_tokens_per_second,
                    "train_loss": latest_train_loss,
                    "validation_loss": latest_validation_loss,
                    "learning_rate": latest_learning_rate,
                },
            )
            progress_bar.set_postfix(
                loss=f"{latest_train_loss:.4f}",
                lr=f"{latest_learning_rate:.2e}",
            )

        if completed_iterations % args.log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - interval_start_time
            iterations_in_interval = completed_iterations - interval_start_iteration
            tokens_in_interval = iterations_in_interval * tokens_per_iteration
            tokens_per_second = tokens_in_interval / elapsed_seconds
            wall_clock_sec = (
                wall_time_offset + time.perf_counter() - session_start_time
            )
            tokens_processed = completed_iterations * tokens_per_iteration
            train_record = {
                "type": "train",
                "step": completed_iterations,
                "iteration": completed_iterations,
                "tokens_processed": tokens_processed,
                "wall_clock_sec": wall_clock_sec,
                "train_loss": latest_train_loss,
                "loss": latest_train_loss,
                "perplexity": finite_perplexity(latest_train_loss),
                "lr": learning_rate,
                "learning_rate": learning_rate,
                "gradient_norm_before_clipping": gradient_norm.item(),
                "tokens_per_second": tokens_per_second,
                "interval_seconds": elapsed_seconds,
            }
            append_jsonl(metrics_path, train_record)
            print(
                f"iteration={completed_iterations} "
                f"train_loss={latest_train_loss:.6f} "
                f"lr={learning_rate:.6g} "
                f"grad_norm={gradient_norm.item():.4f} "
                f"tokens/s={tokens_per_second:,.0f}"
            )
            interval_start_time = time.perf_counter()
            interval_start_iteration = completed_iterations

        if completed_iterations % args.validation_interval == 0:
            latest_validation_loss = evaluate(
                model=model,
                validation_data=valid_data,
                batch_size=(
                    args.batch_size
                    if args.validation_batch_size is None
                    else args.validation_batch_size
                ),
                context_length=args.context_length,
                validation_batches=args.validation_batches,
                device=device,
                validation_seed=args.validation_seed,
            )
            append_jsonl(
                metrics_path,
                {
                    "type": "validation",
                    "step": completed_iterations,
                    "iteration": completed_iterations,
                    "tokens_processed": (
                        completed_iterations * tokens_per_iteration
                    ),
                    "wall_clock_sec": (
                        wall_time_offset
                        + time.perf_counter()
                        - session_start_time
                    ),
                    "val_loss": latest_validation_loss,
                    "loss": latest_validation_loss,
                    "perplexity": finite_perplexity(latest_validation_loss),
                    "lr": learning_rate,
                    "learning_rate": learning_rate,
                },
            )
            print(
                f"iteration={completed_iterations} "
                f"validation_loss={latest_validation_loss:.6f} "
                "validation_ppl="
                f"{format_optional_float(finite_perplexity(latest_validation_loss))}"
            )
            interval_start_time = time.perf_counter()
            interval_start_iteration = completed_iterations

        if completed_iterations % args.checkpoint_interval == 0:
            checkpoint_path = save_numbered_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                completed_iterations=completed_iterations,
            )
            most_recent_checkpoint_iteration = completed_iterations
            print(f"checkpoint saved: {checkpoint_path}")

    if (
        not args.no_save_final_checkpoint
        and most_recent_checkpoint_iteration != args.total_iterations
    ):
        checkpoint_path = save_numbered_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            completed_iterations=args.total_iterations,
        )
        print(f"final checkpoint saved: {checkpoint_path}")

    if args.total_iterations % args.validation_interval != 0:
        latest_validation_loss = evaluate(
            model=model,
            validation_data=valid_data,
            batch_size=(
                args.batch_size
                if args.validation_batch_size is None
                else args.validation_batch_size
            ),
            context_length=args.context_length,
            validation_batches=args.validation_batches,
            device=device,
            validation_seed=args.validation_seed,
        )
        append_jsonl(
            metrics_path,
            {
                "type": "validation",
                "step": args.total_iterations,
                "iteration": args.total_iterations,
                "tokens_processed": (
                    args.total_iterations * tokens_per_iteration
                ),
                "wall_clock_sec": (
                    wall_time_offset + time.perf_counter() - session_start_time
                ),
                "val_loss": latest_validation_loss,
                "loss": latest_validation_loss,
                "perplexity": finite_perplexity(latest_validation_loss),
                "lr": latest_learning_rate,
                "learning_rate": latest_learning_rate,
                "final": True,
            },
        )
        print(
            f"iteration={args.total_iterations} "
            f"final_validation_loss={latest_validation_loss:.6f} "
            "final_validation_ppl="
            f"{format_optional_float(finite_perplexity(latest_validation_loss))}"
        )

    final_wall_clock_sec = (
        wall_time_offset + time.perf_counter() - session_start_time
    )
    progress_bar.close()
    append_jsonl(
        metrics_path,
        {
            "type": "session_end",
            "step": args.total_iterations,
            "iteration": args.total_iterations,
            "wall_clock_sec": final_wall_clock_sec,
            "tokens_processed": (
                args.total_iterations * tokens_per_iteration
            ),
            "status": "completed",
        },
    )
    summary = {
        "status": "completed",
        "run_name": args.run_name,
        "final_step": args.total_iterations,
        "total_tokens_processed": (
            args.total_iterations * tokens_per_iteration
        ),
        "effective_batch_size": effective_batch_size,
        "wall_clock_sec": final_wall_clock_sec,
        "final_train_loss": latest_train_loss,
        "final_validation_loss": latest_validation_loss,
        "final_validation_perplexity": (
            finite_perplexity(latest_validation_loss)
            if latest_validation_loss is not None
            else None
        ),
        "final_learning_rate": latest_learning_rate,
        "parameter_count": parameter_count,
        "overfit_single_batch": args.overfit_single_batch,
        "config": serialize_args(args),
    }
    write_json(summary_path, summary)
    write_json(
        progress_path,
        {
            "status": "completed",
            "run_name": args.run_name,
            "step": args.total_iterations,
            "total_steps": args.total_iterations,
            "percent_complete": 100.0,
            "tokens_processed": args.total_iterations * tokens_per_iteration,
            "total_tokens": args.total_iterations * tokens_per_iteration,
            "elapsed_sec": final_wall_clock_sec,
            "estimated_remaining_sec": 0.0,
            "tokens_per_second": (
                (args.total_iterations - start_iteration)
                * tokens_per_iteration
                / (final_wall_clock_sec - wall_time_offset)
            ),
            "train_loss": latest_train_loss,
            "validation_loss": latest_validation_loss,
            "learning_rate": latest_learning_rate,
        },
    )
    print(f"summary saved: {summary_path}")


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
