from __future__ import annotations

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from cs336_basics.generation import generate_text
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer_experiments import load_tokenizer_artifact


@dataclass(frozen=True)
class ModelGenerationSpec:
    name: str
    config_path: str
    checkpoint_path: str
    tokenizer_path: str
    prompt: str


@dataclass(frozen=True)
class SamplingSetting:
    name: str
    temperature: float
    top_p: float


MODEL_SPECS = {
    "tinystories": ModelGenerationSpec(
        name="tinystories",
        config_path="configs/tinystories_lr6e4.json",
        checkpoint_path="runs/lr_sweep/tinystories_lr6e4/checkpoint_final.pt",
        tokenizer_path="artifacts/tinystories_10k.json",
        prompt="Once upon a time, there was a little girl named Lily who",
    ),
    "owt": ModelGenerationSpec(
        name="owt",
        config_path="configs/owt_baseline.json",
        checkpoint_path="runs/owt_baseline/checkpoint_final.pt",
        tokenizer_path="artifacts/owt_32k.json",
        prompt="The rapid development of artificial intelligence has",
    ),
}


SAMPLING_SETTINGS = (
    SamplingSetting(name="baseline", temperature=0.8, top_p=0.95),
    SamplingSetting(name="low_temperature", temperature=0.5, top_p=0.95),
    SamplingSetting(name="low_top_p", temperature=0.8, top_p=0.8),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate final TinyStories and OWT samples.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument("--output-dir", default="generations")
    parser.add_argument("--tinystories-prompt")
    parser.add_argument("--owt-prompt")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json_atomically(output_path: Path, value: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_text_atomically(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _resolve_eos_token_id(tokenizer: Any, eos_token: str = "<|endoftext|>") -> int:
    eos_bytes = eos_token.encode("utf-8")
    eos_token_id = next(
        (token_id for token_id, token_bytes in tokenizer.vocab.items() if token_bytes == eos_bytes),
        None,
    )
    if eos_token_id is None:
        raise ValueError(f"EOS token is not present in the tokenizer vocabulary: {eos_token!r}")
    return eos_token_id


def _load_model(spec: ModelGenerationSpec, device: torch.device) -> tuple[TransformerLM, Any, dict[str, Any]]:
    config_path = Path(spec.config_path)
    checkpoint_path = Path(spec.checkpoint_path)
    tokenizer_path = Path(spec.tokenizer_path)
    for path in (config_path, checkpoint_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    configuration = _load_json(config_path)
    model_config = configuration["model"]
    model = TransformerLM(**model_config, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(model_state)
    model.eval()

    tokenizer = load_tokenizer_artifact(tokenizer_path)
    if len(tokenizer.vocab) != model_config["vocab_size"]:
        raise ValueError(
            f"Tokenizer vocab_size={len(tokenizer.vocab)} does not match "
            f"model vocab_size={model_config['vocab_size']} for {spec.name}."
        )
    return model, tokenizer, configuration


def _prompt_for_model(spec: ModelGenerationSpec, args: argparse.Namespace) -> str:
    override = args.tinystories_prompt if spec.name == "tinystories" else args.owt_prompt
    return spec.prompt if override is None else override


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_new_tokens < 256:
        raise ValueError("The assignment requires max_new_tokens >= 256.")
    if len(set(args.models)) != len(args.models):
        raise ValueError("models must not contain duplicates.")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    experiment_summary: dict[str, object] = {
        "experiment_name": "final_text_generation",
        "device": str(device),
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "sampling_settings": [asdict(setting) for setting in SAMPLING_SETTINGS],
        "samples": [],
    }

    for model_name in args.models:
        spec = MODEL_SPECS[model_name]
        prompt = _prompt_for_model(spec, args)
        if not prompt:
            raise ValueError(f"Prompt for {model_name} must be non-empty.")

        print(f"Loading model={model_name} from {spec.checkpoint_path}", flush=True)
        model, tokenizer, configuration = _load_model(spec, device)
        eos_token_id = _resolve_eos_token_id(tokenizer)
        prompt_token_ids = tokenizer.encode(prompt)

        for setting in SAMPLING_SETTINGS:
            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed)
            start_time = time.perf_counter()
            result = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_token_id,
                temperature=setting.temperature,
                top_p=setting.top_p,
                device=device,
                generator=generator,
            )
            elapsed_seconds = time.perf_counter() - start_time
            stopped_on_eos = bool(result.generated_token_ids and result.generated_token_ids[-1] == eos_token_id)
            stop_reason = "eos" if stopped_on_eos else "max_new_tokens"
            generated_text = result.text[len(prompt) :] if result.text.startswith(prompt) else result.text
            sample_name = f"{model_name}_{setting.name}"
            sample: dict[str, object] = {
                "sample_name": sample_name,
                "model": model_name,
                "config_path": spec.config_path,
                "checkpoint_path": spec.checkpoint_path,
                "tokenizer_path": spec.tokenizer_path,
                "model_config": configuration["model"],
                "prompt": prompt,
                "prompt_token_count": len(prompt_token_ids),
                "generated_token_count": len(result.generated_token_ids),
                "total_token_count": len(result.token_ids),
                "temperature": setting.temperature,
                "top_p": setting.top_p,
                "seed": args.seed,
                "eos_token_id": eos_token_id,
                "stop_reason": stop_reason,
                "elapsed_seconds": elapsed_seconds,
                "generated_text": generated_text,
                "full_text": result.text,
            }
            model_output_dir = output_dir / model_name
            _write_json_atomically(model_output_dir / f"{sample_name}.json", sample)
            _write_text_atomically(model_output_dir / f"{sample_name}.txt", result.text)
            samples = experiment_summary["samples"]
            assert isinstance(samples, list)
            samples.append(sample)
            _write_json_atomically(output_dir / "summary.json", experiment_summary)
            print(
                f"Generated {sample_name}: tokens={len(result.generated_token_ids)}, "
                f"stop={stop_reason}, elapsed={elapsed_seconds:.2f}s",
                flush=True,
            )

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_json_atomically(output_dir / "summary.json", experiment_summary)
    print(f"Generation summary: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
