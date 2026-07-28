"""Run the pinned official attention tests through a read-only local overlay."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch

from .common import (
    STARTER_COMMIT,
    SUBMISSION_ROOT,
    configure_formal_run,
    public_run_record,
    upsert_json_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _selected_output(captured: str) -> list[str]:
    selected: list[str] = []
    for line in captured.splitlines():
        normalized = line.strip()
        if "tests/test_attention.py::" in normalized and any(
            marker in normalized
            for marker in (" PASSED", " FAILED", " SKIPPED")
        ):
            selected.append(
                normalized[normalized.index("tests/test_attention.py::") :]
            )
        elif (
            " passed" in normalized
            or " failed" in normalized
            or " skipped" in normalized
        ) and " in " in normalized:
            selected.append(normalized.strip("= "))
    return selected


def main() -> int:
    args = parse_args()
    run = configure_formal_run(seed=args.seed, tf32_enabled=False)
    official_source = args.official_test.resolve()
    adapter_source = SUBMISSION_ROOT / "tests" / "adapters.py"
    if not official_source.is_file():
        raise FileNotFoundError("official test file is missing")
    official_sha256 = hashlib.sha256(
        official_source.read_bytes()
    ).hexdigest()

    captured = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="a2k-official-tests-") as raw_temp:
        overlay = Path(raw_temp)
        tests = overlay / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("", encoding="utf-8")
        shutil.copy2(official_source, tests / "test_attention.py")
        os.symlink(adapter_source, tests / "adapters.py")
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(
            captured
        ):
            exit_code = int(
                pytest.main(
                    [
                        "-v",
                        "--tb=short",
                        "--rootdir",
                        str(overlay),
                        str(tests / "test_attention.py"),
                    ]
                )
            )

    selected = _selected_output(captured.getvalue())
    lines = [
        "OpenMOSS A2-K official GPU test run (public, de-identified)",
        f"starter commit: {STARTER_COMMIT}",
        "command: python -m student_scripts.a2k.official_tests "
        "--official-test ../assignment2-systems/tests/test_attention.py",
        f"official test sha256: {official_sha256}",
        f"GPU: {torch.cuda.get_device_name(0)}",
        (
            "allocator: "
            f"{run.allocator.allocator_limit_mib} MiB / "
            f"fraction {run.allocator.allocator_fraction:.12f}"
        ),
        f"free memory at start: {run.free_memory_mib_at_start:.2f} MiB",
        *selected,
        f"pytest exit code: {exit_code}",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = public_run_record(
        run=run,
        experiment="official_attention_tests",
        command=(
            "python -m student_scripts.a2k.official_tests "
            "--official-test ../assignment2-systems/tests/test_attention.py"
        ),
        timer="pytest wall time",
        warmup={"kind": "official tests"},
        measurement={"official_test_sha256": official_sha256},
        extra={"pytest_exit_code": exit_code},
    )
    upsert_json_record(
        args.metadata,
        record,
        key_fields=("experiment",),
    )
    return 0 if exit_code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
