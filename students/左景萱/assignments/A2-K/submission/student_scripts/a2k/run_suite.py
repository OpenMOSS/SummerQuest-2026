#!/usr/bin/env python3
"""Run the complete A2-K evidence suite serially in isolated processes.

The coordinator deliberately performs no CUDA work itself.  It creates a
private raw-evidence directory, strips credentials from every child
environment, and launches one child at a time.  The formal and development
CUDA modes use the complete handout matrices; only the explicitly selected
CPU ``--dry-run`` mode substitutes tiny control-flow cases.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

from student_scripts.a2k.runtime import (
    STARTER_COMMIT,
    atomic_write_json,
    child_process_environment,
)


SCHEMA_VERSION = "cs336.a2k.suite-manifest.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RUNTIME_ROOT = (PROJECT_ROOT / ".runtime").resolve()
PHASES = ("forward", "backward", "forward_backward")
IMPLEMENTATIONS = ("eager", "compiled", "triton")
CORE_SEQUENCE_LENGTHS = (512, 2048, 8192)
BOUNDARY_SEQUENCE_LENGTHS = (16384,)
HEAD_DIMS = (64, 128)
CORRECTNESS_SEEDS = (17, 42, 336)
CORRECTNESS_HEAD_DIMS = (32, 64, 128)
_SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")
_CREDENTIAL_ENV_NAME = re.compile(
    r"(?:TOKEN|PASSWORD|PASSWD|SECRET|COOKIE|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL|AUTH|USERNAME|USER_ID|APP_ID|CLIENT_ID)",
    re.IGNORECASE,
)

# These bounds are deliberately per fresh process, rather than one timeout for
# the whole suite.  The larger phases contain their own serial case matrices;
# attention and correctness already launch one process per case here.
UNIT_TEST_TIMEOUT_SECONDS = 10 * 60
CORRECTNESS_CASE_TIMEOUT_SECONDS = 5 * 60
CORRECTNESS_PHASE_TIMEOUT_SECONDS = 19 * CORRECTNESS_CASE_TIMEOUT_SECONDS
CHECKPOINT_TIMEOUT_SECONDS = 2 * 60 * 60
COMPILE_TIMEOUT_SECONDS = 2 * 60 * 60
ATTENTION_CASE_TIMEOUT_SECONDS = 5 * 60
ATTENTION_PHASE_TIMEOUT_SECONDS = 66 * ATTENTION_CASE_TIMEOUT_SECONDS
TERMINATION_GRACE_SECONDS = 10
TIMEOUT_RETURN_CODE = 124


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def attention_cases(*, dry_run: bool) -> list[dict[str, Any]]:
    """Return the exact CUDA matrix or an explicit tiny CPU smoke matrix."""

    if dry_run:
        return [
            {
                "case_id": f"n00016-d032-{implementation}-{phase}",
                "matrix": "dry_run_control_flow",
                "sequence_length": 16,
                "head_dim": 32,
                "implementation": implementation,
                "phase": phase,
            }
            for implementation in IMPLEMENTATIONS
            for phase in PHASES
        ]

    rows: list[dict[str, Any]] = []
    for sequence_length in CORE_SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for implementation in IMPLEMENTATIONS:
                for phase in PHASES:
                    rows.append(
                        {
                            "case_id": f"n{sequence_length:05d}-d{head_dim:03d}-{implementation}-{phase}",
                            "matrix": "core",
                            "sequence_length": sequence_length,
                            "head_dim": head_dim,
                            "implementation": implementation,
                            "phase": phase,
                        }
                    )
    for sequence_length in BOUNDARY_SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for implementation in ("eager", "triton"):
                for phase in PHASES:
                    rows.append(
                        {
                            "case_id": f"n{sequence_length:05d}-d{head_dim:03d}-{implementation}-{phase}",
                            "matrix": "boundary",
                            "sequence_length": sequence_length,
                            "head_dim": head_dim,
                            "implementation": implementation,
                            "phase": phase,
                        }
                    )
    if len(rows) != 66:
        raise AssertionError("the fixed A2-K attention matrix must contain 66 cases")
    return rows


def correctness_cases(*, dry_run: bool) -> list[dict[str, Any]]:
    """Return the fixed per-process correctness matrix."""

    if dry_run:
        return [{"seed": 17, "head_dim": 32, "causal": False, "dtype": "float32", "sequence_length": 16}]
    rows = [
        {
            "seed": seed,
            "head_dim": head_dim,
            "causal": causal,
            "dtype": "bfloat16",
            "sequence_length": 128,
        }
        for seed in CORRECTNESS_SEEDS
        for head_dim in CORRECTNESS_HEAD_DIMS
        for causal in (False, True)
    ]
    rows.append({"seed": 17, "head_dim": 128, "causal": True, "dtype": "float32", "sequence_length": 128})
    return rows


def _run_id(value: str) -> str:
    if _SAFE_RUN_ID.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("run id must contain only letters, digits, dot, dash, or underscore")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run all A2-K tests and benchmarks in serial isolated processes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="new private raw-artifact directory below the shared cs336 workspace",
    )
    parser.add_argument("--run-id", type=_run_id, help="safe run label used when --raw-dir is omitted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--development-cuda",
        action="store_true",
        help="run the full matrix on non-standard CUDA hardware under the same 23 GiB allocator cap; never formal evidence",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run tiny CPU control-flow cases; never GPU correctness or performance evidence",
    )
    return parser


def _resolve_raw_dir(args: argparse.Namespace) -> tuple[Path, str]:
    if args.raw_dir is not None and args.run_id is not None:
        raise ValueError("use either --raw-dir or --run-id, not both")
    run_id = args.run_id or f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{time.time_ns() % 1_000_000:06d}"
    raw_dir = args.raw_dir or (RUNTIME_ROOT / "a2k" / "raw" / run_id)
    raw_dir = raw_dir.expanduser().resolve()
    allowed = False
    for root in (WORKSPACE_ROOT.resolve(), RUNTIME_ROOT):
        try:
            raw_dir.relative_to(root)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise ValueError("--raw-dir must stay below the shared cs336 workspace or the project runtime root")
    if raw_dir.exists():
        raise FileExistsError("raw artifact directory already exists; refusing to mix or overwrite runs")
    return raw_dir, run_id


def _mode(args: argparse.Namespace) -> str:
    if args.dry_run:
        return "dry_run"
    if args.development_cuda:
        return "development_cuda"
    return "formal_cuda"


def _mode_flags(args: argparse.Namespace) -> list[str]:
    if args.dry_run:
        return ["--dry-run"]
    if args.development_cuda:
        return ["--development-cuda"]
    return []


def _write_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _run_command(
    argv: Sequence[str],
    *,
    namespace: str,
    log_path: Path,
    timeout_seconds: float,
    environment_updates: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run one bounded process group with incremental private logging."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    environment = child_process_environment(namespace)
    if environment_updates:
        forbidden = sorted(name for name in environment_updates if _CREDENTIAL_ENV_NAME.search(name))
        if forbidden:
            raise ValueError("environment_updates must not contain credential-like variable names")
        environment.update(environment_updates)
    # Make Python workers flush diagnostic lines promptly.  Child output goes
    # directly to a mode-0600 file, so it survives coordinator interruption and
    # never needs to be retained in memory.
    environment["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_APPEND, 0o600)
    started = time.monotonic()
    timed_out = False
    process_return_code: int | None = None
    launch_error: str | None = None
    with os.fdopen(descriptor, "a", encoding="utf-8", errors="replace", buffering=1) as log_file:
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=PROJECT_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            launch_error = type(exc).__name__
            log_file.write(f"\n[a2k coordinator] child launch failed ({launch_error}, errno={exc.errno})\n")
            return_code = 127
        else:
            try:
                process_return_code = process.wait(timeout=timeout_seconds)
                return_code = process_return_code
            except subprocess.TimeoutExpired:
                timed_out = True
                log_file.write(f"\n[a2k coordinator] timeout after {timeout_seconds:g} seconds; terminating process group\n")
                process_return_code = _terminate_process_group(process)
                return_code = TIMEOUT_RETURN_CODE
            except BaseException:
                log_file.write("\n[a2k coordinator] coordinator interrupted; terminating process group\n")
                _terminate_process_group(process)
                raise
    elapsed_seconds = time.monotonic() - started
    return {
        "return_code": return_code,
        "process_return_code": process_return_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "elapsed_seconds": elapsed_seconds,
        "launch_error": launch_error,
        "log": log_path.relative_to(log_path.parents[2]).as_posix() if len(log_path.parents) >= 3 else log_path.name,
    }


def _terminate_process_group(process: subprocess.Popen[Any]) -> int | None:
    """Terminate a child session, escalating to SIGKILL after a short grace."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            return process.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            # A process stuck in uninterruptible kernel sleep cannot be reaped
            # here; returning keeps the evidence coordinator itself bounded.
            return process.poll()


_PYTEST_COUNT = {
    name: re.compile(rf"(?<!\d)(\d+)\s+{pattern}\b", re.IGNORECASE)
    for name, pattern in {
        "passed": "passed",
        "failed": "failed",
        "skipped": "skipped",
        "xfailed": "xfailed",
        "xpassed": "xpassed",
        "errors": "errors?",
    }.items()
}


def parse_pytest_summary(output: str) -> dict[str, Any]:
    """Extract aggregate pytest counts without copying path-bearing output."""

    counts: dict[str, int] = {}
    for name, pattern in _PYTEST_COUNT.items():
        matches = pattern.findall(output)
        counts[name] = int(matches[-1]) if matches else 0
    total = sum(counts.values())
    return {"parsed": total > 0, "total": total, **counts}


def _manifest_task(
    manifest: dict[str, Any],
    *,
    name: str,
    state: str,
    artifacts: list[str],
    return_codes: list[int],
    process_count: int | None = None,
    timeout_seconds: float | None = None,
) -> None:
    task = manifest["tasks"].setdefault(name, {})
    task.update(
        state=state,
        artifacts=artifacts,
        process_count=len(return_codes) if process_count is None else process_count,
        return_codes=list(return_codes),
    )
    if timeout_seconds is not None:
        task["timeout_seconds"] = timeout_seconds


def _start_manifest_task(
    raw_dir: Path,
    manifest: dict[str, Any],
    *,
    name: str,
    artifacts: list[str],
    process_count: int,
    timeout_seconds: float,
    has_cases: bool = False,
) -> None:
    """Persist phase state before any child for the phase can start."""

    task: dict[str, Any] = {
        "state": "running",
        "artifacts": artifacts,
        "process_count": process_count,
        "return_codes": [],
        "timeout_seconds": timeout_seconds,
        "started_at_utc": _utc_now(),
    }
    if has_cases:
        task["cases"] = {}
    manifest["tasks"][name] = task
    atomic_write_json(raw_dir / "manifest.json", manifest)


def _start_manifest_case(
    raw_dir: Path,
    manifest: dict[str, Any],
    *,
    task_name: str,
    case_id: str,
    artifact: str,
    log: str,
    timeout_seconds: float,
) -> None:
    """Persist a case as running before its fresh process is launched."""

    manifest["tasks"][task_name]["cases"][case_id] = {
        "state": "running",
        "artifact": artifact,
        "log": log,
        "timeout_seconds": timeout_seconds,
        "started_at_utc": _utc_now(),
    }
    atomic_write_json(raw_dir / "manifest.json", manifest)


def _finish_manifest_case(
    raw_dir: Path,
    manifest: dict[str, Any],
    *,
    task_name: str,
    case_id: str,
    state: str,
    result: Mapping[str, Any],
    artifact_present: bool,
) -> None:
    """Durably record one terminal case outcome before moving to the next."""

    case = manifest["tasks"][task_name]["cases"][case_id]
    case.update(
        state=state,
        return_code=result["return_code"],
        timed_out=result["timed_out"],
        elapsed_seconds=result["elapsed_seconds"],
        artifact_present=artifact_present,
        finished_at_utc=_utc_now(),
    )
    task = manifest["tasks"][task_name]
    task["return_codes"].append(result["return_code"])
    atomic_write_json(raw_dir / "manifest.json", manifest)


def _finish_manifest_task(
    raw_dir: Path,
    manifest: dict[str, Any],
    *,
    name: str,
    state: str,
    artifacts: list[str],
    return_codes: list[int],
    process_count: int | None = None,
) -> None:
    """Persist a terminal phase state immediately after aggregation."""

    _manifest_task(
        manifest,
        name=name,
        state=state,
        artifacts=artifacts,
        return_codes=return_codes,
        process_count=process_count,
    )
    manifest["tasks"][name]["finished_at_utc"] = _utc_now()
    atomic_write_json(raw_dir / "manifest.json", manifest)


def _run_unit_tests(raw_dir: Path, manifest: dict[str, Any], namespace: str) -> None:
    directory = raw_dir / "unit_tests"
    directory.mkdir(parents=True)
    artifacts = ["unit_tests/result.json", "unit_tests/runtime_guard.json", "unit_tests/output.txt"]
    _start_manifest_task(
        raw_dir,
        manifest,
        name="unit_tests",
        artifacts=artifacts,
        process_count=1,
        timeout_seconds=UNIT_TEST_TIMEOUT_SECONDS,
    )
    bootstrap = directory / "bootstrap"
    bootstrap.mkdir()
    runtime_result = directory / "runtime_guard.json"
    # Python imports sitecustomize before pytest.  This preserves the exact
    # official command while applying the allocator guard before tests can
    # create their first CUDA tensor.  The generated hook remains private.
    _write_private_text(
        bootstrap / "sitecustomize.py",
        "from pathlib import Path\n"
        "import os\n"
        "try:\n"
        "    from student_scripts.a2k.runtime import atomic_write_json, prepare_runtime\n"
        "    mode = os.environ['A2K_UNIT_TEST_MODE']\n"
        "    guard = prepare_runtime(dry_run=mode == 'dry_run', development_cuda=mode == 'development_cuda')\n"
        "    atomic_write_json(Path(os.environ['A2K_UNIT_TEST_RUNTIME_RESULT']), guard.metadata)\n"
        "except BaseException:\n"
        "    os._exit(86)\n",
    )
    command = ["uv", "run", "pytest", "tests/test_attention.py", "-v"]
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    pythonpath = str(bootstrap) if not inherited_pythonpath else os.pathsep.join((str(bootstrap), inherited_pythonpath))
    result = _run_command(
        command,
        namespace=f"{namespace}-unit-tests",
        log_path=directory / "output.txt",
        timeout_seconds=UNIT_TEST_TIMEOUT_SECONDS,
        environment_updates={
            "A2K_UNIT_TEST_MODE": manifest["execution_mode"],
            "A2K_UNIT_TEST_RUNTIME_RESULT": str(runtime_result),
            "PYTHONPATH": pythonpath,
            # Keep the exact official `uv run pytest ...` command while using
            # the same frozen interpreter/environment as every other case.
            "UV_PROJECT_ENVIRONMENT": sys.prefix,
            "UV_NO_SYNC": "1",
        },
    )
    output = (directory / "output.txt").read_text(encoding="utf-8", errors="replace")
    payload = {
        "schema_version": "cs336.a2k.unit-tests.v1",
        "command": "uv run pytest tests/test_attention.py -v",
        "return_code": result["return_code"],
        "timed_out": result["timed_out"],
        "timeout_seconds": result["timeout_seconds"],
        "summary": parse_pytest_summary(output),
        "raw_output_retained_privately": True,
    }
    atomic_write_json(directory / "result.json", payload)
    state = "timed_out" if result["timed_out"] else "completed" if payload["summary"]["parsed"] and runtime_result.is_file() else "invalid"
    _finish_manifest_task(
        raw_dir,
        manifest,
        name="unit_tests",
        state=state,
        artifacts=artifacts,
        return_codes=[result["return_code"]],
        process_count=1,
    )


_CORRECTNESS_WORKER = """
import json
from pathlib import Path
import sys
from student_scripts.a2k import correctness
from student_scripts.a2k.runtime import atomic_write_json
case = json.loads(sys.argv[1])
dry_run = sys.argv[2] == "dry_run"
development_cuda = sys.argv[2] == "development_cuda"
correctness.correctness_cases = lambda *, dry_run: [case]
payload, exit_code = correctness.run(dry_run=dry_run, development_cuda=development_cuda)
atomic_write_json(Path(sys.argv[3]), payload)
raise SystemExit(exit_code)
""".strip()


def _correctness_case_id(case: Mapping[str, Any]) -> str:
    causal = "causal" if case["causal"] else "noncausal"
    return f"seed{case['seed']}-n{case['sequence_length']}-d{case['head_dim']}-{case['dtype']}-{causal}"


def _run_correctness(raw_dir: Path, args: argparse.Namespace, manifest: dict[str, Any], namespace: str) -> None:
    directory = raw_dir / "correctness"
    case_directory = directory / "cases"
    log_directory = directory / "logs"
    case_directory.mkdir(parents=True)
    log_directory.mkdir(parents=True)
    cases = correctness_cases(dry_run=args.dry_run)
    expected_count = len(cases)
    artifacts = ["correctness/result.json", "correctness/cases/*.json", "correctness/logs/*.txt", "correctness/output.txt"]
    _start_manifest_task(
        raw_dir,
        manifest,
        name="correctness",
        artifacts=artifacts,
        process_count=expected_count,
        timeout_seconds=expected_count * CORRECTNESS_CASE_TIMEOUT_SECONDS,
        has_cases=True,
    )
    return_codes: list[int] = []
    case_records: list[dict[str, Any]] = []
    runtime_evidence: list[dict[str, Any]] = []
    template: dict[str, Any] | None = None
    summaries: list[str] = []
    for index, case in enumerate(cases, start=1):
        case_id = _correctness_case_id(case)
        output = case_directory / f"{case_id}.json"
        artifact = f"correctness/cases/{case_id}.json"
        log = f"correctness/logs/{case_id}.txt"
        _start_manifest_case(
            raw_dir,
            manifest,
            task_name="correctness",
            case_id=case_id,
            artifact=artifact,
            log=log,
            timeout_seconds=CORRECTNESS_CASE_TIMEOUT_SECONDS,
        )
        command = [
            sys.executable,
            "-c",
            _CORRECTNESS_WORKER,
            json.dumps(case, separators=(",", ":")),
            _mode(args),
            str(output),
        ]
        result = _run_command(
            command,
            namespace=f"{namespace}-correctness-{index:02d}",
            log_path=log_directory / f"{case_id}.txt",
            timeout_seconds=CORRECTNESS_CASE_TIMEOUT_SECONDS,
        )
        return_codes.append(result["return_code"])
        if not output.is_file():
            summaries.append(f"{case_id}: missing result (exit {result['return_code']})")
            _finish_manifest_case(
                raw_dir,
                manifest,
                task_name="correctness",
                case_id=case_id,
                state="timed_out" if result["timed_out"] else "invalid",
                result=result,
                artifact_present=False,
            )
            continue
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summaries.append(f"{case_id}: invalid result (exit {result['return_code']})")
            _finish_manifest_case(
                raw_dir,
                manifest,
                task_name="correctness",
                case_id=case_id,
                state="timed_out" if result["timed_out"] else "invalid",
                result=result,
                artifact_present=True,
            )
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list) or len(payload["cases"]) != 1:
            summaries.append(f"{case_id}: invalid result schema (exit {result['return_code']})")
            _finish_manifest_case(
                raw_dir,
                manifest,
                task_name="correctness",
                case_id=case_id,
                state="timed_out" if result["timed_out"] else "invalid",
                result=result,
                artifact_present=True,
            )
            continue
        template = template or payload
        case_record = payload["cases"][0]
        case_records.append(case_record)
        runtime_evidence.append({"case_id": case_record.get("case_id", case_id), "runtime": payload.get("runtime")})
        summaries.append(f"{case_id}: {case_record.get('status', payload.get('status'))} (exit {result['return_code']})")
        _finish_manifest_case(
            raw_dir,
            manifest,
            task_name="correctness",
            case_id=case_id,
            state="timed_out" if result["timed_out"] else "completed",
            result=result,
            artifact_present=True,
        )

    valid = template is not None and len(case_records) == expected_count and len(runtime_evidence) == expected_count
    if valid:
        checks = [
            result.get("status")
            for case_record in case_records
            for result in case_record.get("implementations", {}).values()
            if result.get("status") != "skipped_non_authoritative"
        ]
        skipped = sum(result.get("status") == "skipped_non_authoritative" for case_record in case_records for result in case_record.get("implementations", {}).values())
        combined = {key: value for key, value in template.items() if key not in {"cases", "summary", "status", "runtime", "error"}}
        combined.update(
            status=(
                "dry_run_ok"
                if args.dry_run and all(status == "pass" for status in checks)
                else "development_pass"
                if args.development_cuda and all(status == "pass" for status in checks)
                else "pass"
                if all(status == "pass" for status in checks)
                else "fail"
            ),
            runtime=runtime_evidence[0]["runtime"],
            case_runtime_evidence=runtime_evidence,
            cases=case_records,
            summary={
                "case_count": len(case_records),
                "implementation_check_count": len(checks),
                "passed": checks.count("pass"),
                "failed": checks.count("fail"),
                "errors": checks.count("error"),
                "oom": checks.count("oom"),
                "skipped": skipped,
                "bf16_case_count": sum(case.get("dtype") == "bfloat16" for case in case_records),
                "fp32_tf32_disabled_case_count": sum(case.get("dtype") == "float32" and case.get("tf32_policy") == "ieee" for case in case_records),
            },
        )
        atomic_write_json(directory / "result.json", combined)
    _write_private_text(directory / "output.txt", "\n".join(summaries) + "\n")
    any_timeout = any(case["state"] == "timed_out" for case in manifest["tasks"]["correctness"]["cases"].values())
    state = "timed_out" if any_timeout else "completed" if valid and (directory / "result.json").is_file() else "invalid"
    _finish_manifest_task(
        raw_dir,
        manifest,
        name="correctness",
        state=state,
        artifacts=artifacts,
        return_codes=return_codes,
        process_count=expected_count,
    )


def _run_checkpointing(raw_dir: Path, args: argparse.Namespace, manifest: dict[str, Any], namespace: str) -> None:
    directory = raw_dir / "checkpointing"
    directory.mkdir(parents=True)
    artifacts = ["checkpointing/result.json", "checkpointing/result.csv", "checkpointing/output.txt"]
    _start_manifest_task(
        raw_dir,
        manifest,
        name="checkpointing",
        artifacts=artifacts,
        process_count=1,
        timeout_seconds=CHECKPOINT_TIMEOUT_SECONDS,
    )
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.checkpointing",
        "--runtime-dir",
        str(directory / "runtime"),
        "--json-output",
        str(directory / "result.json"),
        "--csv-output",
        str(directory / "result.csv"),
        "--seed",
        str(args.seed),
        *_mode_flags(args),
    ]
    result = _run_command(
        command,
        namespace=f"{namespace}-checkpoint",
        log_path=directory / "output.txt",
        timeout_seconds=CHECKPOINT_TIMEOUT_SECONDS,
    )
    exists = (directory / "result.json").is_file() and (directory / "result.csv").is_file()
    process_count = 0
    if exists:
        try:
            payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            process_count = len(payload.get("results", [])) if isinstance(payload, dict) else 0
        except (OSError, json.JSONDecodeError):
            process_count = 0
    state = "timed_out" if result["timed_out"] else "completed" if exists and process_count > 0 else "invalid"
    _finish_manifest_task(
        raw_dir,
        manifest,
        name="checkpointing",
        state=state,
        artifacts=artifacts,
        return_codes=[result["return_code"]],
        process_count=process_count,
    )


def _run_compile(raw_dir: Path, args: argparse.Namespace, manifest: dict[str, Any], namespace: str) -> None:
    directory = raw_dir / "compile_comparison"
    directory.mkdir(parents=True)
    artifacts = ["compile_comparison/result.json", "compile_comparison/result.csv", "compile_comparison/output.txt"]
    _start_manifest_task(
        raw_dir,
        manifest,
        name="compile_comparison",
        artifacts=artifacts,
        process_count=1,
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
    )
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.compile_comparison",
        "--runtime-dir",
        str(directory / "runtime"),
        "--json-output",
        str(directory / "result.json"),
        "--csv-output",
        str(directory / "result.csv"),
        "--seed",
        str(args.seed),
        *_mode_flags(args),
    ]
    result = _run_command(
        command,
        namespace=f"{namespace}-compile",
        log_path=directory / "output.txt",
        timeout_seconds=COMPILE_TIMEOUT_SECONDS,
    )
    exists = (directory / "result.json").is_file() and (directory / "result.csv").is_file()
    process_count = 0
    if exists:
        try:
            payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            process_count = len(payload.get("results", [])) if isinstance(payload, dict) else 0
        except (OSError, json.JSONDecodeError):
            process_count = 0
    state = "timed_out" if result["timed_out"] else "completed" if exists and process_count > 0 else "invalid"
    _finish_manifest_task(
        raw_dir,
        manifest,
        name="compile_comparison",
        state=state,
        artifacts=artifacts,
        return_codes=[result["return_code"]],
        process_count=process_count,
    )


def _run_attention(raw_dir: Path, args: argparse.Namespace, manifest: dict[str, Any], namespace: str) -> None:
    directory = raw_dir / "attention"
    case_directory = directory / "cases"
    log_directory = directory / "logs"
    case_directory.mkdir(parents=True)
    log_directory.mkdir(parents=True)
    cases = attention_cases(dry_run=args.dry_run)
    artifacts = ["attention/index.json", "attention/cases/*.json", "attention/logs/*.txt"]
    _start_manifest_task(
        raw_dir,
        manifest,
        name="attention",
        artifacts=artifacts,
        process_count=len(cases),
        timeout_seconds=len(cases) * ATTENTION_CASE_TIMEOUT_SECONDS,
        has_cases=True,
    )
    records: list[dict[str, Any]] = []
    return_codes: list[int] = []
    for case in cases:
        output = case_directory / f"{case['case_id']}.json"
        artifact = f"attention/cases/{case['case_id']}.json"
        log = f"attention/logs/{case['case_id']}.txt"
        _start_manifest_case(
            raw_dir,
            manifest,
            task_name="attention",
            case_id=case["case_id"],
            artifact=artifact,
            log=log,
            timeout_seconds=ATTENTION_CASE_TIMEOUT_SECONDS,
        )
        command = [
            sys.executable,
            "-m",
            "student_scripts.a2k.attention_benchmark",
            "--sequence-length",
            str(case["sequence_length"]),
            "--head-dim",
            str(case["head_dim"]),
            "--implementation",
            str(case["implementation"]),
            "--phase",
            str(case["phase"]),
            "--seed",
            str(args.seed),
            "--output",
            str(output),
            *_mode_flags(args),
        ]
        result = _run_command(
            command,
            namespace=f"{namespace}-attention-{case['case_id']}",
            log_path=log_directory / f"{case['case_id']}.txt",
            timeout_seconds=ATTENTION_CASE_TIMEOUT_SECONDS,
        )
        return_codes.append(result["return_code"])
        artifact_present = output.is_file()
        records.append(
            {
                **case,
                "artifact": artifact,
                "log": log,
                "return_code": result["return_code"],
                "timed_out": result["timed_out"],
                "elapsed_seconds": result["elapsed_seconds"],
                "artifact_present": artifact_present,
            }
        )
        _finish_manifest_case(
            raw_dir,
            manifest,
            task_name="attention",
            case_id=case["case_id"],
            state="timed_out" if result["timed_out"] else "completed" if artifact_present else "invalid",
            result=result,
            artifact_present=artifact_present,
        )
        print(f"attention {case['case_id']}: exit {result['return_code']}", flush=True)

    index = {
        "schema_version": "cs336.a2k.attention-index.v1",
        "mode": _mode(args),
        "serial": True,
        "one_fresh_python_process_per_case": True,
        "expected_case_count": len(cases),
        "cases": records,
    }
    atomic_write_json(directory / "index.json", index)
    any_timeout = any(record["timed_out"] for record in records)
    complete = len(records) == len(cases) and all(record["artifact_present"] for record in records)
    state = "timed_out" if any_timeout else "completed" if complete else "invalid"
    _finish_manifest_task(
        raw_dir,
        manifest,
        name="attention",
        state=state,
        artifacts=artifacts,
        return_codes=return_codes,
        process_count=len(cases),
    )


def run_suite(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.dry_run and args.development_cuda:
        raise ValueError("--dry-run and --development-cuda are mutually exclusive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    raw_dir, run_id = _resolve_raw_dir(args)
    raw_dir.mkdir(parents=True, exist_ok=False)
    namespace = f"suite-{run_id}"
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": _utc_now(),
        "run_id": run_id,
        "starter_commit": STARTER_COMMIT,
        "seed": args.seed,
        "execution_mode": _mode(args),
        "authoritative": _mode(args) == "formal_cuda",
        "matrix_contract": ("tiny CPU control-flow matrix; not GPU evidence" if args.dry_run else "complete fixed A2-K CUDA matrix; no shape reduction"),
        "serial_execution": True,
        "credential_environment_forwarded": False,
        "tasks": {},
    }
    atomic_write_json(raw_dir / "manifest.json", manifest)

    runners = (
        _run_unit_tests,
        _run_correctness,
        _run_checkpointing,
        _run_compile,
        _run_attention,
    )
    try:
        for runner in runners:
            if runner is _run_unit_tests:
                runner(raw_dir, manifest, namespace)
            else:
                runner(raw_dir, args, manifest, namespace)
            atomic_write_json(raw_dir / "manifest.json", manifest)
        invalid = [name for name, task in manifest["tasks"].items() if task["state"] != "completed"]
        nonzero = [code for task in manifest["tasks"].values() for code in task["return_codes"] if code != 0]
        manifest["status"] = "invalid" if invalid else ("completed_with_failures" if nonzero else "completed")
    except BaseException:
        interrupted_at = _utc_now()
        for task in manifest["tasks"].values():
            if task.get("state") == "running":
                task["state"] = "interrupted"
                task["finished_at_utc"] = interrupted_at
            for case in task.get("cases", {}).values():
                if case.get("state") == "running":
                    case["state"] = "interrupted"
                    case["finished_at_utc"] = interrupted_at
        manifest["status"] = "interrupted"
        manifest["finished_at_utc"] = interrupted_at
        atomic_write_json(raw_dir / "manifest.json", manifest)
        raise
    manifest["finished_at_utc"] = _utc_now()
    atomic_write_json(raw_dir / "manifest.json", manifest)
    return raw_dir, manifest


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.development_cuda:
        parser.error("--dry-run and --development-cuda are mutually exclusive")
    try:
        raw_dir, manifest = run_suite(args)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(f"raw artifacts: {raw_dir}")
    print(f"suite status: {manifest['status']}")
    return 0 if manifest["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
