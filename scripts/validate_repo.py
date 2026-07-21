#!/usr/bin/env python3
"""Validate student submission structure and obvious public-repo safety issues."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STUDENTS = ROOT / "students"
TEMPLATE = STUDENTS / "_template"
MAX_STUDENT_FILE_BYTES = 5 * 1024 * 1024
A2P_MAX_REPORT_BYTES = 1 * 1024 * 1024
A2P_MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024
A2K_MAX_REPORT_BYTES = 1 * 1024 * 1024
A2K_MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024

TEMPLATE_FILES = (
    "PROFILE.md",
    "assignments/A0/README.md",
)

ASSIGNMENT_TEMPLATE_FILES = ("README.md",)

STUDENT_FILES = (
    "PROFILE.md",
    "assignments/A0/README.md",
)

FEISHU_URL = re.compile(
    r"https://[^\s)>]*(?:feishu\.cn|larksuite\.com)/(?:docx|wiki)/[^\s)>]+",
    re.IGNORECASE,
)
PLACEHOLDER = re.compile(r"<[^>\n]+>")
PROFILE_GUIDANCE = re.compile(r"^>\s*\[填写参考\]", re.MULTILINE)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
SECRET_VALUE = re.compile(
    r"(?i)['\"]?(?:app[_ -]?secret|client[_ -]?secret|api[_ -]?key|"
    r"verification[_ -]?token|encrypt[_ -]?key|webhook[_ -]?secret|"
    r"access[_ -]?token|refresh[_ -]?token|password)['\"]?"
    r"\s*(?:=|:)\s*['\"]?(?!(?:replace|example|your|changeme))"
    r"[A-Za-z0-9_./+\-=]{16,}"
)
KNOWN_TOKEN = re.compile(
    r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|"
    r"sk-[A-Za-z0-9_-]{20,})"
)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)")
DENIED_STUDENT_SUFFIXES = {
    ".7z",
    ".bz2",
    ".db",
    ".gz",
    ".key",
    ".pem",
    ".rar",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".xz",
    ".zip",
}

A2P_REQUIRED_FILES = (
    "results/benchmark.csv",
    "results/profile/trace_summary.csv",
    "results/profile/run_metadata.json",
    "results/mixed_precision.json",
    "results/memory/peaks.csv",
    "results/memory/run_metadata.json",
)
A2P_ALLOWED_RESULT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt"}
A2P_ALLOWED_ASSET_SUFFIXES = {".jpeg", ".jpg", ".png", ".svg", ".webp"}
A2P_ALLOWED_TOP_LEVEL = {"README.md", "assets", "results", "submission"}

A2K_REQUIRED_FILES = (
    "results/correctness.json",
    "results/unit_tests.txt",
    "results/checkpointing.csv",
    "results/attention_baseline.csv",
    "results/compile_comparison.csv",
    "results/flash_benchmark.csv",
    "results/memory_evidence.json",
    "results/run_metadata.json",
)
A2K_ALLOWED_RESULT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt"}
A2K_ALLOWED_ASSET_SUFFIXES = {".jpeg", ".jpg", ".png", ".svg", ".webp"}
A2K_ALLOWED_TOP_LEVEL = {"README.md", "assets", "results", "submission"}


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ""


def validate_template(errors: list[str]) -> None:
    for relative in TEMPLATE_FILES:
        path = TEMPLATE / relative
        if not path.is_file():
            errors.append(f"missing template file: {path.relative_to(ROOT)}")
    assignment_template = STUDENTS / "_assignment_template"
    for relative in ASSIGNMENT_TEMPLATE_FILES:
        path = assignment_template / relative
        if not path.is_file():
            errors.append(f"missing assignment template file: {path.relative_to(ROOT)}")
    a1_template = STUDENTS / "_assignment_templates" / "A1" / "README.md"
    if not a1_template.is_file():
        errors.append(
            f"missing A1 assignment template: {a1_template.relative_to(ROOT)}"
        )
    a2p_template = STUDENTS / "_assignment_templates" / "A2-P" / "README.md"
    if not a2p_template.is_file():
        errors.append(
            f"missing A2-P assignment template: {a2p_template.relative_to(ROOT)}"
        )
    a2k_template = STUDENTS / "_assignment_templates" / "A2-K" / "README.md"
    if not a2k_template.is_file():
        errors.append(
            f"missing A2-K assignment template: {a2k_template.relative_to(ROOT)}"
        )
    vendored_a1 = ROOT / "starter" / "A1"
    if vendored_a1.exists():
        errors.append(
            "A1 upstream repository must remain external at ../assignment1-basics; "
            "do not vendor starter/A1"
        )
    vendored_a2p = ROOT / "starter" / "A2-P"
    if vendored_a2p.exists():
        errors.append(
            "A2-P upstream repository must remain external at "
            "../assignment2-systems; do not vendor starter/A2-P"
        )
    vendored_a2k = ROOT / "starter" / "A2-K"
    if vendored_a2k.exists():
        errors.append(
            "A2-K upstream repository must remain external at "
            "../assignment2-systems; do not vendor starter/A2-K"
        )


def validate_a2p_submission(
    assignment: Path, report: str, errors: list[str]
) -> None:
    relative = assignment.relative_to(ROOT)
    readme = assignment / "README.md"
    if readme.stat().st_size > A2P_MAX_REPORT_BYTES:
        errors.append(f"A2-P README exceeds 1 MiB: {readme.relative_to(ROOT)}")

    for target in MARKDOWN_LINK.findall(report):
        if target.startswith("https://") or target.startswith("#"):
            continue
        if "://" in target or target.startswith("/"):
            errors.append(
                f"A2-P README link must use HTTPS or remain inside SummerQuest-2026: "
                f"{target}"
            )
            continue
        relative_target = target.split("#", 1)[0].split("?", 1)[0]
        if not relative_target:
            continue
        resolved = (readme.parent / relative_target).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(
                "A2-P external repository links must use GitHub HTTPS absolute URLs: "
                f"{target}"
            )

    top_level = {path.name for path in assignment.iterdir()}
    unexpected = sorted(top_level - A2P_ALLOWED_TOP_LEVEL)
    for name in unexpected:
        errors.append(f"unexpected A2-P top-level entry: {relative}/{name}")

    for required in A2P_REQUIRED_FILES:
        path = assignment / required
        if not path.is_file():
            errors.append(
                f"missing required A2-P result file: {path.relative_to(ROOT)}"
            )

    submission = assignment / "submission"
    profiling = submission / "profiling"
    submission_files = (
        sorted(path for path in submission.rglob("*") if path.is_file())
        if submission.is_dir()
        else []
    )
    profiling_files = (
        sorted(path for path in profiling.rglob("*") if path.is_file())
        if profiling.is_dir()
        else []
    )
    if not profiling_files:
        errors.append(f"A2-P submission has no profiling Python files: {relative}")
    for path in submission_files:
        submission_relative = path.relative_to(submission)
        if (
            not submission_relative.parts
            or submission_relative.parts[0] != "profiling"
            or path.suffix != ".py"
        ):
            errors.append(
                f"A2-P submission only accepts submission/profiling/**/*.py: "
                f"{path.relative_to(ROOT)}"
            )

    results = assignment / "results"
    if not results.is_dir():
        errors.append(f"missing A2-P results directory: {relative}/results")
    else:
        for path in sorted(item for item in results.rglob("*") if item.is_file()):
            if path.suffix.lower() not in A2P_ALLOWED_RESULT_SUFFIXES:
                errors.append(
                    f"unsupported A2-P result file type: {path.relative_to(ROOT)}"
                )

    assets = assignment / "assets"
    asset_files = (
        sorted(path for path in assets.rglob("*") if path.is_file())
        if assets.is_dir()
        else []
    )
    if len(asset_files) < 3:
        errors.append(f"A2-P requires at least three report images: {relative}/assets")
    for path in asset_files:
        if path.suffix.lower() not in A2P_ALLOWED_ASSET_SUFFIXES:
            errors.append(f"unsupported A2-P asset type: {path.relative_to(ROOT)}")
        asset_reference = path.relative_to(assignment).as_posix()
        if asset_reference not in report:
            errors.append(
                f"A2-P asset is not referenced from README.md: "
                f"{path.relative_to(ROOT)}"
            )

    attachment_bytes = sum(
        path.stat().st_size
        for directory in (results, assets)
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
    )
    if attachment_bytes > A2P_MAX_ATTACHMENT_BYTES:
        errors.append(
            f"A2-P results/ and assets/ exceed the 2 MiB attachment budget: {relative}"
        )


def _a2k_submission_file_allowed(path: Path, submission: Path) -> bool:
    relative = path.relative_to(submission)
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("cs336_systems", "a2k"):
        return path.suffix == ".py"
    if parts == ("tests", "adapters.py"):
        return True
    if len(parts) >= 3 and parts[:2] == ("student_scripts", "a2k"):
        return path.suffix == ".py"
    return False


def _validate_a2k_memory_evidence(path: Path, errors: list[str]) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        allocator = data["allocator"]
        allocator_fraction = float(allocator["allocator_fraction"])
        allocator_limit = float(allocator["allocator_limit_mib"])
        hard_limit = float(data["hard_limit_mib"])
        peak_allocated = float(data["pytorch_peak_allocated_mib"])
        peak_reserved = float(data["pytorch_peak_reserved_mib"])
        within_24gib = data["within_24gib"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        errors.append(
            f"invalid A2-K memory evidence schema: {path.relative_to(ROOT)}"
        )
        return

    if allocator_limit != 23552 or hard_limit != 24576:
        errors.append(
            f"A2-K memory evidence must use 23552/24576 MiB limits: "
            f"{path.relative_to(ROOT)}"
        )
    if not 0 < allocator_fraction <= 1:
        errors.append(
            f"A2-K allocator fraction must be in (0, 1]: {path.relative_to(ROOT)}"
        )
    if peak_allocated < 0 or peak_reserved < peak_allocated:
        errors.append(
            f"A2-K memory peaks are inconsistent: {path.relative_to(ROOT)}"
        )
    if peak_reserved > 23552:
        errors.append(
            f"A2-K peak reserved exceeds 23552 MiB: {path.relative_to(ROOT)}"
        )
    if within_24gib is not True:
        errors.append(
            f"A2-K memory evidence must report within_24gib=true: "
            f"{path.relative_to(ROOT)}"
        )

    nvidia_smi = data.get("nvidia_smi")
    if isinstance(nvidia_smi, dict) and "max_gpu_memory_used_mib" in nvidia_smi:
        try:
            external_peak = float(nvidia_smi["max_gpu_memory_used_mib"])
        except (TypeError, ValueError):
            errors.append(
                f"invalid A2-K nvidia-smi peak: {path.relative_to(ROOT)}"
            )
        else:
            if external_peak > 24576:
                errors.append(
                    f"A2-K external memory peak exceeds 24576 MiB: "
                    f"{path.relative_to(ROOT)}"
                )


def validate_a2k_submission(
    assignment: Path, report: str, errors: list[str]
) -> None:
    relative = assignment.relative_to(ROOT)
    readme = assignment / "README.md"
    if readme.stat().st_size > A2K_MAX_REPORT_BYTES:
        errors.append(f"A2-K README exceeds 1 MiB: {readme.relative_to(ROOT)}")

    for target in MARKDOWN_LINK.findall(report):
        if target.startswith("https://") or target.startswith("#"):
            continue
        if "://" in target or target.startswith("/"):
            errors.append(
                "A2-K README link must use HTTPS or remain inside SummerQuest-2026: "
                f"{target}"
            )
            continue
        relative_target = target.split("#", 1)[0].split("?", 1)[0]
        if not relative_target:
            continue
        resolved = (readme.parent / relative_target).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(
                "A2-K external repository links must use GitHub HTTPS absolute URLs: "
                f"{target}"
            )

    top_level = {path.name for path in assignment.iterdir()}
    for name in sorted(top_level - A2K_ALLOWED_TOP_LEVEL):
        errors.append(f"unexpected A2-K top-level entry: {relative}/{name}")

    for required in A2K_REQUIRED_FILES:
        path = assignment / required
        if not path.is_file():
            errors.append(
                f"missing required A2-K result file: {path.relative_to(ROOT)}"
            )

    memory_evidence = assignment / "results" / "memory_evidence.json"
    if memory_evidence.is_file():
        _validate_a2k_memory_evidence(memory_evidence, errors)

    submission = assignment / "submission"
    submission_files = (
        sorted(path for path in submission.rglob("*") if path.is_file())
        if submission.is_dir()
        else []
    )
    package = submission / "cs336_systems" / "a2k"
    package_files = (
        sorted(path for path in package.rglob("*") if path.is_file())
        if package.is_dir()
        else []
    )
    student_scripts = submission / "student_scripts" / "a2k"
    script_files = (
        sorted(path for path in student_scripts.rglob("*") if path.is_file())
        if student_scripts.is_dir()
        else []
    )
    adapters = submission / "tests" / "adapters.py"

    if not any(path.suffix == ".py" for path in package_files):
        errors.append(
            f"A2-K submission has no cs336_systems/a2k/**/*.py files: {relative}"
        )
    if not adapters.is_file():
        errors.append(f"A2-K submission is missing tests/adapters.py: {relative}")
    if not any(path.suffix == ".py" for path in script_files):
        errors.append(
            f"A2-K submission has no student_scripts/a2k/**/*.py files: {relative}"
        )
    for path in submission_files:
        if not _a2k_submission_file_allowed(path, submission):
            errors.append(
                f"unsupported A2-K submission file: {path.relative_to(ROOT)}"
            )

    results = assignment / "results"
    if not results.is_dir():
        errors.append(f"missing A2-K results directory: {relative}/results")
    else:
        for path in sorted(item for item in results.rglob("*") if item.is_file()):
            if path.suffix.lower() not in A2K_ALLOWED_RESULT_SUFFIXES:
                errors.append(
                    f"unsupported A2-K result file type: {path.relative_to(ROOT)}"
                )

    assets = assignment / "assets"
    asset_files = (
        sorted(path for path in assets.rglob("*") if path.is_file())
        if assets.is_dir()
        else []
    )
    if len(asset_files) < 2:
        errors.append(f"A2-K requires at least two report images: {relative}/assets")
    for path in asset_files:
        if path.suffix.lower() not in A2K_ALLOWED_ASSET_SUFFIXES:
            errors.append(f"unsupported A2-K asset type: {path.relative_to(ROOT)}")
        asset_reference = path.relative_to(assignment).as_posix()
        if asset_reference not in report:
            errors.append(
                f"A2-K asset is not referenced from README.md: "
                f"{path.relative_to(ROOT)}"
            )

    attachment_bytes = sum(
        path.stat().st_size
        for directory in (results, assets)
        if directory.is_dir()
        for path in directory.rglob("*")
        if path.is_file()
    )
    if attachment_bytes > A2K_MAX_ATTACHMENT_BYTES:
        errors.append(
            f"A2-K results/ and assets/ exceed the 2 MiB attachment budget: {relative}"
        )


def validate_assignment(student: Path, assignment: Path, errors: list[str]) -> None:
    relative = assignment.relative_to(ROOT)
    readme = assignment / "README.md"
    if not readme.is_file():
        errors.append(f"missing public assignment README: {relative}/README.md")
        return

    report = read_text(readme)
    placeholder_text = (
        report.replace("<|endoftext|>", "") if assignment.name == "A1" else report
    )
    if PLACEHOLDER.search(placeholder_text):
        errors.append(f"unfilled placeholder: {relative}/README.md")
    if not FEISHU_URL.search(report):
        errors.append(f"missing Feishu supplement URL: {relative}/README.md")

    if assignment.name == "A0":
        report = report.lower()
        for marker in ("nvidia-smi", "gpustat", "exit code"):
            if marker not in report:
                errors.append(f"A0 report missing '{marker}': {relative}/README.md")
    elif assignment.name == "A2-P":
        validate_a2p_submission(assignment, report, errors)
    elif assignment.name == "A2-K":
        validate_a2k_submission(assignment, report, errors)


def validate_student(student: Path, errors: list[str]) -> None:
    name = student.name
    relative = student.relative_to(ROOT)
    if name.startswith(".") or any(char.isspace() for char in name):
        errors.append(
            f"student directory must use a real name without spaces: {relative}"
        )

    for required in STUDENT_FILES:
        path = student / required
        if not path.is_file():
            errors.append(f"missing student file: {path.relative_to(ROOT)}")

    for required in STUDENT_FILES:
        path = student / required
        if path.is_file() and PLACEHOLDER.search(read_text(path)):
            errors.append(f"unfilled placeholder: {path.relative_to(ROOT)}")

    profile = student / "PROFILE.md"
    if profile.is_file():
        profile_text = read_text(profile)
        if not FEISHU_URL.search(profile_text):
            errors.append(f"missing Feishu profile URL: {profile.relative_to(ROOT)}")
        if PROFILE_GUIDANCE.search(profile_text):
            errors.append(f"template guidance not removed: {profile.relative_to(ROOT)}")

    assignments = student / "assignments"
    if assignments.is_dir():
        for assignment in sorted(
            path for path in assignments.iterdir() if path.is_dir()
        ):
            if not re.fullmatch(r"A(?:[0-1]|2-[KP]|[3-6])", assignment.name):
                errors.append(
                    "unknown assignment directory; expected A0, A1, A2-P, A2-K, "
                    f"or A3-A6: {assignment.relative_to(ROOT)}"
                )
                continue
            validate_assignment(student, assignment, errors)

    for path in student.rglob("*"):
        if path.is_symlink():
            errors.append(
                f"symbolic links are not allowed in student submissions: {path.relative_to(ROOT)}"
            )
        elif path.is_file() and path.stat().st_size > MAX_STUDENT_FILE_BYTES:
            errors.append(
                f"student file exceeds 5 MiB; use an approved external artifact location: "
                f"{path.relative_to(ROOT)}"
            )
        elif path.is_file() and path.suffix.lower() in DENIED_STUDENT_SUFFIXES:
            errors.append(
                f"archive, database, or key file is not allowed in student submissions: "
                f"{path.relative_to(ROOT)}"
            )


def validate_secrets(errors: list[str]) -> None:
    ignored_parts = {".git", ".venv", "__pycache__"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in ignored_parts for part in path.parts):
            continue
        text = read_text(path)
        if not text:
            continue
        if PRIVATE_KEY.search(text):
            errors.append(f"private key material detected: {path.relative_to(ROOT)}")
        if SECRET_VALUE.search(text):
            errors.append(
                f"possible credential value detected: {path.relative_to(ROOT)}"
            )
        if KNOWN_TOKEN.search(text):
            errors.append(f"known token format detected: {path.relative_to(ROOT)}")


def main() -> int:
    errors: list[str] = []
    validate_template(errors)
    validate_secrets(errors)

    student_dirs = sorted(
        path
        for path in STUDENTS.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )
    for student in student_dirs:
        validate_student(student, errors)

    if errors:
        print("Repository validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    if student_dirs:
        print(f"Repository validation passed for {len(student_dirs)} student(s).")
    else:
        print(
            "Repository validation passed. No student submissions yet; templates are present."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
