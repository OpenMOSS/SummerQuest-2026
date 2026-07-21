"""Locate and copy A2-K code from the required upstream repository."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


A2K_REPOSITORY = "https://github.com/stanford-cs336/assignment2-systems.git"
A2K_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
A2K_DIRECTORY = "assignment2-systems"


class SourceError(RuntimeError):
    """Raised when the required A2-K upstream repository is incompatible."""


def source_path(root: Path) -> Path:
    """Return the one supported A2-K workspace location."""
    return root.resolve().parent / A2K_DIRECTORY


def git_output(source: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise SourceError("git is not installed or not available on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise SourceError(
            f"cannot read A2-K upstream repository at {source}: "
            f"{exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def validate_source(root: Path) -> Path:
    """Validate that the sibling repo contains and descends from the pinned starter."""
    source = source_path(root)
    if not source.is_dir():
        raise FileNotFoundError(
            "missing A2-K upstream repository; expected ../assignment2-systems next "
            f"to {root.name}: {source}\nRun: git clone {A2K_REPOSITORY} {source}"
        )

    commit = git_output(source, "rev-parse", f"{A2K_COMMIT}^{{commit}}")
    if commit != A2K_COMMIT:
        raise SourceError(
            "../assignment2-systems does not contain the pinned A2-K starter commit"
        )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "merge-base",
                "--is-ancestor",
                A2K_COMMIT,
                "HEAD",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SourceError(
            "../assignment2-systems HEAD must be the pinned A2-K commit or a student "
            "branch based on it"
        ) from exc

    package = source / "cs336_systems"
    adapters = source / "tests" / "adapters.py"
    pyproject = source / "pyproject.toml"
    if not package.is_dir() or not adapters.is_file() or not pyproject.is_file():
        raise SourceError(
            "../assignment2-systems is incomplete; expected cs336_systems/, "
            "tests/adapters.py, and pyproject.toml"
        )
    return source


def _copy_python_tree(source: Path, destination: Path) -> None:
    if source.exists() and not source.is_dir():
        raise SourceError(f"A2-K source path must be a directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_symlink():
            raise SourceError(f"symlinks are not allowed in synced A2-K files: {path}")
        if not path.is_file() or path.suffix != ".py":
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def copy_submission(source: Path, destination: Path) -> None:
    """Replace the public A2-K copy with the student-authored allowlist."""
    package = source / "cs336_systems" / "a2k"
    adapters = source / "tests" / "adapters.py"
    scripts = source / "student_scripts" / "a2k"

    for path in (package, adapters, scripts):
        if path.is_symlink():
            raise SourceError(f"symlinks are not allowed in synced A2-K files: {path}")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    _copy_python_tree(package, destination / "cs336_systems" / "a2k")
    (destination / "tests").mkdir()
    shutil.copy2(adapters, destination / "tests" / "adapters.py")
    _copy_python_tree(scripts, destination / "student_scripts" / "a2k")
