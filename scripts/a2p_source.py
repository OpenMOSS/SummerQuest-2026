"""Locate and copy A2-P profiling code from the required upstream repository."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


A2P_REPOSITORY = "https://github.com/stanford-cs336/assignment2-systems.git"
A2P_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
A2P_DIRECTORY = "assignment2-systems"


class SourceError(RuntimeError):
    """Raised when the required A2-P upstream repository is incompatible."""


def source_path(root: Path) -> Path:
    """Return the one supported A2-P workspace location."""
    return root.resolve().parent / A2P_DIRECTORY


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
            f"cannot read A2-P upstream repository at {source}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def validate_source(root: Path) -> Path:
    """Validate that the sibling repo contains and descends from the pinned starter."""
    source = source_path(root)
    if not source.is_dir():
        raise FileNotFoundError(
            "missing A2-P upstream repository; expected ../assignment2-systems next "
            f"to {root.name}: {source}\nRun: git clone {A2P_REPOSITORY} {source}"
        )

    commit = git_output(source, "rev-parse", f"{A2P_COMMIT}^{{commit}}")
    if commit != A2P_COMMIT:
        raise SourceError(
            "../assignment2-systems does not contain the pinned A2-P starter commit"
        )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "merge-base",
                "--is-ancestor",
                A2P_COMMIT,
                "HEAD",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SourceError(
            "../assignment2-systems HEAD must be the pinned A2-P commit or a student "
            "branch based on it"
        ) from exc

    required = (
        source / "cs336_systems",
        source / "tests",
        source / "pyproject.toml",
    )
    if (
        not required[0].is_dir()
        or not required[1].is_dir()
        or not required[2].is_file()
    ):
        raise SourceError(
            "../assignment2-systems is incomplete; expected cs336_systems/, tests/, "
            "and pyproject.toml"
        )
    return source


def copy_profiling(source: Path, destination: Path) -> None:
    """Replace the submission copy with Python files from profiling/ only."""
    source_directory = source / "profiling"
    if source_directory.exists() and not source_directory.is_dir():
        raise SourceError("../assignment2-systems/profiling must be a directory")

    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    if not source_directory.is_dir():
        return

    for path in source_directory.rglob("*"):
        if path.is_symlink():
            raise SourceError(f"symlinks are not allowed in synced A2-P files: {path}")
        if not path.is_file() or path.suffix != ".py":
            continue
        relative = path.relative_to(source_directory)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
