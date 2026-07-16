"""Locate and copy student A1 work from the required sibling repository."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


A1_REPOSITORY = "https://github.com/stanford-cs336/assignment1-basics.git"
A1_COMMIT = "a158843b20107949f1a8d7df1b05cd33b9166712"
A1_TREE = "1d7ee3636de57c499d13847edc02881e5b516bb4"
A1_DIRECTORY = "assignment1-basics"


class SourceError(RuntimeError):
    """Raised when the required sibling A1 repository is missing or incompatible."""


def source_path(root: Path) -> Path:
    """Return the one supported A1 workspace location."""
    return root.resolve().parent / A1_DIRECTORY


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
            f"cannot read A1 repository at {source}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.strip()


def validate_source(root: Path) -> Path:
    """Validate that the sibling repo contains and descends from the pinned starter."""
    source = source_path(root)
    if not source.is_dir():
        raise FileNotFoundError(
            "missing sibling A1 repository; expected ../assignment1-basics next to "
            f"{root.name}: {source}\nRun: git clone {A1_REPOSITORY} {source}"
        )

    commit = git_output(source, "rev-parse", f"{A1_COMMIT}^{{commit}}")
    tree = git_output(source, "rev-parse", f"{A1_COMMIT}^{{tree}}")
    if commit != A1_COMMIT or tree != A1_TREE:
        raise SourceError(
            "../assignment1-basics does not contain the pinned A1 starter commit"
        )
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "merge-base",
                "--is-ancestor",
                A1_COMMIT,
                "HEAD",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SourceError(
            "../assignment1-basics HEAD must be the pinned commit or a student branch based on it"
        ) from exc

    package = source / "cs336_basics"
    adapters = source / "tests" / "adapters.py"
    if not package.is_dir() or not adapters.is_file():
        raise SourceError(
            "../assignment1-basics is incomplete; expected cs336_basics/ and tests/adapters.py"
        )
    return source


def copy_submission(source: Path, destination: Path) -> None:
    """Replace the student submission copy with selected files from the A1 worktree."""
    selected_directories = [source / "cs336_basics"]
    selected_directories.extend(
        source / directory
        for directory in ("scripts", "configs")
        if (source / directory).is_dir()
    )
    for directory in selected_directories:
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise SourceError(
                    f"symlinks are not allowed in synced A1 files: {path}"
                )

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".git")
    destination.mkdir(parents=True, exist_ok=True)

    package_destination = destination / "cs336_basics"
    if package_destination.exists():
        shutil.rmtree(package_destination)
    shutil.copytree(source / "cs336_basics", package_destination, ignore=ignore)

    tests_destination = destination / "tests"
    if tests_destination.exists():
        shutil.rmtree(tests_destination)
    tests_destination.mkdir()
    shutil.copy2(source / "tests" / "adapters.py", tests_destination / "adapters.py")

    for directory in ("scripts", "configs"):
        source_directory = source / directory
        destination_directory = destination / directory
        if destination_directory.exists():
            shutil.rmtree(destination_directory)
        if source_directory.is_dir():
            shutil.copytree(source_directory, destination_directory, ignore=ignore)
        else:
            destination_directory.mkdir()
