from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from pathlib import PurePosixPath
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import a2p_source  # noqa: E402
import validate_pr_scope  # noqa: E402
import validate_repo  # noqa: E402
from create_assignment import create_assignment  # noqa: E402
from sync_a2p_submission import sync_submission  # noqa: E402


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_source(parent: Path) -> tuple[Path, str]:
    source = parent / "assignment2-systems"
    (source / "cs336_systems").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "profiling").mkdir()
    (source / "cs336_systems" / "__init__.py").write_text("BASE = True\n")
    (source / "tests" / "test_public.py").write_text("def test_public(): pass\n")
    (source / "profiling" / "benchmark.py").write_text("MODE = 'base'\n")
    (source / "profiling" / "notes.txt").write_text("not submitted\n")
    (source / "pyproject.toml").write_text("[project]\nname='test-a2'\nversion='0'\n")
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test User")
    git(source, "config", "user.email", "test@example.com")
    git(source, "add", ".")
    git(source, "commit", "-q", "-m", "starter")
    return source, git(source, "rev-parse", "HEAD")


def make_summerquest(parent: Path) -> Path:
    root = parent / "SummerQuest-2026"
    student = root / "students" / "测试同学"
    student.mkdir(parents=True)
    (student / "PROFILE.md").write_text("profile\n")
    template = root / "students" / "_assignment_templates" / "A2-P"
    template.mkdir(parents=True)
    (template / "README.md").write_text("# A2-P <姓名> <A编号>\n")
    return root


class A2PSubmissionToolsTests(unittest.TestCase):
    def test_maintainer_a2p_pr_is_not_treated_as_a_student_pr(self) -> None:
        student, label = validate_pr_scope.validate_scope(
            [
                PurePosixPath("students/README.md"),
                PurePosixPath("students/_assignment_templates/A2-P/README.md"),
                PurePosixPath("assignments/A2-P/README.md"),
            ]
        )
        self.assertIsNone(student)
        self.assertEqual(label, "[MAINTAINER]")

    def test_pr_scope_accepts_a2p_as_one_review_unit(self) -> None:
        student, label = validate_pr_scope.validate_scope(
            [
                PurePosixPath(
                    "students/测试同学/assignments/A2-P/README.md"
                ),
                PurePosixPath(
                    "students/测试同学/assignments/A2-P/results/benchmark.csv"
                ),
            ],
            "[A2-P] 测试同学 - 完成 Profiling 作业",
        )
        self.assertEqual(student, "测试同学")
        self.assertEqual(label, "[A2-P]")

    def test_plain_a2_identifier_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = make_summerquest(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "A2-P"):
                create_assignment(root, "测试同学", "A2")

    def test_create_and_sync_copy_only_profiling_python_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, commit = make_source(parent)
            root = make_summerquest(parent)

            with mock.patch.object(a2p_source, "A2P_COMMIT", commit):
                assignment = create_assignment(root, "测试同学", "A2-P")
                profiling = assignment / "submission" / "profiling"
                self.assertEqual(
                    (profiling / "benchmark.py").read_text(), "MODE = 'base'\n"
                )
                self.assertFalse((profiling / "notes.txt").exists())
                self.assertTrue((assignment / "results" / "profile").is_dir())
                self.assertTrue((assignment / "results" / "memory").is_dir())
                self.assertTrue((assignment / "assets").is_dir())

                (source / "profiling" / "benchmark.py").write_text(
                    "MODE = 'updated'\n"
                )
                (source / "profiling" / "nested").mkdir()
                (source / "profiling" / "nested" / "nvtx.py").write_text("RANGES = 1\n")
                sync_submission(root, "测试同学")
                self.assertEqual(
                    (profiling / "benchmark.py").read_text(), "MODE = 'updated'\n"
                )
                self.assertTrue((profiling / "nested" / "nvtx.py").is_file())

    def test_missing_sibling_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = make_summerquest(Path(temp_dir))
            with self.assertRaisesRegex(FileNotFoundError, "../assignment2-systems"):
                create_assignment(root, "测试同学", "A2-P")

    def test_a2p_validator_enforces_the_submission_whitelist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment = root / "students" / "测试同学" / "assignments" / "A2-P"
            (assignment / "submission" / "profiling").mkdir(parents=True)
            (assignment / "results" / "profile").mkdir(parents=True)
            (assignment / "results" / "memory").mkdir()
            (assignment / "assets").mkdir()

            report = (
                "# A2-P\n\n![compute](assets/compute.png)\n\n"
                "![memory 128](assets/memory-128.png)\n\n"
                "![memory 2048](assets/memory-2048.png)\n"
            )
            (assignment / "README.md").write_text(report)
            (assignment / "submission" / "profiling" / "benchmark.py").write_text(
                "MODE = 'train_step'\n"
            )
            for relative in validate_repo.A2P_REQUIRED_FILES:
                path = assignment / relative
                path.write_text("{}\n" if path.suffix == ".json" else "value\n")
            (assignment / "assets" / "compute.png").write_bytes(b"png")
            (assignment / "assets" / "memory-128.png").write_bytes(b"png")
            (assignment / "assets" / "memory-2048.png").write_bytes(b"png")

            with mock.patch.object(validate_repo, "ROOT", root):
                errors: list[str] = []
                validate_repo.validate_a2p_submission(assignment, report, errors)
                self.assertEqual(errors, [])

                external_relative_report = (
                    report
                    + "\n[source](../../../../../assignment2-systems/profiling/run.py)\n"
                )
                errors = []
                validate_repo.validate_a2p_submission(
                    assignment, external_relative_report, errors
                )
                self.assertTrue(
                    any("GitHub HTTPS absolute URLs" in error for error in errors)
                )

                github_report = (
                    report
                    + "\n[source](https://github.com/stanford-cs336/"
                    "assignment2-systems/blob/ca8bc81a59b70516f7ebb2da4808daade877c736/"
                    "cs336_systems/model.py)\n"
                )
                errors = []
                validate_repo.validate_a2p_submission(
                    assignment, github_report, errors
                )
                self.assertEqual(errors, [])

                (assignment / "assets" / "memory-2048.png").unlink()
                errors = []
                validate_repo.validate_a2p_submission(assignment, report, errors)
                self.assertTrue(
                    any("at least three report images" in error for error in errors)
                )
                (assignment / "assets" / "memory-2048.png").write_bytes(b"png")

                forbidden = assignment / "submission" / "notes.txt"
                forbidden.write_text("not allowed\n")
                errors = []
                validate_repo.validate_a2p_submission(assignment, report, errors)
                self.assertTrue(
                    any("submission/profiling/**/*.py" in error for error in errors)
                )

                forbidden.unlink()
                (assignment / "assets" / "extra.png").write_bytes(b"123456")
                errors = []
                with mock.patch.object(
                    validate_repo, "A2P_MAX_ATTACHMENT_BYTES", 5
                ):
                    validate_repo.validate_a2p_submission(assignment, report, errors)
                self.assertTrue(
                    any("2 MiB attachment budget" in error for error in errors)
                )


if __name__ == "__main__":
    unittest.main()
