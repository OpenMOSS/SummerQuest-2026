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

import a2k_source  # noqa: E402
import validate_pr_scope  # noqa: E402
import validate_repo  # noqa: E402
from create_assignment import create_assignment  # noqa: E402
from sync_a2k_submission import sync_submission  # noqa: E402


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_source(parent: Path) -> tuple[Path, str]:
    source = parent / "assignment2-systems"
    (source / "cs336_systems" / "a2k").mkdir(parents=True)
    (source / "cs336_systems" / "a2d").mkdir()
    (source / "tests").mkdir()
    (source / "student_scripts" / "a2k").mkdir(parents=True)
    (source / "local_results").mkdir()

    (source / "cs336_systems" / "__init__.py").write_text("BASE = True\n")
    (source / "cs336_systems" / "a2k" / "__init__.py").write_text("")
    (source / "cs336_systems" / "a2k" / "flash.py").write_text(
        "KERNEL = 'base'\n"
    )
    (source / "cs336_systems" / "a2k" / "notes.txt").write_text(
        "not submitted\n"
    )
    (source / "cs336_systems" / "a2d" / "ddp.py").write_text(
        "NOT_A2K = True\n"
    )
    (source / "tests" / "adapters.py").write_text("ADAPTER = True\n")
    (source / "tests" / "test_attention.py").write_text(
        "def test_public(): pass\n"
    )
    (source / "student_scripts" / "a2k" / "benchmark.py").write_text(
        "RUN = 1\n"
    )
    (source / "student_scripts" / "a2k" / "notes.md").write_text(
        "not submitted\n"
    )
    (source / "local_results" / "trace.json").write_text("{}\n")
    (source / "pyproject.toml").write_text(
        "[project]\nname='test-a2k'\nversion='0'\n"
    )

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
    template = root / "students" / "_assignment_templates" / "A2-K"
    template.mkdir(parents=True)
    (template / "README.md").write_text("# A2-K <姓名> <A编号>\n")
    return root


class A2KSubmissionToolsTests(unittest.TestCase):
    def test_pr_scope_accepts_a2k_as_one_review_unit(self) -> None:
        student, label = validate_pr_scope.validate_scope(
            [
                PurePosixPath(
                    "students/测试同学/assignments/A2-K/README.md"
                ),
                PurePosixPath(
                    "students/测试同学/assignments/A2-K/results/memory_evidence.json"
                ),
            ],
            "[A2-K] 测试同学 - 完成显存与 Kernel 作业",
        )
        self.assertEqual(student, "测试同学")
        self.assertEqual(label, "[A2-K]")

    def test_create_and_sync_copy_only_a2k_python_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, commit = make_source(parent)
            root = make_summerquest(parent)

            with mock.patch.object(a2k_source, "A2K_COMMIT", commit):
                assignment = create_assignment(root, "测试同学", "A2-K")
                submission = assignment / "submission"
                self.assertTrue(
                    (
                        submission
                        / "cs336_systems"
                        / "a2k"
                        / "flash.py"
                    ).is_file()
                )
                self.assertTrue((submission / "tests" / "adapters.py").is_file())
                self.assertTrue(
                    (
                        submission
                        / "student_scripts"
                        / "a2k"
                        / "benchmark.py"
                    ).is_file()
                )
                self.assertFalse(
                    (
                        submission
                        / "cs336_systems"
                        / "a2k"
                        / "notes.txt"
                    ).exists()
                )
                self.assertFalse(
                    (submission / "cs336_systems" / "a2d").exists()
                )
                self.assertFalse(
                    (submission / "tests" / "test_attention.py").exists()
                )
                self.assertFalse((submission / "local_results").exists())
                self.assertTrue((assignment / "results").is_dir())
                self.assertTrue((assignment / "assets").is_dir())

                (source / "cs336_systems" / "a2k" / "flash.py").write_text(
                    "KERNEL = 'updated'\n"
                )
                sync_submission(root, "测试同学")
                self.assertEqual(
                    (
                        submission
                        / "cs336_systems"
                        / "a2k"
                        / "flash.py"
                    ).read_text(),
                    "KERNEL = 'updated'\n",
                )

    def test_missing_sibling_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = make_summerquest(Path(temp_dir))
            with self.assertRaisesRegex(FileNotFoundError, "../assignment2-systems"):
                create_assignment(root, "测试同学", "A2-K")

    def test_a2k_validator_enforces_results_code_and_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment = root / "students" / "测试同学" / "assignments" / "A2-K"
            (assignment / "submission" / "cs336_systems" / "a2k").mkdir(
                parents=True
            )
            (assignment / "submission" / "tests").mkdir()
            (assignment / "submission" / "student_scripts" / "a2k").mkdir(
                parents=True
            )
            (assignment / "results").mkdir()
            (assignment / "assets").mkdir()

            report = (
                "# A2-K\n\n![memory](assets/memory.png)\n\n"
                "![speed](assets/speed.png)\n"
            )
            (assignment / "README.md").write_text(report)
            (
                assignment
                / "submission"
                / "cs336_systems"
                / "a2k"
                / "flash.py"
            ).write_text("KERNEL = True\n")
            (assignment / "submission" / "tests" / "adapters.py").write_text(
                "ADAPTER = True\n"
            )
            (
                assignment
                / "submission"
                / "student_scripts"
                / "a2k"
                / "benchmark.py"
            ).write_text("RUN = True\n")
            for relative in validate_repo.A2K_REQUIRED_FILES:
                path = assignment / relative
                if path.name == "memory_evidence.json":
                    path.write_text(
                        '{"allocator":{"allocator_fraction":0.96,'
                        '"allocator_limit_mib":23552},"hard_limit_mib":24576,'
                        '"pytorch_peak_allocated_mib":20000,'
                        '"pytorch_peak_reserved_mib":21000,'
                        '"within_24gib":true}\n'
                    )
                else:
                    path.write_text("{}\n" if path.suffix == ".json" else "value\n")
            (assignment / "assets" / "memory.png").write_bytes(b"png")
            (assignment / "assets" / "speed.png").write_bytes(b"png")

            with mock.patch.object(validate_repo, "ROOT", root):
                errors: list[str] = []
                validate_repo.validate_a2k_submission(assignment, report, errors)
                self.assertEqual(errors, [])

                forbidden = (
                    assignment
                    / "submission"
                    / "cs336_systems"
                    / "distributed.py"
                )
                forbidden.write_text("NOT_ALLOWED = True\n")
                errors = []
                validate_repo.validate_a2k_submission(assignment, report, errors)
                self.assertTrue(
                    any("unsupported A2-K submission file" in error for error in errors)
                )

                forbidden.unlink()
                (assignment / "results" / "large.txt").write_bytes(
                    b"x" * (validate_repo.A2K_MAX_ATTACHMENT_BYTES + 1)
                )
                errors = []
                validate_repo.validate_a2k_submission(assignment, report, errors)
                self.assertTrue(
                    any("2 MiB attachment budget" in error for error in errors)
                )

                (assignment / "results" / "large.txt").unlink()
                (assignment / "results" / "memory_evidence.json").write_text(
                    '{"allocator":{"allocator_fraction":0.96,'
                    '"allocator_limit_mib":23552},"hard_limit_mib":24576,'
                    '"pytorch_peak_allocated_mib":24000,'
                    '"pytorch_peak_reserved_mib":24000,'
                    '"within_24gib":true}\n'
                )
                errors = []
                validate_repo.validate_a2k_submission(assignment, report, errors)
                self.assertTrue(
                    any("peak reserved exceeds 23552 MiB" in error for error in errors)
                )

    def test_a2k_validator_rejects_cross_repository_relative_links(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment = root / "students" / "测试同学" / "assignments" / "A2-K"
            (assignment / "submission" / "cs336_systems" / "a2k").mkdir(
                parents=True
            )
            (assignment / "submission" / "tests").mkdir()
            (assignment / "submission" / "student_scripts" / "a2k").mkdir(
                parents=True
            )
            (assignment / "results").mkdir()
            (assignment / "assets").mkdir()
            report = (
                "# A2-K\n\n"
                "[bad](../../../../../../assignment2-systems/README.md)\n\n"
                "![memory](assets/memory.png)\n\n"
                "![speed](assets/speed.png)\n"
            )
            (assignment / "README.md").write_text(report)
            (
                assignment / "submission" / "cs336_systems" / "a2k" / "flash.py"
            ).write_text("KERNEL = True\n")
            (assignment / "submission" / "tests" / "adapters.py").write_text(
                "ADAPTER = True\n"
            )
            (
                assignment
                / "submission"
                / "student_scripts"
                / "a2k"
                / "benchmark.py"
            ).write_text("RUN = True\n")
            for relative in validate_repo.A2K_REQUIRED_FILES:
                path = assignment / relative
                if path.name == "memory_evidence.json":
                    path.write_text(
                        '{"allocator":{"allocator_fraction":0.96,'
                        '"allocator_limit_mib":23552},"hard_limit_mib":24576,'
                        '"pytorch_peak_allocated_mib":20000,'
                        '"pytorch_peak_reserved_mib":21000,'
                        '"within_24gib":true}\n'
                    )
                else:
                    path.write_text("{}\n" if path.suffix == ".json" else "value\n")
            (assignment / "assets" / "memory.png").write_bytes(b"png")
            (assignment / "assets" / "speed.png").write_bytes(b"png")

            with mock.patch.object(validate_repo, "ROOT", root):
                errors: list[str] = []
                validate_repo.validate_a2k_submission(assignment, report, errors)
                self.assertTrue(
                    any("GitHub HTTPS absolute URLs" in error for error in errors)
                )


if __name__ == "__main__":
    unittest.main()
