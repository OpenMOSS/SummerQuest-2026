from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import a1_source  # noqa: E402
from create_assignment import create_assignment  # noqa: E402
from sync_a1_submission import sync_submission  # noqa: E402


def git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_source(parent: Path) -> tuple[Path, str, str]:
    source = parent / "assignment1-basics"
    (source / "cs336_basics").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "cs336_basics" / "__init__.py").write_text("BASE = True\n")
    (source / "tests" / "adapters.py").write_text("BASE = True\n")
    (source / "tests" / "test_public.py").write_text("def test_public(): pass\n")
    (source / "uv.lock").write_text("locked\n")
    git(source, "init", "-q")
    git(source, "config", "user.name", "Test User")
    git(source, "config", "user.email", "test@example.com")
    git(source, "add", ".")
    git(source, "commit", "-q", "-m", "starter")
    return (
        source,
        git(source, "rev-parse", "HEAD"),
        git(source, "rev-parse", "HEAD^{tree}"),
    )


def make_summerquest(parent: Path) -> Path:
    root = parent / "SummerQuest-2026"
    student = root / "students" / "测试同学"
    student.mkdir(parents=True)
    (student / "PROFILE.md").write_text("profile\n")
    template = root / "students" / "_assignment_templates" / "A1"
    template.mkdir(parents=True)
    (template / "README.md").write_text("# A1 <姓名> <A编号>\n")
    return root


class A1SubmissionToolsTests(unittest.TestCase):
    def test_create_and_sync_use_only_the_fixed_sibling_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            source, commit, tree = make_source(parent)
            root = make_summerquest(parent)

            (source / "cs336_basics" / "implementation.py").write_text("VALUE = 1\n")
            (source / "tests" / "adapters.py").write_text("STUDENT = True\n")
            (source / "scripts").mkdir()
            (source / "scripts" / "train.py").write_text("print('train')\n")
            (source / "configs").mkdir()
            (source / "configs" / "tiny.json").write_text("{}\n")
            (source / "data").mkdir()
            (source / "data" / "large.txt").write_text("not submitted\n")

            with (
                mock.patch.object(a1_source, "A1_COMMIT", commit),
                mock.patch.object(a1_source, "A1_TREE", tree),
            ):
                assignment = create_assignment(root, "测试同学", "A1")

                self.assertEqual(
                    (assignment / "submission" / "tests" / "adapters.py").read_text(),
                    "STUDENT = True\n",
                )
                self.assertTrue(
                    (
                        assignment / "submission" / "cs336_basics" / "implementation.py"
                    ).is_file()
                )
                self.assertTrue(
                    (assignment / "submission" / "scripts" / "train.py").is_file()
                )
                self.assertTrue(
                    (assignment / "submission" / "configs" / "tiny.json").is_file()
                )
                self.assertFalse(
                    (assignment / "submission" / "tests" / "test_public.py").exists()
                )
                self.assertFalse((assignment / "submission" / "data").exists())
                self.assertFalse((assignment / "submission" / "uv.lock").exists())

                (source / "cs336_basics" / "implementation.py").write_text(
                    "VALUE = 2\n"
                )
                sync_submission(root, "测试同学")
                self.assertEqual(
                    (
                        assignment / "submission" / "cs336_basics" / "implementation.py"
                    ).read_text(),
                    "VALUE = 2\n",
                )

    def test_missing_sibling_repository_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = make_summerquest(Path(temp_dir))
            with self.assertRaisesRegex(FileNotFoundError, "../assignment1-basics"):
                create_assignment(root, "测试同学", "A1")


if __name__ == "__main__":
    unittest.main()
