from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from pathlib import PurePosixPath
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_pr_scope  # noqa: E402
import validate_repo  # noqa: E402
from create_assignment import create_assignment  # noqa: E402


def make_summerquest(parent: Path) -> Path:
    root = parent / "SummerQuest-2026"
    student = root / "students" / "测试同学"
    student.mkdir(parents=True)
    (student / "PROFILE.md").write_text("profile\n")
    template = root / "students" / "_assignment_templates" / "A3"
    template.mkdir(parents=True)
    (template / "README.md").write_text("# A3 <姓名> <A编号>\n")
    (template / "requirements.txt").write_text("# standard library only\n")
    return root


def make_valid_assignment(root: Path) -> tuple[Path, str]:
    assignment = root / "students" / "测试同学" / "assignments" / "A3"
    (assignment / "analysis").mkdir(parents=True)
    (assignment / "results").mkdir()
    (assignment / "assets").mkdir()

    report = (
        "# A3\n\n"
        "![fit](assets/fit.png)\n\n"
        "![diagnostics](assets/diagnostics.png)\n"
    )
    (assignment / "README.md").write_text(report)
    (assignment / "requirements.txt").write_text(
        "numpy>=2.0,<3\nmatplotlib>=3.9,<4\n"
    )
    (assignment / "analysis" / "fit.py").write_text("print('fit')\n")

    experiment_path = assignment / "results" / "experiments.csv"
    with experiment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validate_repo.A3_EXPERIMENT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "exp-public-001",
                "hypothesis": "token scaling",
                "submitted_config": json.dumps({"model": {"hidden_size": 128}}),
                "resolved_config": json.dumps({"parameters": 1000}),
                "status": "completed",
                "reserved_seconds": 300,
                "used_seconds": 280,
                "validation_losses": json.dumps([4.0, 3.8]),
                "final_validation_loss": 3.8,
                "fit_role": "fit",
                "exclusion_reason": "",
            }
        )

    (assignment / "results" / "fit_summary.json").write_text(
        json.dumps(
            {
                "model_name": "power-law",
                "target": "final_validation_loss",
                "parameters": {"alpha": 0.1},
                "num_fit_runs": 1,
                "diagnostics": {"rmse": 0.01},
                "generated_at": "2026-08-25T00:00:00Z",
            }
        )
        + "\n"
    )
    (assignment / "results" / "final_prediction.json").write_text(
        json.dumps(
            {
                "predicted_final_loss": 3.25,
                "predicted_final_loss_lower": 3.18,
                "predicted_final_loss_upper": 3.36,
                "final_config_hash": "sha256:public-example",
                "analysis_version": "abc1234",
                "generated_at": "2026-08-25T00:00:00Z",
            }
        )
        + "\n"
    )
    (assignment / "assets" / "fit.png").write_bytes(b"png")
    (assignment / "assets" / "diagnostics.png").write_bytes(b"png")
    return assignment, report


class A3SubmissionToolsTests(unittest.TestCase):
    def test_pr_scope_accepts_a3_as_one_review_unit(self) -> None:
        student, label = validate_pr_scope.validate_scope(
            [
                PurePosixPath("students/测试同学/assignments/A3/README.md"),
                PurePosixPath(
                    "students/测试同学/assignments/A3/results/final_prediction.json"
                ),
            ],
            "[A3] 测试同学 - 完成 Scaling Laws 作业",
        )
        self.assertEqual(student, "测试同学")
        self.assertEqual(label, "[A3]")

    def test_create_assignment_uses_a3_template_and_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = make_summerquest(Path(temp_dir))
            assignment = create_assignment(root, "测试同学", "A3")

            self.assertIn("# A3 测试同学 A3", (assignment / "README.md").read_text())
            self.assertTrue((assignment / "requirements.txt").is_file())
            self.assertTrue((assignment / "analysis").is_dir())
            self.assertTrue((assignment / "results").is_dir())
            self.assertTrue((assignment / "assets").is_dir())

    def test_a3_validator_accepts_required_artifacts_and_dependency_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment, report = make_valid_assignment(root)

            with mock.patch.object(validate_repo, "ROOT", root):
                errors: list[str] = []
                validate_repo.validate_a3_submission(assignment, report, errors)
                self.assertEqual(errors, [])

                (assignment / "requirements.txt").unlink()
                (assignment / "pyproject.toml").write_text(
                    "[project]\nname='a3-analysis'\nversion='0.1.0'\n"
                )
                errors = []
                validate_repo.validate_a3_submission(assignment, report, errors)
                self.assertEqual(errors, [])

    def test_a3_validator_rejects_missing_or_ambiguous_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment, report = make_valid_assignment(root)

            (assignment / "results" / "experiments.jsonl").write_text(
                json.dumps({field: "value" for field in validate_repo.A3_EXPERIMENT_FIELDS})
                + "\n"
            )
            (assignment / "results" / "fit_summary.json").unlink()
            (assignment / "uv.lock").write_text("not allowed\n")

            with mock.patch.object(validate_repo, "ROOT", root):
                errors: list[str] = []
                validate_repo.validate_a3_submission(assignment, report, errors)

            self.assertTrue(any("exactly one" in error for error in errors))
            self.assertTrue(any("missing required A3 result" in error for error in errors))
            self.assertTrue(any("unexpected A3 top-level" in error for error in errors))

    def test_a3_validator_rejects_invalid_prediction_and_attachment_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            assignment, report = make_valid_assignment(root)
            prediction = assignment / "results" / "final_prediction.json"
            data = json.loads(prediction.read_text())
            data["predicted_final_loss_lower"] = 3.5
            prediction.write_text(json.dumps(data) + "\n")
            (assignment / "results" / "large.txt").write_bytes(
                b"x" * (validate_repo.A3_MAX_ATTACHMENT_BYTES + 1)
            )

            with mock.patch.object(validate_repo, "ROOT", root):
                errors: list[str] = []
                validate_repo.validate_a3_submission(assignment, report, errors)

            self.assertTrue(any("lower <= point <= upper" in error for error in errors))
            self.assertTrue(any("2 MiB attachment budget" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
