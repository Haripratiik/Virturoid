"""A genuinely unknown request must stop before the asynchronous build can choose a default body."""

import tempfile
import unittest
from pathlib import Path


class IntentClarificationTests(unittest.TestCase):
    def test_autonomous_build_returns_a_valid_clarification_report_without_robot_artifacts(self):
        from virturoid.services.autonomous_build import autonomous_build

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "unconfirmed"
            report = autonomous_build("build a blorptron", out, memory_dir=Path(tmp) / "memory")
            self.assertTrue(report.validate().ok, report.validate().issues)
            self.assertFalse(report.feasible)
            self.assertFalse(report.succeeded)
            self.assertEqual("clarify_intent", report.decisions[0].stage)
            self.assertFalse((out / "robot" / "robot_genome.json").exists())
            self.assertTrue((out / "reports" / "autonomy_report.json").exists())

    def test_job_result_exposes_clarification_instead_of_an_active_package(self):
        from virturoid.services.job_registry import _run_autonomous_build

        with tempfile.TemporaryDirectory() as tmp:
            result = _run_autonomous_build(
                {"args": {"prompt": "build a blorptron"}, "cancel_requested": False, "events": []},
                Path(tmp),
            )
            self.assertTrue(result["requires_clarification"])
            self.assertIsNone(result["output_name"])
            self.assertIn("clarif", result["clarification"].lower())


if __name__ == "__main__":
    unittest.main()
