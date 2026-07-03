import importlib.util
import tempfile
import unittest
from pathlib import Path

from virturoid.mobile_base import build_mobile_base_project

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_PROMPT = "Build a mobile robot base that can drive and navigate indoors to deliver parts."


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class NavigationTaskLoopTests(unittest.TestCase):
    def test_mobile_base_navigates_to_goals(self):
        from virturoid.services.navigation_controller import run_navigation_evaluation

        with tempfile.TemporaryDirectory() as tmpdir:
            project = build_mobile_base_project(_PROMPT, Path(tmpdir) / "mobile")
            report = run_navigation_evaluation(project["output_dir"])

            self.assertEqual("navigation", report["task"])
            self.assertGreater(report["total_episodes"], 0)
            self.assertGreaterEqual(report["success_rate"], 0.75)
            self.assertLessEqual(report["success_rate"], 1.0)
            # The legacy two-wheel package must clear most route scenes, not merely move in one easy scene.
            self.assertGreaterEqual(report["reached"], 3)
            self.assertTrue((project["output_dir"] / "reports" / "navigation_evaluation_report.json").exists())

    def test_composed_rover_navigates_omnidirectionally(self):
        """A mobile base COMPOSED from blocks (free-base 4-wheel rover) navigates to goals in any
        direction — the navigation evaluator auto-calibrates the drive signs + drives all wheels grouped
        by side, so task-matched evaluation works for composed mobile bases, not just the legacy builder."""
        import mujoco
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.navigation_controller import run_navigation_episode

        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(compose_robot("a mobile base to deliver parts indoors")))
        goals = [(1.0, 0.0), (0.0, -0.9), (-0.7, 0.4)]    # forward, sideways, behind
        reached = sum(run_navigation_episode(mj, g, [], horizon=3000)["status"] == "reached" for g in goals)
        self.assertGreaterEqual(reached, 2)               # reaches goals in multiple directions

    def test_autonomous_build_runs_navigation_task_loop_for_mobile(self):
        from virturoid.services.autonomous_build import autonomous_build

        with tempfile.TemporaryDirectory() as tmpdir:
            report = autonomous_build(
                _PROMPT, Path(tmpdir) / "mobile", target_success_rate=0.7, memory_dir=Path(tmpdir) / "mem"
            )
            self.assertEqual("mobile_base", report.robot_class)
            # The mobile base gets a real navigation task loop, not just a foundation package.
            self.assertTrue(any(d.stage == "navigate" for d in report.decisions))
            self.assertTrue(report.succeeded)
            self.assertGreaterEqual(report.final_success_rate, report.target_success_rate)
            self.assertTrue(report.compute.get("physics_executed"))


if __name__ == "__main__":
    unittest.main()
