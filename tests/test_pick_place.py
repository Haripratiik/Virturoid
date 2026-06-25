import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.mvp import write_mvp_robot_arm_project

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class PickPlaceEvaluationTests(unittest.TestCase):
    def test_real_pick_place_places_blocks_and_clusters_failures(self):
        from virturoid.services.physics_evaluator import run_physics_pick_place_evaluation

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "mvp_robot_arm")

            report = run_physics_pick_place_evaluation(
                output_dir, scene_uris=["simulation/baseline_scene_set.json"]
            )

            self.assertTrue(report.validate().ok, report.validate().issues)
            self.assertEqual("scripted_pick_place", report.controller)
            self.assertGreater(report.total_episodes, 0)
            self.assertEqual(report.total_episodes, len(report.episodes))
            self.assertGreaterEqual(report.success_rate, 0.0)
            self.assertLessEqual(report.success_rate, 1.0)
            # The controller must actually place at least one block in a matching bin.
            self.assertGreater(report.blocks_placed, 0)

            # Report and episode trace are written and reloadable.
            report_path = output_dir / "reports" / "physics_evaluation_report.json"
            self.assertTrue(report_path.exists())
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report.total_episodes, saved["total_episodes"])
            self.assertEqual("idealized_pin", saved["grasp_model"])
            gap_path = output_dir / "reports" / "manipulation_fidelity_gap_report.json"
            self.assertTrue(gap_path.exists())
            gap = json.loads(gap_path.read_text(encoding="utf-8"))
            self.assertTrue(gap["contact_grasp_certified"])
            self.assertIsNone(gap["export_blocker"])
            self.assertGreaterEqual(gap["contact_grasp_success_rate"], 0.5)
            self.assertEqual("idealized_pin", gap["grasp_model"])
            grasp_path = output_dir / "reports" / "grasp_evaluation_report.json"
            self.assertTrue(grasp_path.exists())
            grasp = json.loads(grasp_path.read_text(encoding="utf-8"))
            self.assertEqual("contact", grasp["grasp_model"])
            self.assertGreaterEqual(grasp["success_rate"], 0.5)
            self.assertGreater(grasp["mean_lift_m"], 0.03)
            from virturoid.services.readiness_ledger import build_product_readiness_ledger
            ledger = build_product_readiness_ledger(output_dir, robot_class="manipulator")
            self.assertTrue(ledger.safe_to_export)
            self.assertEqual("attained", ledger.by_stage["physics_evaluated"].status)
            trace = (output_dir / "runs" / "mvp_pick_place_eval" / "logs" / "episode_trace.jsonl").read_text(encoding="utf-8")
            rows = [json.loads(line) for line in trace.splitlines() if line.strip()]
            self.assertEqual(report.total_episodes, len(rows))
            self.assertIn("status", rows[0])

            # Every non-success episode contributes to a failure cluster with a regression idea.
            failures = sum(c.count for c in report.failure_clusters)
            self.assertEqual(report.total_episodes - report.success_count, failures)
            for cluster in report.failure_clusters:
                self.assertTrue(cluster.suggested_regression)

    def test_controller_succeeds_on_a_baseline_scene(self):
        import mujoco

        from virturoid.services.pick_place_controller import run_pick_place_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "mvp_robot_arm")
            scene_set = json.loads((output_dir / "simulation" / "baseline_scene_set.json").read_text(encoding="utf-8"))
            scene = scene_set["scenes"][0]
            xml = output_dir / "simulation" / "mujoco" / "scenes" / "baseline" / f"{scene['id']}.xml"
            model = mujoco.MjModel.from_xml_path(str(xml))
            objects = {o["name"]: tuple(o["pose_xyz_rpy"][:3]) for o in scene["objects"]}

            outcome = run_pick_place_episode(model, {"objects": objects})

            self.assertEqual(2, outcome["block_count"])
            # The baseline scene is the easy case; the arm should place both blocks.
            self.assertEqual("success", outcome["status"])
            self.assertEqual(2, outcome["placed_count"])


if __name__ == "__main__":
    unittest.main()
