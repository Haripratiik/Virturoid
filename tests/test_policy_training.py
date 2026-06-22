import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.mvp import write_mvp_robot_arm_project

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class PolicyTrainingTests(unittest.TestCase):
    def test_training_improves_reward_and_exports_runnable_controller(self):
        from virturoid.services.controller_exporter import export_controller_bundle
        from virturoid.services.policy_trainer import train_arm_controller_from_export

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "mvp_robot_arm")

            # Small search keeps the test fast; we assert learning + contract, not full success.
            policy = train_arm_controller_from_export(
                output_dir, iterations=5, candidates=10, horizon=220
            )

            self.assertTrue(policy.validate().ok, policy.validate().issues)
            self.assertEqual(3, len(policy.action_dimensions))
            self.assertEqual(["base_yaw", "shoulder_pitch", "elbow_pitch"], policy.joint_names)
            # CEM never returns a controller worse than the starting point.
            self.assertGreaterEqual(policy.training.best_reward, policy.training.initial_reward)
            self.assertGreater(policy.control_frequency_hz, 0)
            # Part-derived effort limit flows into the action space.
            self.assertGreater(policy.action_dimensions[0].effort_limit, 0)

            # Trained policy + metrics artifacts exist.
            self.assertTrue((output_dir / "software" / "controller" / "trained_policy.json").exists())
            self.assertTrue((output_dir / "training" / "policy_training_metrics.json").exists())

            # Replayable eval episodes were written.
            trace = (output_dir / "runs" / "mvp_policy_eval" / "logs" / "episode_trace.jsonl").read_text(encoding="utf-8")
            episodes = [json.loads(line) for line in trace.splitlines() if line.strip()]
            self.assertGreaterEqual(len(episodes), 1)
            self.assertIn("reach_distance_m", episodes[0])
            self.assertTrue((output_dir / episodes[0]["rollout_uri"]).exists())

            # Controller export is inference-ready and standalone (no virturoid/mujoco import).
            bundle_path = export_controller_bundle(output_dir, policy)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual({"base_yaw": 0, "shoulder_pitch": 1, "elbow_pitch": 2}, bundle["joint_mapping"])
            self.assertEqual(3, len(bundle["action_outputs"]))

            params = json.loads((output_dir / "software" / "controller" / "policy_params.json").read_text(encoding="utf-8"))
            spec = importlib.util.spec_from_file_location(
                "exported_controller", output_dir / "software" / "controller" / "controller.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            controller = module.ReachController(params)
            targets = controller.infer([0.4, -0.1])
            self.assertEqual({"base_yaw", "shoulder_pitch", "elbow_pitch"}, set(targets))
            # Outputs respect the joint position clamps.
            self.assertGreaterEqual(targets["shoulder_pitch"], -1.57)
            self.assertLessEqual(targets["shoulder_pitch"], 1.57)


if __name__ == "__main__":
    unittest.main()
