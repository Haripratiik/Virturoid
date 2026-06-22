import json
import tempfile
import unittest
from pathlib import Path

from virturoid.mvp import write_mvp_robot_arm_project
from virturoid.services.mujoco_runner import mujoco_available, run_physics_rollout_from_export


@unittest.skipUnless(mujoco_available(), "MuJoCo Python package is not installed.")
class MujocoRunnerTests(unittest.TestCase):
    def test_physics_rollout_produces_real_traces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "mvp_robot_arm")

            result = run_physics_rollout_from_export(output_dir, control_horizon=200)

            self.assertTrue(result.validate().ok, result.validate().issues)
            self.assertEqual("mujoco.physics_v0", result.adapter_name)
            self.assertGreater(result.total_episodes, 0)
            self.assertEqual(result.total_episodes, len(result.episodes))
            self.assertGreaterEqual(result.measured_success_rate, 0.0)
            self.assertLessEqual(result.measured_success_rate, 1.0)

            # Result summary artifact is written and reloadable.
            result_path = output_dir / "runs" / "mvp_physics" / "physics_run_result.json"
            self.assertTrue(result_path.exists())
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result.total_episodes, saved["total_episodes"])

            # Episode trace carries real physics metrics, not synthetic constants.
            trace_lines = [
                json.loads(line)
                for line in (output_dir / result.episode_trace_uri).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(result.total_episodes, len(trace_lines))
            first = trace_lines[0]
            self.assertEqual("mujoco", first["backend"])
            self.assertIn("steps_simulated", first["metrics"])
            self.assertGreater(first["metrics"]["steps_simulated"], 0)
            self.assertIn("max_joint_speed_rad_s", first["metrics"])

            # Per-episode rollout file exists and records evolving joint state.
            rollout_path = output_dir / first["rollout_uri"]
            self.assertTrue(rollout_path.exists())
            samples = [
                json.loads(line)
                for line in rollout_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertGreaterEqual(len(samples), 2)
            self.assertIn("qpos", samples[0])
            self.assertIn("ctrl", samples[0])
            # Physics actually advanced between the first and last recorded sample.
            self.assertLess(samples[0]["t"], samples[-1]["t"])

    def test_physics_adapter_implements_simulator_protocol(self):
        from virturoid.services.mujoco_runner import MujocoSimulatorAdapter
        from virturoid.services.simulator_adapter import EpisodeBatchRequest

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "mvp_robot_arm")
            compiled_index = json.loads(
                (output_dir / "simulation" / "mujoco" / "compiled_scene_index.json").read_text(encoding="utf-8")
            )
            entry = compiled_index["scenes"][0]
            adapter = MujocoSimulatorAdapter(output_dir, control_horizon=100)

            traces = adapter.run_episode_batch(
                EpisodeBatchRequest(
                    backend="mujoco",
                    adapter_name=adapter.name,
                    purpose=entry["purpose"],
                    scene_artifact="simulation/scene_set.json",
                    scenes=[{"id": entry["scene_id"]}],
                    planned_episodes=2,
                    policy_id="policy_test",
                    start_index=1,
                    target_success_rate=0.0,
                    observation_keys=["joint_positions"],
                    action_trace=["observe_scene", "execute_place"],
                    compiled_scene_xml_by_scene_id={entry["scene_id"]: entry["mujoco_xml"]},
                )
            )

            self.assertEqual(2, len(traces))
            self.assertEqual("mujoco.physics_v0", traces[0].adapter_name)
            self.assertTrue(traces[0].rollout_uri)
            self.assertIn("stable", traces[0].metrics)


if __name__ == "__main__":
    unittest.main()
