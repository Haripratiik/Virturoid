import unittest
import tempfile
import json
from pathlib import Path

from virturoid.mvp import write_mvp_robot_arm_project
from virturoid.services.simulator_contract_builder import build_simulator_contract_from_export
from virturoid.services.simulator_adapter import DryRunSimulatorAdapter, EpisodeBatchRequest, default_simulator_adapter


class SimulatorAdapterTests(unittest.TestCase):
    def test_dry_run_adapter_generates_runner_shaped_episode_traces(self):
        adapter = DryRunSimulatorAdapter()

        traces = adapter.run_episode_batch(
            EpisodeBatchRequest(
                backend="mujoco",
                adapter_name=adapter.name,
                purpose="baseline",
                scene_artifact="simulation/baseline_scene_set.json",
                scenes=[{"id": "scene_a"}, {"id": "scene_b"}],
                planned_episodes=3,
                policy_id="policy_test",
                start_index=1,
                target_success_rate=0.9,
                observation_keys=["object_poses", "joint_positions", "wrist_rgbd"],
                action_trace=["observe_scene", "plan_grasp", "place_object"],
                compiled_scene_xml_by_scene_id={
                    "scene_a": "simulation/mujoco/scenes/baseline/scene_a.xml",
                    "scene_b": "simulation/mujoco/scenes/baseline/scene_b.xml",
                },
            )
        )

        self.assertEqual(3, len(traces))
        self.assertEqual("episode_baseline_0001", traces[0].episode_id)
        self.assertEqual("mujoco", traces[0].backend)
        self.assertEqual("dry_run.synthetic_v0", traces[0].adapter_name)
        self.assertEqual("scene_b", traces[1].scene_id)
        self.assertEqual("simulation/mujoco/scenes/baseline/scene_b.xml", traces[1].compiled_scene_xml_uri)
        self.assertIn("wrist_rgbd", traces[0].observation_keys)
        self.assertIn("place_object", traces[0].action_trace)

    def test_default_simulator_adapter_returns_dry_run_adapter_for_mvp_backend(self):
        adapter = default_simulator_adapter("mujoco")

        self.assertEqual("dry_run.synthetic_v0", adapter.name)

    def test_simulator_contract_reports_backend_capabilities_and_scene_xml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = write_mvp_robot_arm_project(Path(tmpdir) / "mvp_robot_arm")
            contract = build_simulator_contract_from_export(output_dir)

            self.assertTrue(contract.validate().ok)
            self.assertEqual("mujoco", contract.backend)
            self.assertGreaterEqual(len(contract.scene_contracts), 20)   # >=20-scene bar (edge_case added)
            self.assertTrue(all(scene.xml_parse_status == "pass" for scene in contract.scene_contracts))
            capabilities = {item.backend: item for item in contract.backend_capabilities}
            self.assertTrue(capabilities["dry_run"].available)
            self.assertIn(capabilities["mujoco"].status, {"real_backend_available", "not_installed"})
            saved_contract = json.loads((output_dir / "simulation" / "simulator_contract.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(saved_contract["scene_contracts"]), 20)


if __name__ == "__main__":
    unittest.main()
