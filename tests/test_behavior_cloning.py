import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from virturoid.mvp import write_mvp_robot_arm_project

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class BehaviorCloningTests(unittest.TestCase):
    def test_bc_policy_learns_planner_and_drives_grasp(self):
        import mujoco

        from virturoid.services.behavior_cloning import BCPolicy, train_bc_policy
        from virturoid.services.pick_place_controller import run_pick_place_episode

        with tempfile.TemporaryDirectory() as tmpdir:
            out = write_mvp_robot_arm_project(Path(tmpdir) / "robot")
            result = train_bc_policy(out, samples=200, seed=11)

            # The policy learned from a meaningful number of expert demonstrations.
            self.assertGreater(result["demonstrations"], 50)
            self.assertLess(result["residual_rad"], 0.2)

            # Artifacts and a policy card are written.
            self.assertTrue((out / "software" / "controller" / "bc_policy.json").exists())
            self.assertTrue((out / "software" / "controller" / "bc_policy_card.json").exists())
            saved = json.loads((out / "software" / "controller" / "bc_policy.json").read_text(encoding="utf-8"))
            self.assertEqual("behavior_cloning_polynomial", saved["policy_type"])

            # The policy is reloadable and drives the task with no inference-time search.
            policy = BCPolicy.from_dict(saved)
            ss = json.loads((out / "simulation" / "baseline_scene_set.json").read_text(encoding="utf-8"))
            scene = ss["scenes"][0]
            model = mujoco.MjModel.from_xml_path(
                str(out / "simulation" / "mujoco" / "scenes" / "baseline" / f"{scene['id']}.xml")
            )
            objects = {o["name"]: tuple(o["pose_xyz_rpy"][:3]) for o in scene["objects"]}
            outcome = run_pick_place_episode(model, {"objects": objects}, target_policy=policy)
            self.assertEqual(2, outcome["block_count"])
            self.assertGreaterEqual(outcome["placed_count"], 1)


if __name__ == "__main__":
    unittest.main()
