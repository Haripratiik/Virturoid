"""§41 / §32.3: demonstration dataset system — record expert rollouts, write a card, learn from them."""

import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.demonstration import DatasetCard

try:
    import mujoco  # noqa: F401
    import numpy as np
    _MUJOCO = True
except Exception:  # noqa: BLE001
    _MUJOCO = False


def _zero_policy_npz(path: Path):
    import numpy as np
    np.savez(path, logstd=np.zeros(5),
             pi_w0=np.zeros((34, 128)), pi_b0=np.zeros(128),
             pi_w1=np.zeros((128, 128)), pi_b1=np.zeros(128),
             pi_w2=np.zeros((128, 5)), pi_b2=np.zeros(5))


def _skill(params_path: str) -> dict:
    return {
        "skill_id": "grasp.test", "task_type": "grasp", "species": "fixed_arm.three_dof.tabletop.gripper",
        "gene_id": "test_gripper", "region": {"x": [0.44, 0.48], "y": [-0.04, 0.04]},
        "base_config": {"kpj": 120.0, "kdj": 12.0, "kpf": 300.0, "kdf": 10.0, "alpha": 0.3,
                        "dq": 0.10, "phases": [0.15, 0.50, 0.65], "ep_len": 200, "fclose": 0.045},
        "obs_dim": 34, "act_dim": 5, "params_path": params_path,
    }


class DatasetCardSchemaTests(unittest.TestCase):
    def test_card_validation(self):
        bad = DatasetCard(id="d", robot="", task="", sources=[], episodes=0, success_rate=2.0)
        codes = {i.code for i in bad.validate().issues}
        self.assertIn("no_episodes", codes)
        self.assertIn("invalid_success_rate", codes)
        good = DatasetCard(id="d", robot="arm", task="grasp", sources=["simulation"], episodes=3,
                           success_rate=0.9)
        self.assertTrue(good.validate().ok)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class DemonstrationDatasetTests(unittest.TestCase):
    def test_record_write_load_and_learn(self):
        from virturoid.services.demonstration_dataset import (
            record_grasp_demonstrations, write_demonstration_dataset,
            load_demonstration_dataset, fit_bc_baseline)
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "p.npz"; _zero_policy_npz(npz)
            eps = record_grasp_demonstrations(_skill(str(npz)), n=3, seed=2)
            self.assertGreaterEqual(len(eps), 1)                       # base grasps -> ≥1 expert demo
            self.assertEqual(eps[0]["obs"].shape[1], 34)
            self.assertEqual(eps[0]["actions"].shape[1], 5)

            out = Path(tmp) / "ds"
            card = write_demonstration_dataset(eps, out, robot="arm", task="grasp",
                                               obs_modalities=["proprioception"], action_space="torque[5]")
            self.assertTrue(card.validate().ok)
            self.assertTrue((out / "dataset_card.json").exists())
            self.assertTrue((out / "index.json").exists())
            self.assertEqual(card.episodes, len(eps))
            self.assertIn("virturoid_npz", card.formats)

            data = load_demonstration_dataset(out)
            self.assertEqual(data["obs"].shape[1], 34)
            self.assertEqual(data["actions"].shape[1], 5)
            self.assertEqual(len(data["episode_lengths"]), len(eps))

            bc = fit_bc_baseline(out)                                  # demonstrations are learnable
            self.assertEqual(bc["obs_dim"], 34)
            self.assertTrue(np.isfinite(bc["val_mse"]))

    def test_write_rejects_shape_mismatch(self):
        from virturoid.services.demonstration_dataset import write_demonstration_dataset
        import numpy as np
        ep_a = {"episode_id": "a", "success": True, "scene_params": {}, "task_language": "",
                "robot_genome_id": "r", "source": "simulation", "return": 0.0,
                "obs": np.zeros((5, 34)), "actions": np.zeros((5, 5)), "rewards": np.zeros(5),
                "robot_qpos": np.zeros((5, 5)), "robot_qvel": np.zeros((5, 5)), "object_pose": np.zeros((5, 7))}
        ep_b = dict(ep_a); ep_b["episode_id"] = "b"; ep_b["obs"] = np.zeros((5, 99))  # wrong obs dim
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_demonstration_dataset([ep_a, ep_b], Path(tmp) / "ds", robot="arm", task="grasp",
                                            obs_modalities=["x"], action_space="t")


if __name__ == "__main__":
    unittest.main()
