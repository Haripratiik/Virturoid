"""§32.5: every policy evaluated the same way — held-out, reproducible, with a Policy Card."""

import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.policy_card import PolicyCard
from virturoid.services.memory_db import MemoryDB

try:
    import mujoco  # noqa: F401
    import numpy as np
    _MUJOCO = True
except Exception:  # noqa: BLE001
    _MUJOCO = False


def _zero_policy_npz(path: Path):
    """A policy with zero weights -> mean action 0 -> ctrl == the (competent) scripted base. Lets the
    evaluator be tested deterministically without shipping the 390 KB trained params."""
    import numpy as np

    np.savez(path, logstd=np.zeros(5),
             pi_w0=np.zeros((34, 128)), pi_b0=np.zeros(128),
             pi_w1=np.zeros((128, 128)), pi_b1=np.zeros(128),
             pi_w2=np.zeros((128, 5)), pi_b2=np.zeros(5))


def _skill(params_path: str) -> dict:
    return {
        "skill_id": "grasp.test", "task_type": "grasp", "robot_class": "manipulator",
        "species": "fixed_arm.three_dof.tabletop.gripper", "gene_id": "test_gripper",
        "success_rate": 0.92, "region": {"x": [0.44, 0.48], "y": [-0.04, 0.04]},
        "base_config": {"kpj": 120.0, "kdj": 12.0, "kpf": 300.0, "kdf": 10.0, "alpha": 0.3,
                        "dq": 0.10, "phases": [0.15, 0.50, 0.65], "ep_len": 200, "fclose": 0.045},
        "obs_dim": 34, "act_dim": 5, "params_path": params_path,
    }


class PolicyCardSchemaTests(unittest.TestCase):
    def test_card_validation_flags_bad_fields(self):
        bad = PolicyCard(id="c", name="", task_type="", compatible_robots=[], success_rate=1.5)
        codes = {i.code for i in bad.validate().issues}
        self.assertIn("invalid_success_rate", codes)
        self.assertTrue({"required"} & codes)
        good = PolicyCard(id="c", name="grasp", task_type="grasp", compatible_robots=["r"], success_rate=0.9)
        self.assertTrue(good.validate().ok)


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class SkillEvaluatorTests(unittest.TestCase):
    def test_evaluate_skill_runs_the_32_5_suite_and_is_reproducible(self):
        from virturoid.services.skill_evaluator import evaluate_skill
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "p.npz"; _zero_policy_npz(npz)
            card, episodes, summary = evaluate_skill(_skill(str(npz)), baseline_n=1, randomized_n=1)
            cats = {c.category for c in card.category_results}
            self.assertEqual(cats, {"baseline", "randomized", "known_failure"})   # §32.5 categories
            self.assertEqual(len(episodes), 1 + 1 + 4)                            # one record per scene
            self.assertTrue(card.validate().ok)
            self.assertTrue(card.reproducible)                                    # deterministic CPU MuJoCo
            self.assertGreaterEqual(summary["categories"]["baseline"]["success"], 0.5)  # base grasps
            self.assertTrue(0.0 <= summary["headline_success"] <= 1.0)
            # failures are SPECIFIC (§26): out-of-region scenes fail with an actionable reason, clustered
            valid_reasons = {"no_grasp_contact", "gripped_no_lift", "lifted_then_dropped", "low_lift"}
            for cl in card.failure_clusters:
                self.assertIn(cl["reason"], valid_reasons)
                self.assertGreaterEqual(cl["count"], 1)
                self.assertTrue(cl["example_scenes"])

    def test_evaluate_and_record_writes_artifacts_and_memory(self):
        from virturoid.services.skill_evaluator import evaluate_and_record
        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "p.npz"; _zero_policy_npz(npz)
            out = Path(tmp) / "eval"
            with MemoryDB(Path(tmp) / "mem.db") as db:
                # bank a stub so record_skill stamping has a row, then evaluate
                sk = _skill(str(npz))
                db.record_skill(sk["skill_id"], "manipulator", "grasp", success_rate=0.92,
                                params_path=str(npz), species=sk["species"], gene_id=sk["gene_id"],
                                region=sk["region"], base_config=sk["base_config"], obs_dim=34, act_dim=5)
                card, summary = evaluate_and_record(sk, out, memory=db, baseline_n=1, randomized_n=1)
                self.assertTrue((out / "policy_card.json").exists())
                self.assertTrue((out / "evaluation_episodes.jsonl").exists())
                self.assertTrue((out / "summary.json").exists())
                self.assertEqual(db.stats()["runs"], 1)                            # eval recorded as a run
                stamped = db.recall_skill("manipulator", "grasp")
                self.assertIn("held-out eval", stamped["notes"])                   # bank stamped honestly


if __name__ == "__main__":
    unittest.main()
