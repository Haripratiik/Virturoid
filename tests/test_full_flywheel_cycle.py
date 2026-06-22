"""Skill-flywheel WRITE loop: synthesize a SKILL_CARD from a TrainedPolicy, bank it, and append
the auditable manifest row — the half that closes ``autonomous_build(train=True)`` so the next
build can warm-start from a banked skill (plan §13 memory, §32 training)."""

import json
import tempfile
import unittest
from pathlib import Path

from virturoid.schemas.trained_policy import (
    ActionDimension,
    PolicyEvalMetrics,
    TrainedPolicy,
)
from virturoid.services.memory_db import MemoryDB
from virturoid.services.skill_flywheel import (
    append_manifest,
    bank_skill_from_card,
    synthesize_skill_card,
    warm_start_for,
)


def _policy(success: float = 0.85) -> TrainedPolicy:
    return TrainedPolicy(
        id="policy_test",
        name="reach_ctrl",
        robot_genome_id="tabletop_arm",
        observation_space=["tx", "ty", "tz"],
        joint_names=["j1", "j2", "j3"],
        action_dimensions=[ActionDimension(f"j{i}", -1.0, 1.0) for i in range(3)],
        control_frequency_hz=50.0,
        low_level_control_frequency_hz=500.0,
        weights=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        input_features=["tx", "ty", "tz"],
        safety_clamps={"max_dq": 0.2},
        evaluation=PolicyEvalMetrics(eval_scene_count=8, success_rate=success),
    )


class FullFlywheelCycleTests(unittest.TestCase):
    def test_card_is_honest_about_the_cem_reach_trainer(self):
        card = synthesize_skill_card(_policy(0.85), "manipulator", "reach",
                                     species="manipulator.tabletop.arm3", gene_id="g1",
                                     params_path="build/out/trained_policy.json")
        self.assertEqual(card["reward_spec"], {})            # cem_reach has no reward recipe
        self.assertIsNone(card["region"])                    # reachable region not computed
        self.assertIsNone(card["params_path"])               # cem_reach cannot reuse its own weights
        self.assertEqual(card["artifact_path"], "build/out/trained_policy.json")  # but recorded for provenance
        self.assertEqual(card["obs_dim"], 3)
        self.assertEqual(card["act_dim"], 3)
        self.assertEqual(card["base_config"]["trainer"], "cem_reach")

    def test_param_reuse_trainer_keeps_params_path(self):
        card = synthesize_skill_card(_policy(0.9), "manipulator", "grasp",
                                     params_path="build/skills/grasp.npz", trainer="mjx_ppo")
        self.assertEqual(card["params_path"], "build/skills/grasp.npz")  # param-mode trainer banks weights

    def test_cem_reach_skill_warm_starts_in_config_mode(self):
        with tempfile.TemporaryDirectory() as tmp, MemoryDB(Path(tmp) / "mem.db") as db:
            card = synthesize_skill_card(_policy(0.85), "manipulator", "reach",
                                         species="manipulator.tabletop.arm3",
                                         params_path="build/out/trained_policy.json")
            sid = bank_skill_from_card(db, card)
            self.assertEqual(db.stats()["skills"], 1)
            ws = warm_start_for(db, "manipulator", "reach", species="manipulator.tabletop.arm3")
            self.assertEqual(ws["skill_id"], sid)
            self.assertEqual(ws["warm_start"], "config")     # no params -> config reuse, never a false param claim

    def test_append_manifest_writes_one_auditable_jsonl_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            mem = Path(tmp) / "memory"
            row = {"skill_id": "reach__x", "success_rate": 0.85, "n_episodes": 8,
                   "ckpt_hash": "abc123", "cycle": 1, "warm_started": False}
            path = append_manifest(mem, row)
            append_manifest(mem, {**row, "cycle": 2, "warm_started": True})
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "flywheel_manifest.jsonl")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)                  # appends, does not overwrite
            self.assertEqual(json.loads(lines[1])["warm_started"], True)


if __name__ == "__main__":
    unittest.main()
