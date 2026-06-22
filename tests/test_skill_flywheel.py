"""M3 G4: the skill flywheel — bank trained policies + warm-start new builds from them."""

import json
import tempfile
import unittest
from pathlib import Path

from virturoid.services.memory_db import MemoryDB
from virturoid.services.skill_flywheel import bank_skill_from_card, skill_id_for, warm_start_for

# a representative SKILL_CARD as emitted by scripts/mjx_residual_grasp.py --save
_GRASP_CARD = {
    "task_type": "grasp", "robot_class": "manipulator",
    "species": "manipulator.tabletop.arm3.gripper", "gene_id": "tabletop_arm__gripper",
    "success_rate": 0.94, "region": {"x": [0.43, 0.49], "y": [-0.08, 0.08]},
    "base_config": {"kpj": 120.0, "alpha": 0.3, "phases": [0.15, 0.5, 0.65]},
    "reward_spec": {"success_gate": "both & lifted>0.05 & d_tcp<0.06"},
    "obs_dim": 34, "act_dim": 5, "params_path": "build/skills/grasp.npz",
}


class SkillFlywheelTests(unittest.TestCase):
    def _db(self, tmp):
        return MemoryDB(Path(tmp) / "mem.db")

    def test_bank_and_recall_round_trips_the_skill(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            sid = bank_skill_from_card(db, _GRASP_CARD)
            self.assertEqual(sid, skill_id_for("grasp", _GRASP_CARD["species"]))
            self.assertEqual(db.stats()["skills"], 1)
            sk = db.recall_skill("manipulator", "grasp")
            self.assertEqual(sk["params_path"], "build/skills/grasp.npz")
            self.assertEqual(sk["region"]["x"], [0.43, 0.49])           # JSON round-tripped
            self.assertEqual(sk["obs_dim"], 34)

    def test_bank_accepts_the_raw_skill_card_line(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            bank_skill_from_card(db, "SKILL_CARD " + json.dumps(_GRASP_CARD))
            self.assertEqual(db.stats()["skills"], 1)

    def test_record_skill_keeps_best_success(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            bank_skill_from_card(db, {**_GRASP_CARD, "success_rate": 0.94})
            bank_skill_from_card(db, {**_GRASP_CARD, "success_rate": 0.80})   # worse — must not overwrite
            self.assertEqual(db.stats()["skills"], 1)                          # idempotent on skill_id
            self.assertAlmostEqual(db.recall_skill("manipulator", "grasp")["success_rate"], 0.94)
            bank_skill_from_card(db, {**_GRASP_CARD, "success_rate": 0.97})   # better — updates
            self.assertAlmostEqual(db.recall_skill("manipulator", "grasp")["success_rate"], 0.97)

    def test_warm_start_prefers_io_compatible_params(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            bank_skill_from_card(db, _GRASP_CARD)
            ws = warm_start_for(db, "manipulator", "grasp", species=_GRASP_CARD["species"],
                                obs_dim=34, act_dim=5)
            self.assertEqual(ws["warm_start"], "params")          # exact + IO match -> direct param reuse
            self.assertEqual(ws["params_path"], "build/skills/grasp.npz")

    def test_warm_start_falls_back_to_config_on_io_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            bank_skill_from_card(db, _GRASP_CARD)
            ws = warm_start_for(db, "manipulator", "grasp", obs_dim=99, act_dim=7)  # different IO
            self.assertEqual(ws["warm_start"], "config")          # reuse reward recipe + region + base cfg
            self.assertEqual(ws["base_config"]["kpj"], 120.0)

    def test_warm_start_cross_body_transfer_candidate(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            bank_skill_from_card(db, _GRASP_CARD)
            # a related body (shares 'gripper'/'arm' species tokens) with no exact class+task skill banked
            ws = warm_start_for(db, "humanoid", "grasp", species="humanoid.upper.arm.gripper",
                                obs_dim=40, act_dim=6)
            self.assertIsNotNone(ws)
            self.assertEqual(ws["warm_start"], "transfer")

    def test_warm_start_returns_none_when_bank_empty(self):
        with tempfile.TemporaryDirectory() as tmp, self._db(tmp) as db:
            self.assertIsNone(warm_start_for(db, "manipulator", "grasp"))


if __name__ == "__main__":
    unittest.main()
