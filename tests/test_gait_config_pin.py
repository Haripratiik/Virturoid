"""Thesis A / dossier risk #10: a banked gait verdict carries the SIM CONFIG it was measured under, so a recalled
'walks 0.65 m' is not a config-free (and therefore misleading) number. Additive pin — recall is unchanged.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "the deploy sim config is read from a compiled MuJoCo model")
class GaitConfigPinTests(unittest.TestCase):
    def test_banked_gait_pins_its_deploy_sim_config(self):
        from virturoid.services.gait_flywheel import bank_gait
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morphology_composer import compose_robot
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="gaitpin_")) / "m.db")
        gene = compose_robot("a quadruped robot dog that walks")
        # a CREDIBLE deployable result (the only thing bank_gait will store)
        res = SimpleNamespace(best_survived=True, best_forward=0.65, best_credible=True,
                              best_height_ratio=0.9, best_params={"amp": 0.5, "freq": 2.0})
        sid = bank_gait(db, gene, res)
        self.assertTrue(sid, "a credible deployable gait must bank")
        bc = json.loads(db.conn.execute("SELECT base_config FROM skills WHERE skill_id=?", (sid,)).fetchone()[0])
        self.assertIn("sim_config", bc, "the bank must pin the sim config the verdict was measured under")
        sc = bc["sim_config"]
        self.assertEqual(sc["controller"], "crawl_gait")
        self.assertEqual(sc["engine"], "mujoco")
        # the physics the 0.65 m is valid under (real, positive timestep; downward gravity)
        self.assertGreater(sc["timestep"], 0.0)
        self.assertLess(sc["gravity_z"], 0.0)


if __name__ == "__main__":
    unittest.main()
