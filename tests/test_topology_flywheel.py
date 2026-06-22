"""Topology flywheel: a winning topology (leg count) is banked and warm-starts the next build — the
species tree improves SHAPE across builds. Gated on a banked transferable policy."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

_MUJOCO = importlib.util.find_spec("mujoco") is not None
_NPZ = Path(__file__).resolve().parent.parent / "models" / "morph_quad_att.npz"


@unittest.skipUnless(_MUJOCO and _NPZ.exists(), "needs MuJoCo + a banked transferable policy")
class TopologyFlywheelTests(unittest.TestCase):
    def test_banks_topology_then_warm_starts(self):
        from virturoid.services.design_flywheel import topology_codesign_with_memory
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.morph_policy import MorphPolicy
        from virturoid.services.steerable_body import steerable_quadruped

        pol = MorphPolicy.from_npz(str(_NPZ))
        seed_body = steerable_quadruped(n_legs=4)
        with tempfile.TemporaryDirectory() as td:
            with MemoryDB(Path(td) / "m.db") as db:
                r1 = topology_codesign_with_memory(seed_body, "walk forward", db, pol, steps=300)
                self.assertFalse(r1["warm_started"])              # nothing banked the first time
                self.assertIn(r1["best_n_legs"], range(3, 8))
                prior = db.best_design(seed_body.robot_class, "legged")
                self.assertIsNotNone(prior)                       # topology was banked

                r2 = topology_codesign_with_memory(seed_body, "walk forward", db, pol, steps=300)
                self.assertTrue(r2["warm_started"])               # 2nd build reuses the banked topology
                self.assertIsNotNone(r2["prior_best"])


if __name__ == "__main__":
    unittest.main()
