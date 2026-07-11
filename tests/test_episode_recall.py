"""P2 — the episode sub-space was indexed on every build but had NO reader (observability gap). recall_knowledge
now exposes it: given a body's just-run behaviour features, it returns the bodies that BEHAVED like it. Honest
scope — behavioural CONTEXT, not a fine-transfer predictor."""
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")


class EpisodeRecallTests(unittest.TestCase):
    def _mem(self):
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        db = MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="epi_")) / "m.db")
        return db, RoboticsVectorMemory(db)

    def test_recall_returns_behaviorally_nearest_episode(self):
        from virturoid.services.morphology_composer import compose_robot
        db, vm = self._mem()
        # two very different behaviours: a brisk walker vs a near-static shuffler
        vm.index_episode("walker", {"forward_m": 1.2, "cadence": 2.1, "upright_frac": 0.95},
                         {"species": "quadruped.a", "status": "walked"})
        vm.index_episode("shuffler", {"forward_m": 0.05, "cadence": 0.2, "upright_frac": 0.6},
                         {"species": "quadruped.b", "status": "marginal"})
        gene = compose_robot("a small quadruped robot dog")
        # a query that BEHAVES like the brisk walker -> walker is the top episode neighbour
        out = vm.recall_knowledge(gene, task_type="locomotion",
                                  behavior_features={"forward_m": 1.15, "cadence": 2.0, "upright_frac": 0.93})
        self.assertIn("episodes", out)
        self.assertTrue(out["episodes"])
        self.assertEqual(out["episodes"][0]["obj_id"], "walker")
        # without behavior features the channel is simply empty (lightweight design-time grounding, no probe)
        cold = vm.recall_knowledge(gene, task_type="locomotion")
        self.assertEqual(cold["episodes"], [])


if __name__ == "__main__":
    unittest.main()
