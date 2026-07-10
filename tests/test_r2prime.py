"""R2' harness (Thesis A decisive measurement) — plumbing + un-gameable verdict + decision rule.

FAST by design: the decisive credible-walk NUMBER needs a realistic search budget (scripts/eval_r2prime.py runs
that), so here we bank a known-good gait DIRECTLY (no minutes-long search) and assert the machinery is correct:
the three knowledge arms deploy the RIGHT gait, the solve verdict cannot be gamed by a dead gait, and the
decision rule maps rates to the dossier's outcomes.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

_BANKED = {"freq": 1.7, "hip_amp": 0.95, "knee_amp": 1.15, "duty": 0.3, "kp": 34.0, "kd": 1.6}


class R2PrimeDecisionTests(unittest.TestCase):
    """Pure decision-rule mapping — no physics, always runs."""

    def _res(self, off, cs, co):
        return {"off": {"verified_solve_rate": off}, "cheatsheet": {"verified_solve_rate": cs},
                "corpus": {"verified_solve_rate": co}}

    def test_decision_rule_maps_every_outcome(self):
        from virturoid.services.r2prime import decision
        self.assertIn("FIX retrieval", decision(self._res(0.5, 0.5, 0.5)))       # corpus <= off
        self.assertIn("FIX retrieval", decision(self._res(0.75, 0.5, 0.5)))      # corpus < off
        self.assertIn("re-scope", decision(self._res(0.25, 0.75, 0.75)))         # corpus ~ cheatsheet (> off)
        self.assertIn("REAL", decision(self._res(0.25, 0.5, 0.9)))               # corpus > both


@unittest.skipUnless(_MUJOCO, "the solve verdict rolls out the crawl gait in MuJoCo")
class R2PrimeMechanismTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="r2t_")) / "m.db")

    def test_arms_deploy_the_right_gait(self):
        # off = cold default; cheatsheet = the one best banked exemplar; corpus = the morphology-matched recall
        from virturoid.services.gait_flywheel import _DEFAULT_GAIT, _DeployResult, bank_gait
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.r2prime import arm_params
        db = self._db()
        gene = compose_robot("a quadruped robot dog")
        # bank a known-good (credible) gait directly — no slow search needed to test the plumbing
        sid = bank_gait(db, gene, _DeployResult(dict(_BANKED),
                        {"forward": 0.8, "height_ratio": 0.9, "survived": True, "credible": True}))
        self.assertTrue(sid, "a credible gait must bank so the corpus is non-empty")
        self.assertEqual(arm_params("off", gene, db), _DEFAULT_GAIT, "off = cold default (no grounding)")
        self.assertEqual(arm_params("cheatsheet", gene, db)["freq"], _BANKED["freq"], "cheatsheet = best banked")
        self.assertEqual(arm_params("corpus", gene, db)["freq"], _BANKED["freq"], "corpus = morphology recall")
        self.assertNotEqual(arm_params("corpus", gene, db)["freq"], _DEFAULT_GAIT["freq"], "grounding changed the gait")

    def test_corpus_falls_back_to_default_on_a_miss(self):
        from virturoid.services.gait_flywheel import _DEFAULT_GAIT
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.r2prime import arm_params
        db = self._db()                                              # empty corpus
        gene = compose_robot("a quadruped robot dog")
        self.assertEqual(arm_params("corpus", gene, db), _DEFAULT_GAIT, "no precedent -> default (never worse than off)")

    def test_solve_verdict_is_ungameable(self):
        # a DEAD gait (no oscillation) can't produce a credible walk, so it never counts as solved
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.r2prime import verified_solve
        gene = compose_robot("a quadruped robot dog")
        dead = {"freq": 0.0, "hip_amp": 0.0, "knee_amp": 0.0, "duty": 0.25, "kp": 32.0, "kd": 1.5}
        solved, r = verified_solve(gene, dead, steps=400)
        self.assertFalse(solved, f"a motionless gait is never a credible walk: {r.get('verdict')}")


if __name__ == "__main__":
    unittest.main()
