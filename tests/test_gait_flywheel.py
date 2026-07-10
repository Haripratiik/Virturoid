"""Gait flywheel COMPOUNDS: a learned gait is banked as a specific reusable skill, recalled for the same body,
and warm-starting from it reaches the walk instantly (search cost -> ~0), with a provenance edge recorded.

This is the honest answer to "does the flywheel actually work": banked -> recalled (specific) -> reused (measured).
MuJoCo-gated (physics is the judge). AGENTS.md offline.
"""
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo — physics is the judge")
class GaitFlywheelTests(unittest.TestCase):
    def _db(self):
        from virturoid.services.memory_db import MemoryDB
        return MemoryDB(db_path=Path(tempfile.mkdtemp(prefix="gaitfw_")) / "m.db")

    def _quad(self):
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.morphology_composer import compose_robot
        p = "a quadruped robot dog"
        return ensure_walkable_quad(compose_robot(p), p)

    def test_bank_only_deployable_and_recall_specific(self):
        from virturoid.services.gait_flywheel import bank_gait, recall_gait
        db, g = self._db(), self._quad()

        class _Fall:  # a fall must NOT be banked (the bank stays a bank of WORKING controllers)
            best_survived, best_forward, best_height_ratio, best_params = False, -0.1, 0.4, {"freq": 1.0}
        self.assertIsNone(bank_gait(db, g, _Fall()))
        self.assertIsNone(recall_gait(db, g))                  # nothing banked yet

        class _Walk:
            best_survived, best_height_ratio = True, 0.85
            best_forward = 1.2
            best_params = {"freq": 1.4, "hip_amp": 1.1, "knee_amp": 0.8, "duty": 0.4, "kp": 200.0, "kd": 7.0}
        sid = bank_gait(db, g, _Walk())
        self.assertTrue(sid and sid.startswith("gait::quadruped::"))   # SPECIFIC key: class + body
        recalled = recall_gait(db, g)                          # retrieve the actual controller params back
        self.assertIsNotNone(recalled)
        self.assertAlmostEqual(recalled["kp"], 200.0, places=3)

    def test_bank_rejects_a_non_credible_slide(self):
        # UN-GAMEABLE: a body that SLIDES forward (survives + travels past the 0.15 m floor but never steps) is not a
        # walk and must never enter the bank — else a slide masquerades as a gait and gets recalled/reused. This is
        # the exact failure the measured sweep surfaced: crawl kp>=40 on a light quad "survives 0.85 m" as a SLIDE.
        from virturoid.services.gait_flywheel import bank_gait, recall_gait
        db, g = self._db(), self._quad()

        class _Slide:                                          # survives + far, but not a credible walk
            best_survived, best_forward, best_height_ratio, best_credible = True, 0.85, 0.79, False
            best_params = {"freq": 1.5, "hip_amp": 0.9, "knee_amp": 1.0, "duty": 0.25, "kp": 40.0, "kd": 2.0}
        self.assertIsNone(bank_gait(db, g, _Slide()))          # a slide is rejected...
        self.assertIsNone(recall_gait(db, g))                  # ...so nothing is banked

        class _Walk:                                           # a shorter but CREDIBLE walk IS bankable
            best_survived, best_forward, best_height_ratio, best_credible = True, 0.55, 0.82, True
            best_params = {"freq": 1.4, "hip_amp": 1.1, "knee_amp": 0.8, "duty": 0.2, "kp": 120.0, "kd": 8.0}
        self.assertIsNotNone(bank_gait(db, g, _Walk()))

    def test_sql_fallback_does_not_recall_a_different_species_without_a_similarity_score(self):
        from virturoid.services.gait_flywheel import recall_gait

        db, g = self._db(), self._quad()
        db.record_skill(
            "gait::quadruped::remote", "quadruped", "locomotion", success_rate=0.9,
            species="different_body", gene_id="remote",
            base_config={"controller": "crawl_gait", "gait_params": {"freq": 1.2}},
        )
        # No vector record exists, so the fallback has no structural distance
        # evidence and must not substitute a merely class-matched gait.
        self.assertIsNone(recall_gait(db, g))

    def test_recall_by_morphology_embedding_across_species(self):
        # the flywheel retrieves via the ROBOTICS TOKENIZATION: a structurally-similar body finds a prior gait
        # even when its species/class STRING differs (semantic morphology retrieval, not exact string match).
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.gait_flywheel import bank_gait, recall_gait
        from virturoid.services.morphology_composer import compose_robot
        db = self._db()
        gA = ensure_walkable_quad(compose_robot("a quadruped robot dog"), "a quadruped robot dog")

        class _Walk:
            best_survived, best_height_ratio, best_forward = True, 0.88, 1.6
            best_params = {"freq": 1.42, "hip_amp": 1.1, "knee_amp": 0.78, "duty": 0.42, "kp": 233.0, "kd": 6.7}
        bank_gait(db, gA, _Walk())

        gB = ensure_walkable_quad(compose_robot("a four legged walking robot"), "a four legged walking robot")
        if getattr(gB, "species", None) != getattr(gA, "species", None):    # only meaningful if strings differ
            recalled = recall_gait(db, gB)
            self.assertIsNotNone(recalled, "morphology embedding should retrieve a structurally-similar prior")
            self.assertAlmostEqual(recalled["kp"], 233.0, places=3)         # reused A's actual controller params

    def test_warm_start_provenance_names_the_actual_parent_at_deploy_horizon(self):
        """The flywheel edge must identify the recalled skill and compare controllers at one horizon."""
        from virturoid.services.gait_flywheel import bank_gait, learn_gait_flywheel
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory

        db, g = self._db(), self._quad()
        prior_params = {"freq": 1.3, "hip_amp": 1.0, "knee_amp": 0.8, "duty": 0.3, "kp": 120.0, "kd": 5.0}

        class _Prior:
            best_survived, best_forward, best_height_ratio, best_credible = True, 0.8, 0.9, True
            best_params = prior_params
        parent_skill = bank_gait(db, g, _Prior())
        self.assertIsNotNone(parent_skill)

        best_params = {**prior_params, "kp": 150.0}
        result = SimpleNamespace(
            best_params=best_params, best_forward=1.1, best_height_ratio=0.9,
            best_survived=True, best_credible=True, prior_transfer_forward=0.7,
        )

        def fake_evaluate(_gene, params, *, steps):
            forward = 1.2 if params.get("kp") == 150.0 else (0.8 if params.get("kp") == 120.0 else 0.1)
            return {"forward": forward, "height_ratio": 0.9, "survived": True,
                    "credible": params.get("kp") != 32.0}

        with patch("virturoid.services.gait_search.search_gait", return_value=result), \
             patch("virturoid.services.gait_search.evaluate_gait", side_effect=fake_evaluate):
            learned = learn_gait_flywheel(g, db, generations=1, pop=2, steps=10, deploy_steps=99)

        self.assertEqual(0.8, learned["prior_deploy_forward_m"])
        self.assertEqual(0.4, learned["compounding_delta"])
        edges = RoboticsVectorMemory(db).provenance_for("skill", learned["banked_skill"])
        edge = edges[-1]
        self.assertEqual(parent_skill, edge["parent_id"])
        meta = json.loads(edge["meta"])
        self.assertEqual(99, meta["horizon_steps"])
        self.assertEqual(0.8, meta["prior_deploy_forward"])

    def test_flywheel_compounds_on_reuse(self):
        from virturoid.services.gait_flywheel import learn_gait_flywheel
        from virturoid.services.gait_search import search_gait
        db, g = self._db(), self._quad()
        # the rest-stance default walk is strong (~0.83 m now that bodies spawn in their baked stance), so the
        # search needs a real budget to BEAT it and bank — the compounding mechanism is unchanged, the baseline rose.
        budget = dict(generations=8, pop=16, steps=800)

        cold = search_gait(g, seed=1, **budget)               # first solve, cold (nothing banked)
        self.assertTrue(cold.best_survived)
        # bank it through the flywheel
        first = learn_gait_flywheel(g, db, seed=1, **budget)
        self.assertTrue(first["banked_skill"])
        self.assertFalse(first["reused_prior"])               # first time: nothing to reuse

        second = learn_gait_flywheel(g, db, seed=2, **budget)  # SAME body again -> must reuse the bank
        self.assertTrue(second["reused_prior"])
        # the recalled prior transfers to (its own) body ~ as well as the banked walk -> instant, no re-search cost.
        self.assertGreaterEqual(second["prior_transfer_forward"], 0.9 * abs(cold.best_forward))
        self.assertTrue(second["survived"])

        # provenance edge recorded -> compounding is MEASURED, not asserted.
        from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
        cs = RoboticsVectorMemory(db).compounding_summary()
        self.assertGreaterEqual(cs.get("edges", 0), 1)


if __name__ == "__main__":
    unittest.main()
