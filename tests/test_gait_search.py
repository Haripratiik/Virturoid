"""Learned deployable gait via CEM (gait_search): un-gameable fitness + it improves a real body's walk.

The breakthrough that closes the MJX->CPU deploy gap by LEARNING the parameters of the controller that already
deploys. Offline; MuJoCo-gated (real physics is the judge). AGENTS.md.
"""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo — physics is the fitness judge")
class GaitSearchTests(unittest.TestCase):
    def _quad(self):
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.morphology_composer import compose_robot
        p = "a quadruped robot dog"
        return ensure_walkable_quad(compose_robot(p), p)

    def test_fitness_is_ungameable(self):
        # a fall must score BELOW an upright-forward gait, so the search can never be won by a face-plant.
        from virturoid.services.gait_search import evaluate_gait
        g = self._quad()
        # a wild high-frequency / high-amplitude gait tends to topple; the tuned crawl stays up.
        upright = evaluate_gait(g, {"freq": 1.4, "hip_amp": 0.9, "knee_amp": 0.9, "duty": 0.13,
                                    "kp": 110.0, "kd": 10.0}, steps=800)
        toppled = evaluate_gait(g, {"freq": 3.2, "hip_amp": 1.5, "knee_amp": 1.5, "duty": 0.42,
                                    "kp": 40.0, "kd": 2.0}, steps=800)
        # the upright gait survives and walks; whatever the toppled one does, it cannot out-score a real walk.
        self.assertTrue(upright["survived"])
        self.assertGreater(upright["fitness"], toppled["fitness"])
        if not toppled["survived"]:
            self.assertLess(toppled["fitness"], 0.0)             # a fall scores negative

    def test_search_learns_a_deployable_walk(self):
        from virturoid.services.gait_search import search_gait
        g = self._quad()
        res = search_gait(g, generations=4, pop=10, steps=800, seed=0, workers=1)
        # it found an upright, surviving, forward gait...
        self.assertTrue(res.best_survived)
        self.assertGreaterEqual(res.best_height_ratio, 0.6)
        self.assertGreater(abs(res.best_forward), 0.2)          # a real walk, not a shuffle
        # ...and CEM improved (monotone non-decreasing best across generations).
        self.assertEqual(res.history, sorted(res.history))
        # the winning params are within the searched bounds (deployable by construction).
        from virturoid.services.gait_search import _HI, _LO
        for k, v in res.best_params.items():
            self.assertGreaterEqual(v, _LO[k] - 1e-6)
            self.assertLessEqual(v, _HI[k] + 1e-6)

    def test_generalizes_to_hexapod(self):
        # the learner is not quad-specific: crawl_gait_rollout handles any leg count, so a hexapod learns a walk too.
        from virturoid.services.gait_search import search_gait
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a hexapod robot")
        res = search_gait(g, generations=4, pop=10, steps=700, seed=1, workers=1)
        self.assertTrue(res.best_survived)
        self.assertGreaterEqual(res.best_height_ratio, 0.6)
        self.assertGreater(abs(res.best_forward), 0.3)          # a real 6-leg walk, deployable by construction

    def test_deterministic_for_seed(self):
        from virturoid.services.gait_search import search_gait
        g = self._quad()
        a = search_gait(g, generations=3, pop=8, steps=600, seed=7, workers=1)
        b = search_gait(g, generations=3, pop=8, steps=600, seed=7, workers=1)
        self.assertEqual(a.best_params, b.best_params)


if __name__ == "__main__":
    unittest.main()
