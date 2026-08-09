"""Learned deployable gait via CEM (gait_search): un-gameable fitness + it improves a real body's walk.

The breakthrough that closes the MJX->CPU deploy gap by LEARNING the parameters of the controller that already
deploys. Offline; MuJoCo-gated (real physics is the judge). AGENTS.md.
"""
import importlib.util
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


class GaitSearchBudgetTests(unittest.TestCase):
    def _result(self, i: int, credible: bool = False) -> dict:
        return {"fitness": float(i), "forward": float(i) / 10.0, "height_ratio": 0.8,
                "survived": True, "credible": credible, "verdict": "CREDIBLE WALK" if credible else "SLIDE"}

    def test_exact_budget_when_no_credible_walk_is_found(self):
        from virturoid.services.gait_search import search_gait
        calls = []

        def fake(_gene, _params, **_kwargs):
            calls.append(1)
            return self._result(len(calls), credible=False)

        with patch("virturoid.services.gait_search.evaluate_gait", side_effect=fake):
            result = search_gait(object(), generations=10, pop=2, max_evals=5, stop_on_credible=True)
        self.assertEqual(5, result.n_evals)
        self.assertEqual("max_evals", result.stopped_reason)
        self.assertEqual(6, len(calls))  # one separately reported baseline + five search candidates

    def test_credible_walk_stops_before_the_budget(self):
        from virturoid.services.gait_search import search_gait
        calls = []

        def fake(_gene, _params, **_kwargs):
            calls.append(1)
            candidate_i = len(calls) - 1  # the first call is the baseline
            return self._result(candidate_i, credible=candidate_i == 3)

        with patch("virturoid.services.gait_search.evaluate_gait", side_effect=fake):
            result = search_gait(object(), generations=4, pop=5, max_evals=9, stop_on_credible=True)
        self.assertEqual(3, result.n_evals)
        self.assertEqual("credible_walk", result.stopped_reason)
        self.assertTrue(result.best_credible)


class ForwardVelSignTests(unittest.TestCase):
    """B5: `forward_vel` was SIGN-INVERTED — a backward walk reported a POSITIVE forward velocity.

    `speed` is already signed at every producer, and gait_search re-applied the sign of `forward` on top of it,
    so (-1)x(-1) = +1. That is not a display bug: every shipped reward template targets this feature, so the
    optimizer PAID MAXIMUM FOR WALKING BACKWARD at the target speed. These tests pin the sign at the feature and
    at the reward, so the two ways it can regress (drop the negative branch, or re-add a sign multiplier) both fail.
    """

    def _rollout(self, speed: float, forward: float) -> dict:
        return {"speed": speed, "forward": forward, "survived": True, "upright_frac": 0.95,
                "height_ratio": 0.9, "support_frac": 0.8, "lateral": 0.05}

    def test_backward_walk_reports_a_negative_forward_vel(self):
        from virturoid.services.gait_search import reward_features_from_rollout
        # the measured Go2 numbers from the bug report, replayed exactly
        f = reward_features_from_rollout(self._rollout(-0.409, -0.491))
        self.assertLess(f["forward_vel"], 0.0, "a body that travelled BACKWARD must not report forward progress")
        self.assertAlmostEqual(f["forward_vel"], -0.409, places=9)

    def test_forward_walk_is_unchanged(self):
        # the fix removed a multiplier that was +1 for every forward walk, so forward bodies must be byte-identical.
        from virturoid.services.gait_search import reward_features_from_rollout
        f = reward_features_from_rollout(self._rollout(+0.409, +0.491))
        self.assertAlmostEqual(f["forward_vel"], +0.409, places=9)

    def test_forward_vel_is_exactly_the_signed_speed_never_a_re_signed_one(self):
        # THE invariant that makes the double sign impossible to reintroduce: the feature IS `speed`, untouched.
        from virturoid.services.gait_search import reward_features_from_rollout
        for speed, forward in ((-0.409, -0.491), (0.409, 0.491), (-0.03, -0.05), (0.0, 0.0), (0.7, 1.2)):
            with self.subTest(speed=speed):
                self.assertEqual(reward_features_from_rollout(self._rollout(speed, forward))["forward_vel"], speed)

    def test_shipped_reward_templates_no_longer_pay_max_for_walking_backward(self):
        # the PRODUCT consequence. velocity_track targets 0.4 m/s: at -0.409 m/s the inverted feature scored
        # 0.9997 of its maximum (measured on the imported Go2). Every template must now rank forward above backward.
        from virturoid.services.gait_search import reward_features_from_rollout
        from virturoid.services.reward_dsl import _TEMPLATES, compile_reward
        back = reward_features_from_rollout(self._rollout(-0.409, -0.491))
        fwd = reward_features_from_rollout(self._rollout(+0.409, +0.491))
        for name in ("velocity_track", "progress_upright", "smooth_march", "clearance_gait"):
            with self.subTest(template=name):
                fn = compile_reward(_TEMPLATES[name])
                self.assertLess(fn(back), fn(fwd),
                                f"{name} pays a backward walk at least as much as the mirror-image forward walk")
        # and specifically: the velocity-tracking term must not treat -0.4 m/s as though it were +0.4 m/s
        vt = compile_reward(_TEMPLATES["velocity_track"])
        self.assertLess(vt(back), 0.5 * vt(fwd))

    def test_improvement_x_cannot_report_backward_travel_as_progress(self):
        # the same sign defect one layer up: abs()/abs() turned "went 12x further BACKWARD" into "12.59x better".
        from virturoid.services.gait_search import GaitSearchResult

        def _res(best, base):
            return GaitSearchResult(best_params={}, best_fitness=0.0, best_forward=best, best_height_ratio=0.9,
                                    best_survived=True, baseline_forward=base).to_dict()

        self.assertEqual(_res(+0.491, 0.039)["improvement_x"], 12.59)     # forward: unchanged
        self.assertEqual(_res(-0.491, 0.039)["improvement_x"], -12.59)    # backward: NEGATIVE, never a multiple
        self.assertIsNone(_res(+0.491, -0.100)["improvement_x"])          # baseline itself backward -> undefined
        self.assertIsNone(_res(+0.491, 0.0)["improvement_x"])


@unittest.skipUnless(_MUJOCO, "needs MuJoCo — physics is the fitness judge")
class ForwardVelSignOnRealHardwareTests(unittest.TestCase):
    """The same claim, on REAL imported hardware rather than a hand-built dict.

    Fixtures have lied in this repo before, so this drives a genuine MuJoCo Menagerie quadruped through
    ``agent_tools.call_tool('ingest_project')`` and sweeps the crawl gait until the body nets BACKWARD, then
    checks the feature the reward reads. Skipped when the Menagerie cache is absent.
    """

    _MENAGERIE = os.path.join(os.path.expanduser("~"), ".cache", "robot_descriptions", "mujoco_menagerie")
    # bounds-corner gaits; at least one nets backward on any of these bodies (measured on all three)
    _GRID = (
        {"freq": 2.8, "hip_amp": 0.4, "knee_amp": 0.5, "kp": 240.0, "kd": 14.0},
        {"freq": 0.8, "hip_amp": 0.4, "knee_amp": 0.5, "kp": 24.0, "kd": 1.0},
        {"freq": 1.1, "hip_amp": 1.5, "knee_amp": 1.5, "kp": 90.0, "kd": 5.0},
        {"freq": 3.2, "hip_amp": 1.5, "knee_amp": 0.5, "kp": 240.0, "kd": 14.0},
    )

    def _imported(self, pkg: str):
        from virturoid.services import session_state
        from virturoid.services.agent_tools import call_tool
        path = os.path.join(self._MENAGERIE, pkg)
        if not os.path.isdir(path):
            self.skipTest(f"no Menagerie checkout at {path}")
        out = call_tool("ingest_project", {"project_path": path, "description": f"{pkg} gait sign audit"})
        self.assertTrue(out.get("ok"), out.get("error"))
        gene = session_state.get_robot(out["result"]["robot_id"])
        self.assertIsNotNone(gene, "ingest_project reported ok but held no robot")
        return gene

    def test_a_real_imported_quadruped_that_walks_backward_reports_it(self):
        from virturoid.services.gait_search import reward_features_from_rollout
        from virturoid.services.morph_policy import crawl_gait_rollout
        gene = self._imported("unitree_go2")
        signs = []
        for params in self._GRID:
            r = crawl_gait_rollout(gene, steps=800, record_qpos=True, **params)
            fv = reward_features_from_rollout(r)["forward_vel"]
            # the invariant holds on EVERY real rollout, whichever way the body went
            self.assertEqual(fv, r["speed"], f"forward_vel diverged from the rollout's own signed speed at {params}")
            signs.append(fv)
            if fv < 0:
                # the case the old code got wrong: a genuinely backward walk must read as backward
                self.assertLess(fv, 0.0)
                self.assertLessEqual(float(r["forward"]), 0.0)
        self.assertTrue(any(s < 0 for s in signs),
                        f"no gait in the grid netted backward on this body ({signs}) — the test proved nothing")


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
        # Compare the tuned crawl against a measured high-amplitude, lightly
        # damped gait that survives but rears/lurches. A faster candidate that
        # genuinely stays level is a valid walk and must not be demoted merely
        # because its parameters look aggressive.
        upright = evaluate_gait(g, {"freq": 1.4, "hip_amp": 0.9, "knee_amp": 0.9, "duty": 0.13,
                                    "kp": 110.0, "kd": 10.0}, steps=800)
        toppled = evaluate_gait(g, {"freq": 0.8, "hip_amp": 1.5, "knee_amp": 1.5, "duty": 0.12,
                                    "kp": 24.0, "kd": 1.0}, steps=800)
        # the upright gait survives and walks; a measured lurch cannot
        # out-score it even when the lurch covers forward distance.
        self.assertTrue(upright["survived"])
        self.assertFalse(toppled["credible"])
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
        # SIGNED, not abs(). This file asserts fifty lines up that "a body that travelled BACKWARD must not
        # report forward progress" — and then wrapped its own headline in abs(), so a search whose entire
        # population only walked backward (best_forward -0.55) cleared a test named "learns a DEPLOYABLE walk".
        self.assertGreater(res.best_forward, 0.2)               # a real FORWARD walk, not a shuffle or a reverse
        # ...and CEM improved (monotone non-decreasing best across generations).
        self.assertEqual(res.history, sorted(res.history))
        # the winning params are within the searched bounds (deployable by construction).
        from virturoid.services.gait_search import _HI, _LO
        for k, v in res.best_params.items():
            self.assertGreaterEqual(v, _LO[k] - 1e-6)
            self.assertLessEqual(v, _HI[k] + 1e-6)
        # the credibility flag is carried and CONSISTENT with the fitness: a credible surviving walk scores its full
        # forward (x1.0), a slide is discounted (x0.3) — so best_credible can never be silently decoupled from fitness.
        self.assertIsInstance(res.best_credible, bool)
        self.assertIn("best_credible", res.to_dict())
        if res.best_survived and res.best_credible:
            self.assertAlmostEqual(res.best_fitness, res.best_forward, places=6)

    def test_generalizes_to_hexapod(self):
        # the learner is not quad-specific: crawl_gait_rollout handles any leg count, so a hexapod learns a walk too.
        from virturoid.services.gait_search import search_gait
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a hexapod robot")
        res = search_gait(g, generations=4, pop=10, steps=700, seed=1, workers=1)
        self.assertTrue(res.best_survived)
        self.assertGreaterEqual(res.best_height_ratio, 0.6)
        self.assertGreater(res.best_forward, 0.3)               # a real 6-leg FORWARD walk (see the abs() note above)

    def test_deterministic_for_seed(self):
        from virturoid.services.gait_search import search_gait
        g = self._quad()
        a = search_gait(g, generations=3, pop=8, steps=600, seed=7, workers=1)
        b = search_gait(g, generations=3, pop=8, steps=600, seed=7, workers=1)
        self.assertEqual(a.best_params, b.best_params)


if __name__ == "__main__":
    unittest.main()
