"""The product's core failure-driven loop on REAL physics (startup plan §26/§11.2/§31.5, Milestone E):
Task -> tiers -> Run -> cluster Failures -> failure-conditioned Regression -> re-Evaluate (confirm cause) ->
Report + export Gate. Locks that the stress tier surfaces real failures, that they cluster, that the
failure-conditioned counterfactuals actually RESOLVE the failure (confirming the diagnosed cause), and that the
export gate keys off baseline+randomized (not the analysis-only stress tier). See [[task-effectiveness-loop]]."""

import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class FailureLoopTests(unittest.TestCase):
    def _run(self):
        from virturoid.services.evaluation_loop import run_pickplace_failure_loop
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a tabletop robot arm", llm=None)
        # small tier counts keep the integration test fast while still exercising every stage
        return run_pickplace_failure_loop(g, seed=0, tier_counts={"smoke": 1, "baseline": 2, "randomized": 2, "stress": 3})

    def test_loop_surfaces_clusters_and_confirms_real_failures(self):
        rep = self._run()
        # named tiers all ran under real physics
        for tier in ("smoke", "baseline", "randomized", "stress"):
            self.assertIn(tier, rep["tiers"])
            self.assertGreater(rep["tiers"][tier]["episodes"], 0)
        # the STRESS tier exists to make the controller fail (reach edge + heavy) -> real material for the loop
        self.assertGreater(rep["tiers"]["stress"]["failures"], 0, "stress tier should surface real failures")
        # failures are clustered by a real label
        self.assertTrue(rep["failure_clusters"])
        self.assertTrue(all(c["failure_label"] for c in rep["failure_clusters"]))
        # failure-CONDITIONED regression: the targeted one-variable counterfactual resolves the failure for the
        # large majority of cases, CONFIRMING the diagnosed cause (plan §31.5). Not 100%-required (physics noise).
        self.assertGreaterEqual(rep["regression"]["n"], 1)
        self.assertGreaterEqual(rep["regression"]["cause_confirmed_rate"], 0.6)

    def test_export_gate_keys_off_baseline_and_randomized_not_stress(self):
        rep = self._run()
        # the gate must IGNORE the stress tier (analysis-only) and key off baseline+randomized (plan §11.2/§15.3)
        blockers_mention_stress = any("stress" in b for b in rep["export_blockers"])
        self.assertFalse(blockers_mention_stress)
        baseline_ok = rep["tiers"]["baseline"]["success_rate"] >= 0.8
        randomized_ok = rep["tiers"]["randomized"]["success_rate"] >= 0.8
        self.assertEqual(rep["export_ready"], baseline_ok and randomized_ok)


if __name__ == "__main__":
    unittest.main()
