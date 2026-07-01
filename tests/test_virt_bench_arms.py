"""VIRT-Bench arms (WS3): Arm A (fixed-pipeline baseline) vs Arm B (our CPG-search harness), both scored by the
INDEPENDENT verifier. The A->B delta is the measured value of the search harness; a task both fail is the next
build priority (honest compass)."""

import unittest

from virturoid.services.virt_bench_arms import run_arm_a, run_arm_b, run_dev_scoreboard


class VirtBenchArmsTests(unittest.TestCase):
    def test_both_arms_return_verified_verdicts(self):
        a = run_arm_a("L1_quad_walk", steps=120)
        b = run_arm_b("L1_quad_walk", steps=120, max_evals=4)
        for r, arm in ((a, "A"), (b, "B")):
            self.assertEqual(r["arm"], arm)
            self.assertIsInstance(r["verified_pass"], bool)      # a real re-run verdict, not a self-claim
            self.assertIn("forward_m", r["metrics"])
        # Arm B actually SEARCHED and selected a CPG (the harness ran), then submitted it for verification
        self.assertGreaterEqual(b["n_evals"], 1)
        self.assertIsNotNone(b["searched"])

    def test_dev_scoreboard_is_honest(self):
        sb = run_dev_scoreboard(steps=120, max_evals=4)
        self.assertGreaterEqual(sb["n_tasks"], 1)                # at least L1 in the dev locomotion split
        # every solved count is grounded in an independent verify, never a self-report
        self.assertEqual(sb["harness_delta"], sb["B_solved"] - sb["A_solved"])
        self.assertTrue(all(isinstance(r["A_pass"], bool) and isinstance(r["B_pass"], bool) for r in sb["rows"]))


if __name__ == "__main__":
    unittest.main()
