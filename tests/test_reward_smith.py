"""Reward-smith scheduler (Pillar 1, Move 5): the sample-efficient loop, unit-tested with a stub trainer.

Pure-Python (no GPU/MuJoCo): a fake evaluator stands in for the expensive training run, so the ASHA
successive-halving + gate-ranked cycle logic is verified offline. The real MJX-PPO trainer + honesty gate +
gait_critic plug into the same injected callables.
"""

import unittest

from virturoid.services.reward_smith import (
    full_fidelity_budget, reward_smith_cycle, successive_halving,
)


class SuccessiveHalvingTests(unittest.TestCase):
    def test_keeps_best_and_only_finalist_reaches_full_fidelity(self):
        cands = [{"id": i, "q": q} for i, q in enumerate([0.1, 0.9, 0.3, 0.5, 0.2, 0.7, 0.4, 0.6])]
        full_evals = []

        def eval_fn(c, fidelity):
            if fidelity == 1.0:
                full_evals.append(c["id"])
            return c["q"]                       # gate score = quality (bigger better)

        ranking, history = successive_halving(cands, eval_fn, rungs=(0.12, 0.30, 1.0), eta=3)
        self.assertEqual(ranking[0][0]["id"], 1)          # the 0.9 candidate wins
        self.assertEqual(len(history), 3)                 # three rungs
        self.assertLessEqual(len(full_evals), 2)          # only the survivor(s) pay a full run, not all 8
        self.assertIn(1, full_evals)                      # and the best one is among them

    def test_empty_candidates(self):
        self.assertEqual(successive_halving([], lambda c, f: 0.0), ([], []))


class RewardSmithCycleTests(unittest.TestCase):
    def test_returns_gate_best_and_certifies_then_stops(self):
        pool = [{"id": i, "q": q} for i, q in enumerate([0.3, 0.6, 0.95, 0.4])]
        res = reward_smith_cycle(
            propose_fn=lambda champ, refl, k: pool,
            eval_fn=lambda c, fidelity: c["q"],          # gate score
            gate_fn=lambda champ, score: score >= 0.9,   # certify at 0.9
            reflect_fn=lambda champ, ranking: {"note": "reflected"},
            iterations=3, k=4)
        self.assertEqual(res["champion"]["id"], 2)        # the 0.95 candidate
        self.assertTrue(res["certified"])
        self.assertEqual(len(res["iterations"]), 1)       # certified on iter 0 -> early stop

    def test_runs_all_iterations_when_never_certified(self):
        pool = [{"id": 0, "q": 0.5}]
        res = reward_smith_cycle(
            propose_fn=lambda champ, refl, k: pool,
            eval_fn=lambda c, fidelity: c["q"],
            gate_fn=lambda champ, score: False,           # never passes the gate
            reflect_fn=lambda champ, ranking: None,
            iterations=3, k=1)
        self.assertFalse(res["certified"])
        self.assertEqual(len(res["iterations"]), 3)

    def test_champion_never_regresses_across_iterations(self):
        # a proposer whose pool worsens each round -> the champion must hold the best-ever, not the latest
        rounds = [[{"id": "a", "q": 0.8}], [{"id": "b", "q": 0.4}], [{"id": "c", "q": 0.5}]]
        it = iter(rounds)
        res = reward_smith_cycle(
            propose_fn=lambda champ, refl, k: next(it),
            eval_fn=lambda c, fidelity: c["q"],
            gate_fn=lambda champ, score: False,
            reflect_fn=lambda champ, ranking: None,
            iterations=3, k=1)
        self.assertEqual(res["champion"]["id"], "a")      # 0.8 held despite later 0.4/0.5 pools
        self.assertAlmostEqual(res["champion_score"], 0.8)


class BudgetTests(unittest.TestCase):
    def test_asha_budget_is_far_cheaper_than_naive_full_runs(self):
        # naive = K*iterations full runs = 24; ASHA screens cheaply so the full-run-equivalents are far fewer
        b = full_fidelity_budget(8, 3, rungs=(0.12, 0.30, 1.0), eta=3, confirm_seeds=2)
        self.assertLess(b, 24.0)
        self.assertGreater(b, 0.0)
        # slowest-box knob (K=6, N=2) should land near ~8 full-run-equivalents (Agent 4's estimate)
        self.assertLess(full_fidelity_budget(6, 2, confirm_seeds=1), 12.0)


if __name__ == "__main__":
    unittest.main()
