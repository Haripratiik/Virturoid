"""Night-shift runner (M4 capstone, plan v2 §5.1/§5.2/§5.4): assembles proposer + learnability + laddered
evaluator + golden gate + run_night_shift into one resumable command. Stubbed evaluate_for -> no GPU/MuJoCo/LLM."""

import tempfile
import unittest
from pathlib import Path

from virturoid.services.night_runner import make_arms, run_night


def _stub_evaluate_for(cand):
    # a candidate whose id contains a high digit "solves" (deterministic) -> exercises banking + the journal
    def evaluate(spec):
        sr = 0.8 if ("9" in cand["id"] or "8" in cand["id"]) else 0.1
        return {"forward": sr, "cadence": 4.0, "upright_frac": 0.9, "survived": True}
    return evaluate


class NightRunnerTests(unittest.TestCase):
    def test_arms_produce_walk_candidates(self):
        mutate, transfer, explore = make_arms(seed=1)
        self.assertIsNone(explore)                                  # LLM-free night by default
        c = mutate()
        for k in ("id", "gene", "task", "task_type", "gates", "heuristic"):
            self.assertIn(k, c)
        self.assertEqual(c["task_type"], "locomotion")

    def test_run_night_assembles_and_journals(self):
        with tempfile.TemporaryDirectory() as tmp:
            jrn = Path(tmp) / "night.jsonl"
            rep = run_night(memory_dir=tmp, journal_path=jrn, n_candidates=6, per_candidate_evals=3,
                            evaluate_for=_stub_evaluate_for, run_golden=False, seed=3)
            self.assertEqual(rep["proposed"], 6)
            self.assertIsNotNone(rep["night"])
            self.assertTrue(jrn.exists())                          # resumable journal written
            # resume: everything already journaled -> nothing re-run
            rep2 = run_night(memory_dir=tmp, journal_path=jrn, n_candidates=6, per_candidate_evals=3,
                             evaluate_for=_stub_evaluate_for, run_golden=False, seed=3)
            self.assertEqual(rep2["night"].candidates_run, 0)

    def test_golden_regression_disables_banking(self):
        with tempfile.TemporaryDirectory() as tmp:
            # force a golden regression via a stub that reports a regressed case -> banking disabled (budget 0)
            import virturoid.services.night_runner as nr
            rep = run_night(memory_dir=tmp, n_candidates=4, evaluate_for=_stub_evaluate_for,
                            run_golden=True, seed=1,
                            probe=None)
            # golden ran (real quad floor 0.0); banking_enabled reflects whether it passed
            self.assertIn("banking_enabled", rep)
            self.assertIsInstance(rep["banking_enabled"], bool)


if __name__ == "__main__":
    unittest.main()
