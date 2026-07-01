"""Night-shift engine core (WS2/N1): autonomous loop over candidate proposals — runs bounded searches, banks
verified wins, counts ANNECS-V novelty, and resumes from a journal (G8). LLM-free, fake evaluators, no MuJoCo."""

import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import (humanoid_upper_body_gene, quadruped_gene, tabletop_arm_gene)
from virturoid.services.night_shift import run_night_shift

_SOLVE = lambda p, h: {"edit_kind": "gains", "params": {"kp": 45.0}}
_FAIL = lambda p, h: {"edit_kind": "gains", "params": {"kp": 10.0}}


def _evaluate_for(_cand):
    # solves iff the proposed gain is strong enough (>=40)
    def evaluate(spec):
        sr = 0.85 if spec.get("params", {}).get("kp", 0) >= 40 else 0.1
        return {"success_rate": sr, "contacted": True, "lifted": sr > 0.5}
    return evaluate


def _cands():
    g = {"task": "grasp", "task_type": "grasp", "gates": {"success_rate": 0.6}, "gate_target": 0.6}
    return [
        {"id": "arm", "gene": tabletop_arm_gene(), "heuristic": _SOLVE, **g},          # solves -> novel
        {"id": "hum", "gene": humanoid_upper_body_gene(), "heuristic": _FAIL, **g},     # fails -> no bank
        {"id": "quad", "gene": quadruped_gene(), "heuristic": _SOLVE, **g},             # solves -> novel
    ]


class NightShiftTests(unittest.TestCase):
    def test_runs_banks_and_counts_novelty(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = run_night_shift(_cands(), _evaluate_for, memory_dir=tmp, llm=None, per_candidate_evals=3)
            self.assertEqual(rep.candidates_run, 3)
            self.assertEqual(rep.banked, 2)                    # arm + quad solved and banked
            self.assertEqual(rep.novel, 2)                     # both are new bodies -> ANNECS-V novel
            self.assertEqual(rep.stopped_reason, "proposals_exhausted")

    def test_budget_is_capped(self):
        with tempfile.TemporaryDirectory() as tmp:
            rep = run_night_shift(_cands(), _evaluate_for, memory_dir=tmp, llm=None, budget_evals=1,
                                  per_candidate_evals=3)
            self.assertLessEqual(rep.candidates_run, 1)        # first candidate uses >=1 eval -> budget out
            self.assertEqual(rep.stopped_reason, "budget_exhausted")

    def test_journal_makes_it_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            jrn = Path(tmp) / "night.jsonl"
            first = run_night_shift(_cands(), _evaluate_for, memory_dir=tmp, llm=None, per_candidate_evals=3,
                                    journal_path=jrn)
            self.assertEqual(first.candidates_run, 3)
            self.assertTrue(jrn.exists())
            # resume with the same journal -> everything already done -> nothing re-run
            second = run_night_shift(_cands(), _evaluate_for, memory_dir=tmp, llm=None, per_candidate_evals=3,
                                     journal_path=jrn)
            self.assertEqual(second.candidates_run, 0)

    def test_repeat_solve_is_not_counted_novel(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = {"task": "grasp", "task_type": "grasp", "gates": {"success_rate": 0.6}, "gate_target": 0.6,
                 "heuristic": _SOLVE}
            dup = [{"id": "a1", "gene": tabletop_arm_gene(), **g},
                   {"id": "a2", "gene": tabletop_arm_gene(), **g}]   # same body twice
            rep = run_night_shift(dup, _evaluate_for, memory_dir=tmp, llm=None, per_candidate_evals=3)
            self.assertGreaterEqual(rep.banked, 1)             # both solve; banking is verified-only
            self.assertEqual(rep.novel, 1)                     # ANNECS-V counts the (identical) body ONCE


if __name__ == "__main__":
    unittest.main()
