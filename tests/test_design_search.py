"""The design-search harness (H4/N1 core): code owns budget/selection/stopping; a pluggable proposer owns
semantics. Verified with deterministic test-double proposer/evaluator (no MuJoCo/LLM) — proves the spine
climbs to a gate pass, respects budget, keeps the best greedily, and reports an honest tree."""

import unittest

from virturoid.services.design_search import run_design_search


def _evaluate(spec):
    """Fake evaluator: the spec's 'forward' scalar becomes a locomotion rollout result."""
    return {"forward": spec["forward"], "cadence": 8.0, "upright_frac": 0.9, "survived": True}


class DesignSearchTests(unittest.TestCase):
    def test_climbs_to_a_gate_pass_and_stops_solved(self):
        def climb(parent, history):
            base = parent.spec["forward"] if parent else 0.0
            return {"forward": round(base + 0.12, 3)}                # 0.12 -> 0.24 -> 0.36 (passes 0.30)
        rep = run_design_search(propose=climb, evaluate=_evaluate, max_evals=10)
        self.assertTrue(rep.solved)
        self.assertEqual(rep.stopped_reason, "solved")
        self.assertEqual(rep.best.verdict, "pass")
        self.assertLessEqual(rep.n_evals, 4)                        # found it fast, under budget

    def test_greedy_keeps_the_best_node(self):
        # a proposer that oscillates; harness must still track the max-fitness node as best
        seq = iter([{"forward": 0.1}, {"forward": 0.25}, {"forward": 0.05}, {"forward": 0.2}])
        rep = run_design_search(propose=lambda p, h: next(seq, None), evaluate=_evaluate, max_evals=4)
        self.assertAlmostEqual(rep.best.spec["forward"], 0.25, places=3)   # the highest-fitness attempt

    def test_no_improvement_stops_and_reports_unsolved(self):
        rep = run_design_search(propose=lambda p, h: {"forward": 0.1}, evaluate=_evaluate,
                                max_evals=50, patience=4)
        self.assertFalse(rep.solved)
        self.assertEqual(rep.stopped_reason, "no_improvement")
        self.assertLess(rep.n_evals, 50)                            # stopped early, didn't burn the whole budget

    def test_budget_is_hard_capped(self):
        seq = iter([{"forward": round(0.02 * i, 3)} for i in range(100)])   # slowly rising, never reaches 0.30 in 5
        rep = run_design_search(propose=lambda p, h: next(seq, None), evaluate=_evaluate,
                                max_evals=5, patience=99)
        self.assertEqual(rep.n_evals, 5)
        self.assertEqual(rep.stopped_reason, "budget_exhausted")

    def test_proposer_exhaustion_is_graceful(self):
        rep = run_design_search(propose=lambda p, h: None, evaluate=_evaluate, max_evals=10)
        self.assertEqual(rep.stopped_reason, "proposer_exhausted")
        self.assertEqual(rep.n_evals, 0)

    def test_tree_and_callback(self):
        banked = []
        def climb(parent, history):
            base = parent.spec["forward"] if parent else 0.0
            return {"forward": round(base + 0.12, 3)}
        rep = run_design_search(propose=climb, evaluate=_evaluate, max_evals=10,
                                on_node=lambda n: banked.append(n.node_id))
        self.assertEqual([t["id"] for t in rep.tree()], list(range(rep.n_evals)))
        self.assertEqual(banked, list(range(rep.n_evals)))          # callback fired per node
        self.assertEqual(rep.tree()[-1]["verdict"], "pass")


if __name__ == "__main__":
    unittest.main()
