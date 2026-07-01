"""The real-physics adapter: the harness searches CPG params on an actual body via CPU MuJoCo rollouts, with an
LLM-free grid proposer (night-shift N1 core). Uses MuJoCo (like the other locomotion tests); kept short."""

import unittest

from virturoid.services.design_search import run_design_search
from virturoid.services.search_adapters import cpg_grid_proposer, make_cpg_evaluate
from virturoid.services.steerable_body import steerable_quadruped


class SearchAdapterTests(unittest.TestCase):
    def test_harness_searches_cpg_on_a_real_body(self):
        gene = steerable_quadruped(n_legs=4, dofs_per_leg=3)
        evaluate = make_cpg_evaluate(gene, steps=120)
        proposer = cpg_grid_proposer([{"calf_phase": 0.0, "freq": 1.5},
                                      {"calf_phase": 1.5708, "freq": 1.5}])
        rep = run_design_search(propose=proposer, evaluate=evaluate, task_type="locomotion", max_evals=5,
                                gates={"forward_m": 0.05, "cadence": 1.0, "upright": 0.4})
        self.assertEqual(rep.n_evals, 2)                        # 2-point grid, then proposer exhausts
        self.assertIn(rep.stopped_reason, ("solved", "proposer_exhausted", "no_improvement"))
        self.assertIsNotNone(rep.best)
        self.assertIn("cpg", rep.best.result)                   # a real rollout result is attached
        self.assertIn("calf_phase", rep.best.result["cpg"])
        # each node has a real diagnosis artifact with a locomotion failure mode
        self.assertIn(rep.best.artifact["failure_mode"],
                      ("walking", "weak_forward", "walks_backward", "shuffle", "leaning", "fell"))


if __name__ == "__main__":
    unittest.main()
