"""Harness banking: a verified design-search win is banked to the flywheel and RECALLED for the same body on
the next search — the compounding/transfer loop (the moat metric) in miniature. Fake evaluator, no MuJoCo/LLM."""

import tempfile
import unittest

from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.services.engineer_mode import run_engineer_search
from virturoid.services.harness_banking import bank_search_result


def _grasp_eval(spec):
    sr = 0.85 if spec.get("params", {}).get("kp", 0) >= 40 else 0.2
    return {"success_rate": sr, "contacted": True, "lifted": sr > 0.5}


def _solving_heuristic(parent, history):
    return {"edit_kind": "gains", "params": {"kp": 45.0}}


class HarnessBankingTests(unittest.TestCase):
    def test_bank_then_recall_closes_the_flywheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            gene = tabletop_arm_gene()
            rep = run_engineer_search(task="grasp a cube", gene=gene, evaluate=_grasp_eval, llm=None,
                                      task_type="grasp", max_evals=3, gates={"success_rate": 0.6},
                                      heuristic=_solving_heuristic)
            self.assertTrue(rep.solved)
            banked = bank_search_result(rep, gene=gene, memory_dir=tmp, task_type="grasp", gate_target=0.6)
            self.assertTrue(banked["banked"])
            # the verified win is now recalled for the SAME body -> warm-start transfer next time
            from virturoid.services.knowledge_context import assemble_prior_knowledge
            block = assemble_prior_knowledge(tmp, gene=gene, task_type="grasp")
            self.assertIn("grasp", block.lower())

    def test_unsolved_search_banks_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            gene = tabletop_arm_gene()
            # heuristic proposes a weak gain -> never passes -> no verified win
            rep = run_engineer_search(task="grasp", gene=gene, evaluate=_grasp_eval, llm=None, task_type="grasp",
                                      max_evals=3, gates={"success_rate": 0.6}, patience=2,
                                      heuristic=lambda p, h: {"edit_kind": "gains", "params": {"kp": 10.0}})
            self.assertFalse(rep.solved)
            banked = bank_search_result(rep, gene=gene, memory_dir=tmp, task_type="grasp", gate_target=0.6)
            self.assertFalse(banked["banked"])
            self.assertIn("no verified win", banked["reason"])


if __name__ == "__main__":
    unittest.main()
