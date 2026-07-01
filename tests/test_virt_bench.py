"""VIRT-Bench (WS3): the frozen task registry + the independent verifier that RE-RUNS a submission in physics
and applies the honesty gate (never the arm's self-claim). Locomotion verification uses a real rollout."""

import unittest

from virturoid.fixtures.gene_library import quadruped_gene
from virturoid.services.virt_bench import get_task, list_tasks, verify_submission


class VirtBenchTests(unittest.TestCase):
    def test_task_registry(self):
        self.assertTrue(list_tasks())
        self.assertTrue(all(k in get_task("L1_quad_walk") for k in ("family", "task_type", "gates", "split")))
        self.assertEqual({t["id"] for t in list_tasks("dev")}, {"L1_quad_walk", "M1_arm_grasp"})
        with self.assertRaises(KeyError):
            get_task("nope")

    def test_verifier_reruns_and_gates_honestly(self):
        # submit a quad with the DEFAULT (untrained) controller -> the verifier re-runs it and honestly gates
        res = verify_submission("L1_quad_walk", quadruped_gene(), policy=None, steps=120)
        self.assertEqual(res["task"], "L1_quad_walk")
        self.assertIn("verified_pass", res)
        self.assertIsInstance(res["verified_pass"], bool)     # a real verdict from a real rollout, not self-claimed
        self.assertIn(res["failure_mode"],
                      ("walking", "weak_forward", "walks_backward", "shuffle", "leaning", "fell"))
        self.assertIn("forward_m", res["metrics"])            # verified metrics from the re-run


if __name__ == "__main__":
    unittest.main()
