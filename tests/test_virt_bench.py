"""VIRT-Bench (WS3): the frozen task registry + the independent verifier that RE-RUNS a submission in physics
and applies the honesty gate (never the arm's self-claim). Locomotion verification uses a real rollout."""

import unittest

from virturoid.fixtures.gene_library import quadruped_gene
from virturoid.services.virt_bench import get_task, list_tasks, verify_submission


class VirtBenchTests(unittest.TestCase):
    def test_task_registry(self):
        self.assertTrue(list_tasks())
        self.assertTrue(all(k in get_task("L1_quad_walk")
                            for k in ("family", "task_type", "gates", "split", "steps", "seed")))
        self.assertEqual({t["id"] for t in list_tasks("dev")},
                         {"L1_quad_walk", "L4_decapod_walk", "M1_arm_grasp", "M4_arm_transport"})
        self.assertEqual(len(list_tasks()), 8)                 # WS5: 4 locomotion + 4 manipulation
        with self.assertRaises(KeyError):
            get_task("nope")

    def test_verifier_records_frozen_provenance(self):
        # §3.1/§3.2: the verdict records WHICH sim + the FROZEN horizon/seed + control decimation it ran under
        res = verify_submission("L1_quad_walk", quadruped_gene(), policy=None, steps=120, decimation=5)
        v = res["verifier"]
        self.assertEqual(v["sim"], "cpu-mujoco")
        self.assertEqual(v["steps"], 120)                  # explicit override honored (tests); default = task's 900
        self.assertEqual(v["seed"], get_task("L1_quad_walk")["seed"])   # seed is FROZEN in the task, not caller-set
        self.assertEqual(v["decimation"], 5)

    def test_verifier_reruns_and_gates_honestly(self):
        # submit a quad with the DEFAULT (untrained) controller -> the verifier re-runs it and honestly gates
        res = verify_submission("L1_quad_walk", quadruped_gene(), policy=None, steps=120)
        self.assertEqual(res["task"], "L1_quad_walk")
        self.assertIn("verified_pass", res)
        self.assertIsInstance(res["verified_pass"], bool)     # a real verdict from a real rollout, not self-claimed
        self.assertIn(res["failure_mode"],
                      ("walking", "weak_forward", "walks_backward", "shuffle", "leaning", "fell"))
        self.assertIn("forward_m", res["metrics"])            # verified metrics from the re-run

    def test_ws3_per_body_upright_and_support(self):
        # WS3: many-legged tasks keep upright 0.6 (now MORPHOLOGY-relative via per-body tau) + add a tripod
        # ``support`` gate so the bar moves SIDEWAYS not down; quad L1 has NO support gate (byte-identical).
        from virturoid.services.morph_policy import upright_height_ratio
        self.assertNotIn("support", get_task("L1_quad_walk")["gates"])          # quad untouched
        for tid in ("L2_hex_walk", "L3_octopod_walk", "L4_decapod_walk"):
            g = get_task(tid)["gates"]
            self.assertEqual(g["upright"], 0.6)                                  # consistent bar, now per-body
            self.assertIn("support", g)                                          # harder tripod companion
        self.assertAlmostEqual(upright_height_ratio(4), 0.70)                    # quad unchanged (L1 preserved)
        self.assertAlmostEqual(upright_height_ratio(2), 0.70)                    # biped stands tall
        self.assertAlmostEqual(upright_height_ratio(6), 0.55)                    # hexapod low tripod
        self.assertAlmostEqual(upright_height_ratio(8), 0.50)                    # octopod
        self.assertAlmostEqual(upright_height_ratio(10), 0.50)                   # decapod


if __name__ == "__main__":
    unittest.main()
