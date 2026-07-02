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
        self.assertEqual(len(list_tasks()), 10)                # 4 walk + 4 manip + L6 (WS8) + M5 grasp-hard (WS9)
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

    def test_ws9_scripted_grasp_brittle_on_low_friction(self):
        # WS9: M5 lowers the cube friction to 0.35 -> the scripted grasp slips and fails the 0.5 gate, while the
        # SAME scripted grasp passes M1 at friction 1.0. The incumbent provably fails the harder task (Pattern A).
        from virturoid.services.virt_bench_arms import _task_body
        gene = _task_body(get_task("M5_arm_grasp_hard"))
        res = verify_submission("M5_arm_grasp_hard", gene, None)
        self.assertLess(res["metrics"]["success_rate"], 0.5)   # brittle floor on the slippery cube
        self.assertFalse(res["verified_pass"])

    @unittest.skipUnless(__import__("pathlib").Path("build/skills/grasp_tabletop_arm.npz").exists(),
                         "banked grasp residual not present")
    def test_ws9_learned_residual_dispatches_and_beats_the_floor(self):
        # WS9: submitting a learned residual (controller_params.residual_params_path) routes through
        # skill_evaluator._grasp_rollout and lifts the low-friction cube where the scripted floor slips.
        from virturoid.services.virt_bench_arms import _task_body
        gene = _task_body(get_task("M5_arm_grasp_hard"))
        floor = verify_submission("M5_arm_grasp_hard", gene, None)["metrics"]["success_rate"] or 0.0
        r = verify_submission("M5_arm_grasp_hard", gene,
                              {"residual_params_path": "build/skills/grasp_tabletop_arm.npz"})
        self.assertGreaterEqual((r["metrics"]["success_rate"] or 0.0) - floor, 0.15)   # WS9 acceptance: >= +0.15
        self.assertTrue(r["verified_pass"])

    def test_ws8_command_track_constant_gait_fails(self):
        # WS8: L6 scores forward-speed TRACKING vs a fast->stop->faster->reverse command. A constant gait — even a
        # GOOD forward walker that passes L1 — CANNOT track it (can't slow/stop/reverse) -> fails on the track gate.
        from virturoid.services.morph_policy import CPG_DEFAULT
        from virturoid.services.steerable_body import steerable_quadruped
        from virturoid.services.virt_bench_arms import _zero_policy_with_cpg
        gene = steerable_quadruped(n_legs=4)
        res = verify_submission("L6_command_track", gene, _zero_policy_with_cpg(gene, CPG_DEFAULT), steps=450)
        self.assertIn("track", res["metrics"])                 # a real tracking score, from the schedule re-run
        self.assertFalse(res["verified_pass"])                 # a fixed gait cannot follow a varied command
        self.assertIsNotNone(res["verifier"].get("track_err"))  # RMS tracking error recorded for provenance


if __name__ == "__main__":
    unittest.main()
