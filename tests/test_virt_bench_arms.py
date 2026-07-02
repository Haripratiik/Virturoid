"""VIRT-Bench arms (WS3): Arm A (fixed-pipeline baseline) vs Arm B (our CPG-search harness), both scored by the
INDEPENDENT verifier. The A->B delta is the measured value of the search harness; a task both fail is the next
build priority (honest compass)."""

import unittest

from virturoid.services import virt_bench_arms
from virturoid.services.virt_bench_arms import run_arm_a, run_arm_b, run_dev_scoreboard, run_head_to_head


class VirtBenchArmsTests(unittest.TestCase):
    def test_both_arms_return_verified_verdicts(self):
        a = run_arm_a("L1_quad_walk", steps=120)
        b = run_arm_b("L1_quad_walk", steps=120, max_evals=4, use_memory=False)  # search-only: deterministic
        for r, arm in ((a, "A"), (b, "B")):
            self.assertEqual(r["arm"], arm)
            self.assertIsInstance(r["verified_pass"], bool)      # a real re-run verdict, not a self-claim
            self.assertIn("forward_m", r["metrics"])
        # Arm B actually SEARCHED and selected a CPG (the harness ran), then submitted it for verification
        self.assertGreaterEqual(b["n_evals"], 1)
        self.assertIsNotNone(b["searched"])

    def test_arm_b_memory_recall_is_verified(self):
        # the MEMORY path: with a controlled pool holding a forward-transferring seed, Arm B recalls + VERIFIES it.
        # build/models holds the banked forward-quad policy that transfers forward to the quad body.
        import os
        if not os.path.isdir("build/models"):
            self.skipTest("no banked models dir")
        b = run_arm_b("L1_quad_walk", steps=200, max_evals=2, use_memory=True, models_dir="build/models")
        self.assertEqual(b["arm"], "B")
        self.assertIsInstance(b["verified_pass"], bool)          # verifier re-ran whichever candidate won
        # if memory won, `recalled` names the seed; either way the result is an independent verify, not a claim
        self.assertIn("forward_m", b["metrics"])

    def test_dev_scoreboard_is_honest(self):
        sb = run_dev_scoreboard(steps=120, max_evals=4, use_memory=False)
        self.assertGreaterEqual(sb["n_tasks"], 1)                # at least L1 in the dev locomotion split
        # every solved count is grounded in an independent verify, never a self-report
        self.assertEqual(sb["harness_delta"], sb["B_solved"] - sb["A_solved"])
        self.assertTrue(all(isinstance(r["A_pass"], bool) and isinstance(r["B_pass"], bool) for r in sb["rows"]))


class ArmBGpuRungTests(unittest.TestCase):
    def test_gpu_rung_composes_and_verifies(self):
        # WS1/#66: with use_gpu, run_arm_b promotes the search winner to a GPU-trained residual, then INDEPENDENTLY
        # verifies it at the frozen horizon. Inject a stub trainer that "returns" the banked forward-quad so the
        # rung is exercised on CPU (no real GPU): the GPU candidate then verifies forward and wins.
        import os
        if not os.path.isfile("build/models/quaddec_fwd.npz"):
            self.skipTest("no banked forward-quad to stand in for the GPU-trained policy")
        seen = {}

        def stub_hifi(spec):
            seen["spec"] = spec
            return {"trained": True, "npz": "build/models/quaddec_fwd.npz", "forward": 0.668, "survived": True}

        b = run_arm_b("L1_quad_walk", steps=120, max_evals=2, use_memory=False, use_gpu=True,
                      gpu_iters=40, gpu_hifi=stub_hifi)
        self.assertTrue(b["verified_pass"])                    # the GPU-trained (stub) policy verified forward
        self.assertIn("GPU residual", b["method"])             # the GPU candidate won the best-verified selection
        self.assertEqual(b["gpu_npz"], "build/models/quaddec_fwd.npz")
        self.assertEqual(b["budget"]["gpu_iters"], 40)         # N16 ledger records the GPU spend
        self.assertIn("spec", seen)                            # the search winner was handed to the trainer

    def test_gpu_off_is_unchanged(self):
        # default use_gpu=False -> no GPU candidate, budget records 0 GPU iters (CPU-only arm behaves as before)
        b = run_arm_b("L1_quad_walk", steps=120, max_evals=2, use_memory=False)
        self.assertEqual(b["budget"]["gpu_iters"], 0)
        self.assertIsNone(b.get("gpu_npz"))


class HeadToHeadTests(unittest.TestCase):
    def test_three_arm_deltas_and_honesty_aggregate_correctly(self):
        # Stub the three arms + task list so we test the AGGREGATION (deltas + honesty), not re-run physics.
        # Scenario over 3 tasks: A solves 1, A+ solves 2 (one an over-claim: claim yes / verify no), B solves 3.
        import virturoid.services.virt_bench_arms as M
        tasks = [{"id": "t1", "family": "locomotion"}, {"id": "t2", "family": "locomotion"},
                 {"id": "t3", "family": "locomotion"}]

        def fake_list_tasks(split):
            return tasks

        # per-task scripted verdicts: (verified_pass, claimed_pass)
        A = {"t1": (True, True), "t2": (False, False), "t3": (False, False)}
        Aplus = {"t1": (True, True), "t2": (True, True), "t3": (False, True)}   # t3 = OVERCLAIM (claim>verify)
        B = {"t1": (True, True), "t2": (True, True), "t3": (True, True)}

        def fake_arm_a(task_id, **kw):
            v, c = A[task_id]
            return {"verified_pass": v, "claimed_pass": c, "metrics": {"forward_m": 0.5 if v else 0.0}}

        def fake_arm_b(task_id, *, use_memory, **kw):
            v, c = (B if use_memory else Aplus)[task_id]
            return {"verified_pass": v, "claimed_pass": c, "metrics": {"forward_m": 0.6 if v else 0.0},
                    "recalled": "seed.npz" if use_memory else None, "searched": {"freq": 1.5}}

        orig = (M.list_tasks, M.run_arm_a, M.run_arm_b)
        M.list_tasks, M.run_arm_a, M.run_arm_b = fake_list_tasks, fake_arm_a, fake_arm_b
        try:
            r = run_head_to_head(split=None, families=("locomotion",))
        finally:
            M.list_tasks, M.run_arm_a, M.run_arm_b = orig

        self.assertEqual((r["A_solved"], r["Aplus_solved"], r["B_solved"]), (1, 2, 3))
        self.assertEqual(r["harness_delta"], 1)                   # A+ - A = 2 - 1
        self.assertEqual(r["transfer_delta"], 1)                  # B  - A+ = 3 - 2
        # honesty: A+ claimed 3 but verified 2 -> overclaim 1 (the deploy-gap detector); A and B honest (0)
        self.assertEqual(r["honesty"]["A+"]["overclaim"], 1)
        self.assertEqual(r["honesty"]["A"]["overclaim"], 0)
        self.assertEqual(r["honesty"]["B"]["overclaim"], 0)
        self.assertEqual(len(r["rows"]), 3)

    def test_prereg_manifest_is_written_and_hashed(self):
        import json
        import tempfile
        from pathlib import Path
        import virturoid.services.virt_bench_arms as M
        # stub arms so this is fast; we only assert the pre-registration side-effect
        M2 = (M.run_arm_a, M.run_arm_b, M.list_tasks)
        M.run_arm_a = lambda tid, **kw: {"verified_pass": False, "claimed_pass": False, "metrics": {"forward_m": 0.0}}
        M.run_arm_b = lambda tid, **kw: {"verified_pass": False, "claimed_pass": False, "metrics": {"forward_m": 0.0},
                                         "recalled": None, "searched": None}
        M.list_tasks = lambda split: []
        try:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / "prereg.json"
                r = run_head_to_head(split="held_out", prereg_path=str(p))
                self.assertTrue(p.exists())
                self.assertIsInstance(r["prereg"], str)          # returned the manifest_hash
                man = json.loads(p.read_text())
                self.assertEqual(man["manifest_hash"], r["prereg"])
                self.assertIn("arms_sha256", man)                # the arms module is version-pinned in the freeze
        finally:
            M.run_arm_a, M.run_arm_b, M.list_tasks = M2


if __name__ == "__main__":
    unittest.main()
