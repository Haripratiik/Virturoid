"""Night-shift engine core (WS2/N1): autonomous loop over candidate proposals — runs bounded searches, banks
verified wins, counts ANNECS-V novelty, and resumes from a journal (G8). LLM-free, fake evaluators, no MuJoCo."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from virturoid.fixtures.gene_library import (humanoid_upper_body_gene, quadruped_gene, tabletop_arm_gene)
from virturoid.services.night_shift import laddered_evaluate_for, run_night_shift

_MUJOCO = importlib.util.find_spec("mujoco") is not None

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

    def test_qd_archive_records_niches(self):
        from virturoid.services.qd_archive import QDArchive
        with tempfile.TemporaryDirectory() as tmp:
            arch = QDArchive(dims=[("n_dof", 0, 30), ("success", 0.0, 2.0)], bins=6)
            rep = run_night_shift(_cands(), _evaluate_for, memory_dir=tmp, llm=None, per_candidate_evals=3,
                                  archive=arch)
            self.assertIsNotNone(rep.qd)                        # dashboard snapshot present when an archive is passed
            self.assertEqual(rep.qd["filled"], 2)              # arm + quad banked into distinct DOF niches
            self.assertGreaterEqual(rep.qd["annecs_v"], 1)
            self.assertGreater(rep.qd["qd_score"], 0.0)


class LadderedEvaluateForTests(unittest.TestCase):
    @unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
    def test_production_evaluate_for_screens_then_routes(self):
        # plan v2 §5.1/N8: each candidate gets a CPU-screen -> GPU-rung ladder. Real screen (short quad rollout),
        # STUB GPU rung (inject train/verify) so it runs on CPU. A surviving screen promotes to the hi-fi verdict.
        seen = {}

        def fake_train(gene, *, out_path, iters, envs, cpg, reward_weights, init_npz=None,
                       decimation=1, action_lpf=0.0, sphere_feet=False, contact_dr=False):
            seen["decimation"], seen["action_lpf"] = decimation, action_lpf
            seen["sphere_feet"], seen["contact_dr"] = sphere_feet, contact_dr
            return out_path                                          # "trained" npz

        ev_for = laddered_evaluate_for(cpu_steps=120, decimation=10, action_lpf=0.2, warm_start_pool=None,
                                       train_fn=fake_train,
                                       verify_fn=lambda g, npz, *, steps, decimation=1, action_lpf=0.0,
                                       sphere_feet=False: {"forward": 0.8, "survived": True})
        evaluate = ev_for({"id": "q", "gene": quadruped_gene(), "task": "walk"})
        r = evaluate({"edit_kind": "cpg", "params": {"calf_phase": 1.5708, "freq": 1.5}})
        self.assertIn(r["rung"], ("screen", "hifi"))                 # screened; promoted iff it survived
        if r["rung"] == "hifi":
            self.assertAlmostEqual(r["forward"], 0.8)                # the GPU-rung verdict
            self.assertEqual(seen["decimation"], 10)                 # deploy-gap fixes routed to the trainer
            self.assertEqual(seen["action_lpf"], 0.2)


if __name__ == "__main__":
    unittest.main()
