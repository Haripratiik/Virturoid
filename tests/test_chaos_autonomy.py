"""Plan v3 M6 — autonomy chaos-test. WS2 shipped FIVE integration bugs that unit tests missed because they only
surface when the loop RUNS on real hardware (a golden-protected random policy; the ladder gated on pure-CPG
survival; a short-horizon recall; an un-uploaded seed; verifying the final not the deploy-best checkpoint). This
suite institutionalizes that lesson: it INJECTS the fault classes (GPU down, trainer crash/timeout, corrupt
checkpoint, un-compilable gene, GPU crash on the expensive rung) and asserts the loop degrades GRACEFULLY — a
clean floor / cheap reject, never a crash and never a silent bank of garbage. No GPU."""

import unittest

from virturoid.services.fidelity_ladder import make_gpu_locomotion_hifi, make_laddered_evaluate


class _Boom(Exception):
    """A stand-in for a CUDA OOM / ssh drop / corrupt-file error raised deep in the trainer or verifier."""


_SURVIVE_VERIFY = lambda *a, **k: {"survived": True, "forward": 0.9, "cadence": 6.0, "upright_frac": 0.9}


class ChaosLadderTests(unittest.TestCase):
    def test_gpu_down_is_an_honest_floor(self):
        # train_fn returns falsy (box unreachable) -> survived=False, trained=False, nothing to bank
        hifi = make_gpu_locomotion_hifi(gene=object(), train_fn=lambda *a, **k: None, verify_fn=_SURVIVE_VERIFY)
        r = hifi({"params": {}})
        self.assertFalse(r["survived"])
        self.assertFalse(r["trained"])
        self.assertIsNone(r["npz"])

    def test_trainer_crash_degrades_not_raises(self):
        def boom(*a, **k):
            raise _Boom("CUDA out of memory at iter 50")
        hifi = make_gpu_locomotion_hifi(gene=object(), train_fn=boom, verify_fn=_SURVIVE_VERIFY)
        r = hifi({"params": {}})                                # MUST NOT raise
        self.assertFalse(r["survived"])
        self.assertIn("train_error", r["note"])

    def test_corrupt_checkpoint_verify_crash_degrades(self):
        def boom(*a, **k):
            raise _Boom("np.load: bad magic number (truncated npz)")
        hifi = make_gpu_locomotion_hifi(gene=object(), train_fn=lambda *a, **k: "x.npz", verify_fn=boom)
        r = hifi({"params": {}})                                # MUST NOT raise
        self.assertFalse(r["survived"])                         # a corrupt checkpoint is never banked
        self.assertIn("verify_error", r["note"])

    def test_uncompilable_gene_is_a_cheap_reject(self):
        def screen_boom(spec):
            raise _Boom("MJCF compile error: duplicate joint name")
        evaluate = make_laddered_evaluate(screen_boom, lambda s: {"survived": True, "forward": 1.0})
        r = evaluate({"params": {}})                            # MUST NOT raise
        self.assertFalse(r["survived"])
        self.assertFalse(r["promoted"])                        # the expensive rung is never reached
        self.assertIn("screen_error", r["note"])

    def test_hifi_crash_falls_back_to_the_screen_verdict(self):
        def hifi_boom(spec):
            raise _Boom("ssh connection dropped mid-training")
        evaluate = make_laddered_evaluate(lambda s: {"survived": True, "forward": 0.05, "cadence": 4.0}, hifi_boom)
        r = evaluate({"params": {}})                            # MUST NOT raise
        self.assertFalse(r["promoted"])
        self.assertEqual(r["rung"], "screen")                  # kept the cheap verdict, no crash
        self.assertIn("hifi_error", r["note"])


class ChaosSearchTests(unittest.TestCase):
    def test_gpu_down_never_certifies_a_walk(self):
        # the whole point of the honest floor: a GPU-down expensive rung must never let the search SOLVE a task
        # (that was the class of the WS2 "banked a policy that didn't walk" bugs).
        from virturoid.services.design_search import run_design_search
        hifi = make_gpu_locomotion_hifi(gene=object(), train_fn=lambda *a, **k: None, verify_fn=_SURVIVE_VERIFY)
        laddered = make_laddered_evaluate(
            lambda s: {"survived": True, "forward": 0.05, "cadence": 4.0, "upright_frac": 0.9}, hifi)

        def propose(parent, history):
            return {"edit_kind": "cpg", "params": {"calf_phase": 0.0}} if parent is None else None

        rep = run_design_search(propose=propose, evaluate=laddered, task_type="locomotion",
                                gates={"forward_m": 0.5, "cadence": 3.0, "upright": 0.6}, max_evals=3)
        self.assertFalse(rep.solved)                           # a GPU-down floor never passes the gate


if __name__ == "__main__":
    unittest.main()
