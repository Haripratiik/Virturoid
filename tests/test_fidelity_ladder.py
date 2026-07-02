"""Fidelity ladder (WS1/H4): the screen->route composer must spend the expensive rung ONLY on candidates that
survive the cheap screen. Verified with stubs (no GPU): a faller is rejected cheaply; a survivor is promoted."""

import unittest

from virturoid.services.fidelity_ladder import make_gpu_locomotion_hifi, make_laddered_evaluate


class FidelityLadderTests(unittest.TestCase):
    def test_faller_rejected_cheaply_survivor_promoted(self):
        hifi_calls = []

        def screen(spec):
            # a spec flagged fall=True fails the screen; otherwise it survives
            return {"forward": 0.05, "cadence": 4.0, "survived": not spec.get("fall", False)}

        def hifi(spec):
            hifi_calls.append(spec)
            return {"forward": 0.9, "cadence": 6.0, "survived": True}   # the GPU-trained residual walks

        evaluate = make_laddered_evaluate(screen, hifi)

        faller = evaluate({"fall": True})
        self.assertEqual(faller["rung"], "screen")
        self.assertFalse(faller["promoted"])

        survivor = evaluate({"fall": False})
        self.assertEqual(survivor["rung"], "hifi")
        self.assertTrue(survivor["promoted"])
        self.assertAlmostEqual(survivor["forward"], 0.9)          # the hi-fi verdict, not the screen's 0.05
        self.assertIn("screen", survivor)                          # screen summary carried for diagnosis

        self.assertEqual(len(hifi_calls), 1)                       # GPU rung touched ONCE (only the survivor)

    def test_gpu_hifi_falls_back_honestly_when_training_fails(self):
        # inject a train_fn that returns None (GPU down / train failed) -> honest survived=False floor, no crash
        hifi = make_gpu_locomotion_hifi(gene=object(), train_fn=lambda *a, **k: None,
                                        verify_fn=lambda *a, **k: {"survived": True})
        r = hifi({"params": {"calf_phase": 0.0, "fwd_gate_w": 0.85}})
        self.assertFalse(r["survived"])
        self.assertFalse(r["trained"])
        self.assertEqual(r["forward"], 0.0)

    def test_gpu_hifi_verifies_trained_policy(self):
        # a successful train returns an npz path; the ladder then VERIFIES it via verify_fn (independent re-run)
        seen = {}

        def fake_train(gene, *, out_path, iters, envs, cpg, reward_weights, init_npz=None,
                       decimation=1, action_lpf=0.0):
            seen["cpg"] = cpg
            seen["reward_weights"] = reward_weights
            seen["init_npz"] = init_npz
            seen["decimation"] = decimation
            seen["action_lpf"] = action_lpf
            return out_path                                       # "trained" -> npz path

        hifi = make_gpu_locomotion_hifi(gene=object(), init_npz="seed.npz", decimation=10, action_lpf=0.2,
                                        train_fn=fake_train,
                                        verify_fn=lambda g, npz, *, steps, decimation=1, action_lpf=0.0:
                                        {"forward": 0.7, "survived": True, "dec_seen": decimation})
        r = hifi({"params": {"calf_phase": 0.0, "freq": 1.5, "fwd_gate_w": 0.85, "prog_w": 6.0}})
        self.assertTrue(r["trained"])
        self.assertAlmostEqual(r["forward"], 0.7)
        self.assertEqual(seen["cpg"], {"calf_phase": 0.0, "freq": 1.5})    # cpg edits routed to trainer
        self.assertEqual(seen["reward_weights"]["fwd_gate_w"], 0.85)       # reward edits routed to trainer
        self.assertEqual(seen["reward_weights"]["prog_w"], 6.0)
        self.assertEqual(seen["init_npz"], "seed.npz")                     # warm-start seed routed to trainer
        self.assertEqual(seen["decimation"], 10)                           # T1.1 decimation routed to trainer
        self.assertEqual(seen["action_lpf"], 0.2)                          # T1.2 LPF routed to trainer
        self.assertEqual(r["dec_seen"], 10)                                # verify deploys at the SAME rate (deploy==train)


if __name__ == "__main__":
    unittest.main()
