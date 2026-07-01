"""Transfer mega-sweep: pick the banked policy that best TRANSFERS (zero-shot forward + upright) to a new body,
to warm-start its training. Encodes the live finding -- a forward-quad residual transfers forward to the hexapod
while hexapod-from-scratch goes backward -- so the ranking must prefer the forward-survived seed."""

import unittest

from virturoid.services.transfer_seed import best_transfer_seed


class TransferSeedTests(unittest.TestCase):
    def test_picks_the_forward_survived_transfer(self):
        # injected zero-shot results standing in for real rollouts (the measured hexapod case)
        table = {
            "quad_fwd.npz": {"npz": "quad_fwd.npz", "forward": 0.098, "survived": True, "feature_dim": 24},
            "hex_backward.npz": {"npz": "hex_backward.npz", "forward": -0.55, "survived": True, "feature_dim": 24},
            "quad_unstable.npz": {"npz": "quad_unstable.npz", "forward": 0.30, "survived": False, "feature_dim": 24},
        }
        best, ranked = best_transfer_seed(gene=None, candidates=list(table),
                                          evaluate=lambda g, c: table[c])
        self.assertEqual(best, "quad_fwd.npz")             # forward AND survived beats faster-but-fell / backward
        self.assertEqual(ranked[0]["npz"], "quad_fwd.npz")

    def test_none_when_all_transfer_backward(self):
        table = {
            "a.npz": {"npz": "a.npz", "forward": -0.4, "survived": True, "feature_dim": 24},
            "b.npz": {"npz": "b.npz", "forward": 0.01, "survived": True, "feature_dim": 24},   # below min_forward
        }
        best, ranked = best_transfer_seed(gene=None, candidates=list(table),
                                          evaluate=lambda g, c: table[c], min_forward=0.02)
        self.assertIsNone(best)                            # nothing transfers forward -> train from scratch
        self.assertEqual(len(ranked), 2)

    def test_broken_checkpoint_is_skipped(self):
        def flaky(g, c):
            if c == "broken.npz":
                raise ValueError("corrupt npz")
            return {"npz": c, "forward": 0.2, "survived": True, "feature_dim": 24}
        best, ranked = best_transfer_seed(gene=None, candidates=["broken.npz", "good.npz"], evaluate=flaky)
        self.assertEqual(best, "good.npz")                 # a broken candidate just doesn't compete
        self.assertEqual([r["npz"] for r in ranked], ["good.npz"])


if __name__ == "__main__":
    unittest.main()
