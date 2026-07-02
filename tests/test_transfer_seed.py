"""Transfer mega-sweep: pick the banked policy that best TRANSFERS (zero-shot forward + upright) to a new body,
to warm-start its training. Encodes the live finding -- a forward-quad residual transfers forward to the hexapod
while hexapod-from-scratch goes backward -- so the ranking must prefer the forward-survived seed."""

import unittest

import tempfile
from pathlib import Path

from virturoid.services.transfer_seed import (best_checkpoint_by_deploy, best_transfer_seed,
                                              gather_banked_policies, transfer_policy_for)


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

    def test_gather_banked_policies_lists_and_dedups(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sub").mkdir()
            (Path(tmp) / "a.npz").write_bytes(b"x")
            (Path(tmp) / "sub" / "b.npz").write_bytes(b"y")
            got = gather_banked_policies(tmp, extra=["/ext/seed.npz"])
            self.assertEqual(got[0], "/ext/seed.npz")                      # extras first
            self.assertTrue(any(g.endswith("a.npz") for g in got) and any(g.endswith("b.npz") for g in got))

    def test_transfer_policy_for_none_when_no_forward(self):
        pol, npz, ranked = transfer_policy_for(gene=None, candidates=[])   # empty pool -> nothing to warm-start from
        self.assertIsNone(pol)
        self.assertIsNone(npz)

    def test_deploy_selection_picks_best_deploying_checkpoint(self):
        # plan v2 T0.1: deploy-sim checkpoint selection — never select on train-sim reward. Encodes the divergence
        # curve (iter-80 deploys WORSE than iter-10), so min_forward is disabled to still return the LEAST-bad if
        # all are negative; here it picks the deploying-forward checkpoint over the latest. (best_checkpoint_by_deploy
        # is best_transfer_seed with this same min_forward=-inf; exercised directly here to inject deterministic evals.)
        table = {
            "hexdec_it10.npz": {"npz": "hexdec_it10.npz", "forward": 0.108, "survived": True, "feature_dim": 24},
            "hexdec_it80.npz": {"npz": "hexdec_it80.npz", "forward": -0.156, "survived": True, "feature_dim": 24},
        }
        best, ranked = best_transfer_seed(None, list(table), min_forward=-1e9, evaluate=lambda g, c: table[c])
        self.assertEqual(best, "hexdec_it10.npz")          # the deploying-forward checkpoint, not the latest
        self.assertTrue(callable(best_checkpoint_by_deploy))   # the T0.1 wrapper exists


if __name__ == "__main__":
    unittest.main()
