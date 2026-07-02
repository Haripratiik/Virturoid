"""Night-shift 3-armed task proposer (plan v2 §5.2): ratio-sampled arm dispatch + AZR learnability band.
Injected arms + probe -> no physics/LLM."""

import unittest

from virturoid.services.night_proposer import (NightProposer, filter_learnable, in_learnability_band,
                                               learnability_reward)


class NightProposerTests(unittest.TestCase):
    def test_learnability_band_and_reward(self):
        self.assertFalse(in_learnability_band(0.0))       # impossible -> reject
        self.assertFalse(in_learnability_band(1.0))       # trivial -> reject
        self.assertTrue(in_learnability_band(0.5))        # hard-but-doable -> accept
        self.assertEqual(learnability_reward(0.0), 0.0)
        self.assertEqual(learnability_reward(1.0), 0.0)
        self.assertAlmostEqual(learnability_reward(0.4), 0.6)   # 1 - rbar

    def test_arms_dispatch_by_ratio(self):
        # counters per arm; each arm returns a tagged candidate
        def arm(name):
            return lambda: {"id": f"{name}", "gene": None, "task": "walk"}
        p = NightProposer(mutate=arm("m"), transfer=arm("t"), explore=arm("e"),
                          ratios=(0.55, 0.30, 0.15), seed=1)
        cands = p.propose(200)
        self.assertEqual(len(cands), 200)
        counts = {a: sum(1 for c in cands if c["arm"] == a) for a in ("mutate", "transfer", "explore")}
        # mutate should dominate, explore rarest (ratios respected within sampling noise)
        self.assertGreater(counts["mutate"], counts["transfer"])
        self.assertGreater(counts["transfer"], counts["explore"])

    def test_unwired_arms_are_skipped(self):
        # LLM explorer off -> only mutate/transfer are drawn; renormalized, never crashes
        p = NightProposer(mutate=lambda: {"id": "m"}, transfer=lambda: {"id": "t"}, explore=None, seed=2)
        cands = p.propose(50)
        self.assertEqual(len(cands), 50)
        self.assertTrue(all(c["arm"] in ("mutate", "transfer") for c in cands))

    def test_filter_learnable_keeps_only_band(self):
        cands = [{"id": "easy"}, {"id": "mid"}, {"id": "hard"}]
        rates = {"easy": 0.95, "mid": 0.5, "hard": 0.05}
        kept = filter_learnable(cands, lambda c: rates[c["id"]])
        self.assertEqual([c["id"] for c in kept], ["mid"])
        self.assertAlmostEqual(kept[0]["learnability"], 0.5)

    def test_empty_when_no_arms(self):
        self.assertEqual(NightProposer().propose(5), [])   # nothing wired -> no candidates, no crash


if __name__ == "__main__":
    unittest.main()
