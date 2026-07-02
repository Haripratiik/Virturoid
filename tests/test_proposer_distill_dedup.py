"""Plan v3 WS12 — dedup-to-DISTINCT + distillation readiness. The rejection-sampling-SFT gain is driven by the
count of DISTINCT verified designs, not raw volume, so the banked corpus must be de-duplicated and gated on the
~1000-distinct evidence anchor before a (GPU) SFT run is worth it. Pure aggregation; no GPU/LLM."""

import unittest

from virturoid.services.proposer_distill import dedup_distinct, distillation_readiness


def _ex(prompt, design_json):
    return {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": design_json}]}


class DistillDedupTests(unittest.TestCase):
    def test_near_duplicate_designs_collapse(self):
        exs = [
            _ex("a quadruped that walks", '{"robot_class":"quad","legs":4,"len":0.50}'),
            _ex("a quadruped that walks", '{"robot_class":"quad","legs":4,"len":0.501}'),  # within rounding -> dup
            _ex("a hexapod that walks", '{"robot_class":"hex","legs":6,"len":0.50}'),
        ]
        distinct = dedup_distinct(exs)
        self.assertEqual(len(distinct), 2)                    # the two near-dup quads collapse to one

    def test_readiness_gate(self):
        exs = [
            _ex("a quadruped that walks", '{"c":"quad","d":0.5}'),
            _ex("a hexapod that walks", '{"c":"hex","d":0.5}'),
        ]
        r = distillation_readiness(exs, target_distinct=5)
        self.assertEqual(r["n_distinct"], 2)
        self.assertEqual(r["n_prompt_families"], 2)
        self.assertFalse(r["ready"])                          # below the anchor -> "collect more"
        self.assertIn("collect more", r["verdict"])
        self.assertTrue(distillation_readiness(exs, target_distinct=2)["ready"])   # meets a lower bar

    def test_empty_is_honest(self):
        r = distillation_readiness([], target_distinct=1000)
        self.assertEqual(r["n_distinct"], 0)
        self.assertFalse(r["ready"])
        self.assertIn("no verified designs", r["verdict"])


if __name__ == "__main__":
    unittest.main()
