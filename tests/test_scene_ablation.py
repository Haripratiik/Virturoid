"""S6 held-out split + scaling-ablation harness. family_to_split must expose a disjoint dev/held_out; the ablation
harness must call the injected train/eval fn once per K against a FIXED held-out pool and report the curve +
monotonicity. With a stub that models the real scaling law (more scenes -> better held-out), the curve rises."""

import unittest

from virturoid.services.scene_family import generate_family
from virturoid.services.scene_ablation import family_to_split, scene_count_ablation


class SceneAblationTests(unittest.TestCase):
    def test_family_to_split_is_disjoint(self):
        fam = generate_family("pick_place_sort", n_train=6, n_held_out=3, seed=1)
        sp = family_to_split(fam)
        self.assertTrue(sp["disjoint"])
        self.assertEqual(len(sp["dev"]), 6)
        self.assertEqual(len(sp["held_out"]), 3)
        self.assertEqual(len(set(sp["dev_keys"]) & set(sp["held_out_keys"])), 0)

    def test_ablation_calls_fixed_heldout_per_k(self):
        seen_k, held_ids = [], []
        def stub(train, held, k):
            seen_k.append((k, len(train)))
            held_ids.append(tuple(s.id for s in held))
            return min(1.0, 0.2 + 0.08 * k)                      # the scaling law: more train scenes -> better
        out = scene_count_ablation("pick_place_sort", [1, 2, 4, 8], stub, n_held_out=3, seed=2)
        self.assertEqual([k for k, _ in out["curve"]], [1, 2, 4, 8])
        self.assertEqual(len(set(held_ids)), 1, "held-out pool must be FIXED across all K")
        # the injected law is monotone-up -> harness reports it
        self.assertEqual(out["monotone_frac"], 1.0)
        self.assertGreater(out["curve"][-1][1], out["curve"][0][1])

    def test_ablation_train_pool_excludes_heldout(self):
        # even though train families are generated per-K, none of their structures may be the eval held-out ones
        def stub(train, held, k):
            train_ids = {s.id for s in train}
            self.assertEqual(len(train_ids & {s.id for s in held}), 0)
            return 0.5
        scene_count_ablation("navigation", [2, 4], stub, n_held_out=2, seed=3)


if __name__ == "__main__":
    unittest.main()
