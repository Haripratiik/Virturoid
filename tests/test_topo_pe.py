"""Phase 5: TopoPE (edit-invariant tree positional code) + kinematic hop-distance — the tokenizer's
structural signals, validated CPU-first. Pure-Python, no deps."""

import unittest

from virturoid.services.topo_pe import hop_distance_matrix, topo_path_vector


class TopoPathVectorTests(unittest.TestCase):
    def test_root_is_zero_and_function_is_deterministic(self):
        self.assertEqual(topo_path_vector([]), [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(topo_path_vector([0, 1, 0]), topo_path_vector([0, 1, 0]))

    def test_distinct_paths_differ_shared_prefix_overlaps(self):
        left, right = topo_path_vector([0]), topo_path_vector([1])
        self.assertNotEqual(left, right)                       # different limbs -> different codes
        # a child's code shares its parent-edge component (same first-prefix hash), so same-subtree parts cluster
        parent = topo_path_vector([0])
        child = topo_path_vector([0, 0])
        shared = sum(1 for a, b in zip(parent, child) if a != 0 and b != 0 and (a > 0) == (b > 0))
        self.assertGreaterEqual(shared, 1)

    def test_edit_invariance_is_a_pure_function_of_the_path(self):
        # the whole point: a part's code depends ONLY on its own root->part path, so growing the tree
        # elsewhere (new siblings/limbs) cannot change it
        self.assertEqual(topo_path_vector([1, 0]), topo_path_vector([1, 0]))
        self.assertEqual(len(topo_path_vector([2, 3, 1], dim=6)), 6)


class HopDistanceTests(unittest.TestCase):
    def test_chain_distances(self):
        # t0 (base) -> t1 -> t2
        d = hop_distance_matrix([-1, 0, 1])
        self.assertEqual(d[0][0], 0)
        self.assertEqual(d[0][1], 1)
        self.assertEqual(d[0][2], 2)
        self.assertEqual(d[1][2], 1)
        self.assertEqual(d[2][0], 2)                            # symmetric

    def test_virtual_base_connects_separate_limbs(self):
        # two limbs off the base: t0->t1 and t2->t3 (t0, t2 both base-attached)
        d = hop_distance_matrix([-1, 0, -1, 2])
        self.assertEqual(d[0][2], 2)                            # base-attached siblings: through the virtual base
        self.assertEqual(d[1][3], 4)                            # t1-t0-[base]-t2-t3
        self.assertEqual(d[0][1], 1)

    def test_empty(self):
        self.assertEqual(hop_distance_matrix([]), [])


if __name__ == "__main__":
    unittest.main()
