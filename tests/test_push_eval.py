"""§4.5 — a real NON-PREHENSILE push: a closed gripper sweeps the box toward a target (no grasp). Success is
measured by displacement toward the target, not a grasp. The mechanism genuinely pushes (mean progress ~0.4);
reliable target-directed pushing across all geometries is still improving (an honest frontier)."""

import unittest

from virturoid.services.morphology_composer import compose_robot
from virturoid.services.push_eval import evaluate_push, push_attempt


class PushEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gene = compose_robot("grasp and lift a box on a table", llm=None)

    def test_push_moves_object_toward_target(self):
        r = evaluate_push(self.gene)
        self.assertEqual(r["attempts"], 3)
        self.assertGreater(r["mean_progress"], 0.25)              # genuinely pushes the box toward the targets
        self.assertTrue(any(c["progress"] > 0.5 for c in r["cases"]))   # at least one near-target push
        for c in r["cases"]:                                     # a NON-prehensile metric: displacement, not grasp
            self.assertIn("progress", c)
            self.assertIn("moved_m", c)

    def test_single_push_attempt_shape(self):
        r = push_attempt(self.gene, (0.38, -0.06), (0.38, 0.10))
        for k in ("success", "progress", "moved_m", "final_dist_to_target", "start_dist"):
            self.assertIn(k, r)


if __name__ == "__main__":
    unittest.main()
