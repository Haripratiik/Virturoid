"""The capstone: evaluate ANY composed robot on its morphology-implied task (§11), one dispatch."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class TaskMatchedEvalTests(unittest.TestCase):
    def _eval(self, prompt):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_matched_eval import evaluate_robot
        return evaluate_robot(compose_robot(prompt), prompt=prompt)

    def test_dispatches_to_the_right_task_per_class(self):
        self.assertEqual(self._eval("a quadruped walking robot")["task"], "locomotion")
        self.assertEqual(self._eval("a mobile base to deliver parts indoors")["task"], "navigation")
        self.assertEqual(self._eval("spray paint a panel with 0.7 m reach")["task"], "spray_coverage")
        self.assertEqual(self._eval("sort red and blue blocks into bins")["task"], "pick_place_sort")

    def test_metrics_are_in_range_and_nonzero_where_expected(self):
        nav = self._eval("a mobile base to deliver parts indoors")
        self.assertGreaterEqual(nav["value"], 0.5)          # the rover reaches most goals
        loco = self._eval("a quadruped walking robot")
        self.assertGreater(loco["value"], 0.1)              # the quadruped walks
        spray = self._eval("spray paint a panel with 0.7 m reach")
        self.assertTrue(0.0 <= spray["value"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
