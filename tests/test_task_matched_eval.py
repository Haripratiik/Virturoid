"""The capstone: evaluate ANY composed robot on its morphology-implied task (§11), one dispatch."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class TaskMatchedEvalTests(unittest.TestCase):
    def _eval(self, prompt, *, walkable: bool = False):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.task_matched_eval import evaluate_robot
        return evaluate_robot(compose_robot(prompt, ensure_walkable=walkable), prompt=prompt)

    def test_dispatches_to_the_right_task_per_class(self):
        self.assertEqual(self._eval("a quadruped walking robot")["task"], "locomotion")
        self.assertEqual(self._eval("a mobile base to deliver parts indoors")["task"], "navigation")
        self.assertEqual(self._eval("spray paint a panel with 0.7 m reach")["task"], "spray_coverage")
        self.assertEqual(self._eval("sort red and blue blocks into bins")["task"], "pick_place_sort")

    def test_metrics_are_in_range_and_nonzero_where_expected(self):
        nav = self._eval("a mobile base to deliver parts indoors")
        self.assertGreaterEqual(nav["value"], 0.5)          # the rover reaches most goals
        # ASK FOR THE WALKABLE BODY (#285), the same way ``test_locomotion`` and ``test_cpg_locomotion`` now do.
        # A bare ``compose_robot`` used to run the walkability gate inside itself and silently hand back the
        # fanned walking template, so this line was measuring THAT and not the composed body. Composing now
        # returns the AUTHORED body and the (unchanged) gate runs for a caller that asks — ``ensure_walkable=True``
        # here, or ``create_robot``, which grounds the body and fits it an operating point before deciding.
        # Measured 2026-08-08 on this prompt, same ``evaluate_robot`` call this assertion makes (metric
        # ``forward_m``): bare compose 0.000 m in 0.15 s, ``ensure_walkable=True`` 1.057 m in 4.59 s. The split
        # is the point — composing is now ~30x cheaper because the walk gate no longer runs on every build.
        # This assertion is about the body the product ships for a WALKING task, so it asks for that body.
        loco = self._eval("a quadruped walking robot", walkable=True)
        self.assertGreater(loco["value"], 0.1)              # the quadruped walks
        spray = self._eval("spray paint a panel with 0.7 m reach")
        self.assertTrue(0.0 <= spray["value"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
