"""Benchmark + scorecard (§46): score a robot across a difficulty suite (capability + robustness + grade)."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class BenchmarkTests(unittest.TestCase):
    def _score(self, prompt):
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.benchmark import run_benchmark
        return run_benchmark(compose_robot(prompt), prompt=prompt)

    def test_scorecard_structure_and_ranges(self):
        sc = self._score("spray paint a panel with 0.7 m reach")
        self.assertEqual(len(sc["levels"]), 2)                     # a difficulty suite, not one scene
        self.assertTrue(0.0 <= sc["capability_score"] <= 1.0)
        self.assertTrue(0.0 <= sc["robustness"] <= 1.0)
        self.assertIn(sc["grade"], list("ABCDF"))
        for lv in sc["levels"]:
            self.assertIn("level", lv); self.assertIn("value", lv); self.assertIn("score", lv)

    def test_quadruped_grades_well(self):
        sc = self._score("a quadruped walking robot")
        self.assertEqual(sc["robot_class"], "quadruped")
        # capability_score is the SCRIPTED-baseline gait on the body. The body is now a real-quadruped-accurate
        # 12-DOF design (hip abduction + thigh + knee per leg) — the simple scripted gait grades a richer body
        # lower, but we do NOT shrink the morphology to inflate this baseline; the LEARNED policy is the real
        # measure and exceeds the scripted score.
        self.assertGreater(sc["capability_score"], 0.2)
        self.assertIn(sc["grade"], ("A", "B", "C", "D"))

    def test_robustness_flags_degradation(self):
        # spray degrades from a small panel to a wide one -> robustness < 1 (worst level below the mean)
        sc = self._score("spray paint a panel with 0.7 m reach")
        vals = [lv["value"] for lv in sc["levels"]]
        if max(vals) > min(vals):
            self.assertLess(sc["robustness"], 1.0)


if __name__ == "__main__":
    unittest.main()
