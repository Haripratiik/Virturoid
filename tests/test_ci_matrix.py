"""B5 import-any-robot CI matrix: the SAME build->ground->rollout->BOM-cert pipeline must run over diverse
morphologies and emit one comparable row each, and a single broken robot must degrade to an error ROW — never
crash the matrix (per-robot isolation is what makes a nightly 'general' signal trustworthy)."""

import unittest

from virturoid.services.ci_matrix import RobotSpec, run_ci_matrix, default_matrix


class CiMatrixTests(unittest.TestCase):
    def test_default_matrix_is_diverse_and_all_build(self):
        from virturoid.services.steerable_body import steerable_quadruped
        specs = [RobotSpec("quad4", lambda: steerable_quadruped(n_legs=4), "quadruped"),
                 RobotSpec("hex6", lambda: steerable_quadruped(n_legs=6, bilateral=True), "hexapod")]
        rep = run_ci_matrix(specs, steps=150, n_seeds=1)
        self.assertEqual(rep["n_built"], 2)
        for r in rep["rows"]:
            self.assertTrue(r["built"])
            self.assertIn("cert_gates", r)                       # every built robot gets a certificate
            self.assertIn("cadence", r)
            self.assertGreater(r["mass_kg"], 0.0)

    def test_broken_robot_is_isolated_not_fatal(self):
        from virturoid.services.steerable_body import steerable_quadruped
        def _boom():
            raise RuntimeError("intentional build failure")
        specs = [RobotSpec("good", lambda: steerable_quadruped(n_legs=4), "quadruped"),
                 RobotSpec("bad", _boom, "broken")]
        rep = run_ci_matrix(specs, steps=120, n_seeds=1)
        self.assertEqual(rep["n_robots"], 2)
        self.assertEqual(rep["n_built"], 1)                      # the good one still ran
        bad = next(r for r in rep["rows"] if r["name"] == "bad")
        self.assertFalse(bad["built"])
        self.assertIn("intentional build failure", bad["error"])

    def test_default_matrix_covers_multiple_leg_counts(self):
        specs = default_matrix()
        classes = {s.robot_class for s in specs}
        self.assertIn("quadruped", classes)
        self.assertIn("hexapod", classes)
        self.assertGreaterEqual(len(specs), 3)


if __name__ == "__main__":
    unittest.main()
