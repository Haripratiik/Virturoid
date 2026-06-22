import dataclasses
import unittest

from virturoid.fixtures.components import extended_component_library
from virturoid.fixtures.gene_library import tabletop_arm_gene
from virturoid.services.buildability import (
    assess_buildability,
    ground_actuators,
    transferability_score,
)


class BuildabilityTests(unittest.TestCase):
    def test_seed_arm_grounds_to_real_actuators(self):
        g = tabletop_arm_gene()
        out = ground_actuators(g, extended_component_library())
        self.assertTrue(out["grounded"], out["issues"])
        # every actuated joint got a real actuator whose stall torque covers its need
        for r in out["joint_actuators"]:
            self.assertIsNotNone(r["actuator"])
            self.assertGreaterEqual(r["stall_nm"], r["need_nm"])

    def test_impossible_torque_is_flagged(self):
        g = tabletop_arm_gene()
        beefy = dataclasses.replace(g, segments=[
            dataclasses.replace(s, actuator_torque_nm=500.0) if s.joint_type == "revolute" else s
            for s in g.segments])
        out = ground_actuators(beefy, extended_component_library())
        self.assertFalse(out["grounded"])
        self.assertTrue(any("strongest actuator" in i for i in out["issues"]))

    def test_confidence_combines_factors_and_is_bounded(self):
        good = {"min_safety_factor": 10.0, "ok": True}
        weak = {"min_safety_factor": 0.5, "ok": False}
        s_full = transferability_score(True, good, 0.9)
        s_low = transferability_score(False, weak, 0.0)
        self.assertGreater(s_full, s_low)
        for s in (s_full, s_low):
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_full_assessment_shape(self):
        out = assess_buildability(tabletop_arm_gene(), extended_component_library(), sim_success=0.8)
        self.assertIn("buildable", out)
        self.assertIn("confidence", out)
        self.assertIn("joint_actuators", out["parts"])
        self.assertIn("min_safety_factor", out["structural"])
        self.assertIn("disclaimer", out)  # honesty: not a transfer guarantee
        self.assertTrue(out["buildable"])  # the seed arm is buildable


if __name__ == "__main__":
    unittest.main()
