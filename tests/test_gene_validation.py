"""Gene-design validation gate: composes existing engineering checks + rest-pose stability/collision into one
honest flag report that ANNOTATES (never hard-rejects) a built RobotGene."""

import importlib.util
import unittest

from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.gene_validation import validate_gene_design
from virturoid.services.morphology_composer import compose_robot

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class GeneValidationTests(unittest.TestCase):
    def test_report_shape_and_clean_quadruped(self):
        r = validate_gene_design(compose_robot("a quadruped walking robot"))
        for k in ("ok", "confidence", "risk_flags", "checks", "disclaimer"):
            self.assertIn(k, r)
        self.assertIsInstance(r["risk_flags"], list)
        self.assertTrue(r["checks"]["kinematic"])
        self.assertGreater(r["confidence"], 0.0)
        self.assertIn("not a", r["disclaimer"].lower())          # honest no-transfer-guarantee disclaimer

    def test_kinematic_invalid_is_fatal(self):
        # Two roots = not a valid tree -> a single fatal flag, short-circuit, ok=False.
        bad = RobotGene(id="x", species="x", robot_class="manipulator", base_mount="table",
                        end_effector_type="none",
                        segments=[GeneSegment(name="a", parent=None, is_end_effector=True),
                                  GeneSegment(name="b", parent=None)])
        r = validate_gene_design(bad)
        self.assertFalse(r["ok"])
        self.assertTrue(any(f["check"] == "kinematic" and f["severity"] == "fatal" for f in r["risk_flags"]))

    def test_under_spec_joint_flagged(self):
        # A joint demanding more torque than the largest catalog actuator is surfaced (high), not hidden.
        base = GeneSegment(name="base", parent=None)
        j = GeneSegment(name="j", parent="base", joint_type="revolute", actuator_torque_nm=500.0,
                        length_m=0.3, radius_m=0.03, mass_kg=0.3, is_end_effector=True)
        g = RobotGene(id="x", species="x", robot_class="manipulator", base_mount="table",
                      end_effector_type="none", segments=[base, j])
        r = validate_gene_design(g)
        self.assertTrue(any(f["check"] == "actuator_under_spec" for f in r["risk_flags"]))

    @unittest.skipUnless(_MUJOCO, "mujoco not installed")
    def test_quadruped_is_stable_at_rest(self):
        r = validate_gene_design(compose_robot("a quadruped walking robot"))
        self.assertIn("stability", r["checks"])          # 4 feet -> support polygon -> CoM inside -> stable
        self.assertTrue(r["checks"]["stability"])

    @unittest.skipUnless(_MUJOCO, "mujoco not installed")
    def test_mobile_base_uses_wheel_footprint_for_stability(self):
        r = validate_gene_design(compose_robot("a mobile base robot that navigates around obstacles"))
        self.assertIn("stability", r["checks"])          # 4 wheels -> support polygon -> CoM inside -> stable
        self.assertTrue(r["checks"]["stability"])
        self.assertFalse(any(f["check"] == "stability" and f["severity"] == "high" for f in r["risk_flags"]))

    @unittest.skipUnless(_MUJOCO, "mujoco not installed")
    def test_top_heavy_pole_is_flagged_unstable(self):
        # A tall thin free-base pole has a point/line support -> statically unstable rest pose.
        base = GeneSegment(name="b", parent=None, shape="capsule", length_m=0.04, radius_m=0.02, mass_kg=0.2)
        pole = GeneSegment(name="p", parent="b", shape="capsule", length_m=1.2, radius_m=0.02, mass_kg=1.0,
                           joint_type=None, is_end_effector=True)
        g = RobotGene(id="pole", species="x", robot_class="legged", base_mount="free",
                      end_effector_type="none", segments=[base, pole])
        r = validate_gene_design(g)
        self.assertFalse(r["checks"].get("stability", True))
        self.assertTrue(any(f["check"] == "stability" for f in r["risk_flags"]))


if __name__ == "__main__":
    unittest.main()
