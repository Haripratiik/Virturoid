import math
import unittest

from virturoid.fixtures.gene_library import (
    humanoid_upper_body_gene,
    tabletop_arm_gene,
)
from virturoid.schemas.gene import GeneSegment, RobotGene
from virturoid.services.structural_validation import (
    MATERIALS_PA,
    validate_structure,
)


def _underbuilt_gene() -> RobotGene:
    """A deliberately infeasible arm: pencil-thin links carrying heavy distal mass.

    2 mm half-width links with kilograms of downstream mass on a long reach — no plausible
    metal survives this in bending. (Valid kinematic tree so the structural check actually
    runs.)
    """
    return RobotGene(
        id="underbuilt", species="x.y.weak", robot_class="manipulator",
        base_mount="table", end_effector_type="gripper",
        segments=[
            GeneSegment("base", parent=None, shape="box", length_m=0.07, radius_m=0.06, mass_kg=0.45),
            GeneSegment("upper", parent="base", shape="cylinder", length_m=0.4, radius_m=0.002, mass_kg=2.0,
                        joint_type="revolute", actuator_torque_nm=12),
            GeneSegment("forearm", parent="upper", shape="cylinder", length_m=0.4, radius_m=0.002, mass_kg=2.0,
                        joint_type="revolute", actuator_torque_nm=12),
            GeneSegment("gripper", parent="forearm", shape="box", length_m=0.05, radius_m=0.002, mass_kg=1.0,
                        joint_type="fixed", is_end_effector=True),
        ],
    )


class StructuralValidationTests(unittest.TestCase):
    def test_seed_arm_passes_with_a_reasonable_material(self):
        report = validate_structure(tabletop_arm_gene(), material="aluminum")
        self.assertTrue(report["ok"], f"seed arm should be buildable in aluminum: {report['issues']}")
        self.assertEqual([], report["issues"])
        self.assertEqual("aluminum", report["material"])
        # Every link clears the default safety-factor floor.
        self.assertTrue(all(link["feasible"] for link in report["links"]))

    def test_humanoid_seed_also_buildable(self):
        report = validate_structure(humanoid_upper_body_gene(), material="aluminum")
        self.assertTrue(report["ok"], f"humanoid should be buildable in aluminum: {report['issues']}")

    def test_underbuilt_gene_fails_and_names_weak_links(self):
        report = validate_structure(_underbuilt_gene(), material="aluminum")
        self.assertFalse(report["ok"])
        self.assertTrue(report["issues"], "expected under-strength issues")
        # The pencil-thin loaded links are flagged infeasible by name.
        infeasible = {link["name"] for link in report["links"] if not link["feasible"]}
        self.assertIn("upper", infeasible)
        self.assertIn("forearm", infeasible)
        # The issue text identifies the weak links and offers a remedy.
        joined = " ".join(report["issues"])
        self.assertIn("upper", joined)
        self.assertIn("forearm", joined)
        self.assertIn("radius", joined.lower())
        self.assertLess(report["min_safety_factor"], 2.0)

    def test_weak_material_has_lower_safety_factor_than_strong_one(self):
        gene = tabletop_arm_gene()
        weak = validate_structure(gene, material="pla")
        strong = validate_structure(gene, material="carbon_fiber")
        self.assertLess(weak["min_safety_factor"], strong["min_safety_factor"])
        # Same geometry/loads -> SF scales linearly with yield strength.
        ratio = MATERIALS_PA["carbon_fiber"] / MATERIALS_PA["pla"]
        self.assertAlmostEqual(
            strong["min_safety_factor"] / weak["min_safety_factor"], ratio, places=3
        )

    def test_report_fields_are_present_and_numeric(self):
        report = validate_structure(tabletop_arm_gene())
        self.assertIn("ok", report)
        self.assertIn("material", report)
        self.assertIsInstance(report["ok"], bool)
        self.assertIsInstance(report["min_safety_factor"], float)
        self.assertGreater(report["min_safety_factor"], 0.0)
        self.assertEqual(len(report["links"]), len(tabletop_arm_gene().segments))
        for link in report["links"]:
            for key in ("name", "stress_pa", "safety_factor", "feasible"):
                self.assertIn(key, link)
            self.assertIsInstance(link["name"], str)
            self.assertIsInstance(link["stress_pa"], float)
            self.assertIsInstance(link["safety_factor"], float)
            self.assertIsInstance(link["feasible"], bool)
            self.assertFalse(math.isnan(link["stress_pa"]))

    def test_payload_lowers_safety_factor(self):
        gene = tabletop_arm_gene()
        no_load = validate_structure(gene, material="pla", payload_kg=0.0)
        heavy = validate_structure(gene, material="pla", payload_kg=5.0)
        self.assertLessEqual(heavy["min_safety_factor"], no_load["min_safety_factor"])

    def test_unknown_material_falls_back_to_aluminum(self):
        report = validate_structure(tabletop_arm_gene(), material="unobtainium")
        self.assertEqual("aluminum", report["material"])
        self.assertTrue(any("unknown material" in i for i in report["issues"]))


if __name__ == "__main__":
    unittest.main()
