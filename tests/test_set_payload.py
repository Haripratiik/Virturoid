"""Capability-driven amend: set_payload upsizes actuators (+ girth) for a target load — "make it lift heavier".

The honest answer to "if I ask the robot to lift heavier, does it amend the robot?" — YES, the joints get stronger
motors and the BOM/mass rise. Offline (AGENTS.md); MuJoCo not required (pure gene mutation + grounding).
"""
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.services.edit_operators import EditError, set_payload  # noqa: E402


def _arm():
    from virturoid.fixtures.gene_library import tabletop_arm_gene
    from virturoid.services.grounded_physics import ground_gene
    g = tabletop_arm_gene()
    ground_gene(g)                                             # give it real grounded actuators/mass first
    return g


def _torques(g):
    return [float(s.actuator_torque_nm or 0.0) for s in g.segments if s.joint_type in ("revolute", "prismatic")]


def _mass(g):
    return sum(float(s.mass_kg or 0.0) for s in g.segments)


class SetPayloadTests(unittest.TestCase):
    def test_upsizes_actuators_and_mass_for_payload(self):
        g = _arm()
        t0, m0 = sum(_torques(g)), _mass(g)
        g2, diff = set_payload(g, payload_kg=10.0)
        self.assertEqual(diff["op"], "set_payload")
        self.assertGreater(diff["load_factor"], 1.0)
        self.assertGreater(sum(_torques(g2)), t0)             # joints got stronger
        self.assertGreater(_mass(g2), m0)                     # bigger motors -> more mass (honest cost)
        self.assertGreater(diff["n_joints_upsized"], 0)

    def test_heavier_payload_scales_more(self):
        _, light = set_payload(_arm(), payload_kg=2.0)
        _, heavy = set_payload(_arm(), payload_kg=20.0)
        self.assertGreater(heavy["load_factor"], light["load_factor"])   # monotone in payload
        heavy_g, _ = set_payload(_arm(), payload_kg=20.0)
        light_g, _ = set_payload(_arm(), payload_kg=2.0)
        self.assertGreaterEqual(sum(_torques(heavy_g)), sum(_torques(light_g)))  # non-decreasing (may saturate)

    def test_out_of_range_payload_teaches(self):
        with self.assertRaises(EditError) as ctx:
            set_payload(_arm(), payload_kg=100.0)
        self.assertIn("out of the safe range", str(ctx.exception))

    def test_no_actuators_errors(self):
        from virturoid.schemas.gene import GeneSegment, RobotGene
        g = RobotGene(id="x", species="static.block", robot_class="static",
                      segments=[GeneSegment(name="base", parent=None, joint_type="fixed",
                                            length_m=0.2, radius_m=0.05, mass_kg=1.0)])
        with self.assertRaises(EditError):
            set_payload(g, payload_kg=5.0)

    def test_over_catalog_payload_is_flagged_not_faked(self):
        # a payload that exceeds the strongest real motor must be reported, not silently maxed-out-and-called-fine
        _, diff = set_payload(_arm(), payload_kg=50.0)
        self.assertIn("undersized_joints", diff)
        self.assertIn("warning", diff)
        self.assertGreater(len(diff["undersized_joints"]), 0)
        # ...and a payload that DOES fit must not be falsely flagged
        _, ok = set_payload(_arm(), payload_kg=3.0)
        self.assertNotIn("warning", ok)

    def test_is_registered(self):
        from virturoid.services.edit_operators import OPERATORS, op_specs
        self.assertIn("set_payload", OPERATORS)
        self.assertTrue(any(s["op"] == "set_payload" for s in op_specs()))


if __name__ == "__main__":
    unittest.main()
