"""Capability-driven amend: set_payload puts a target load ON the robot and re-specs what may be re-spec'd.

The honest answer to "if I ask the robot to lift heavier, does it amend the robot?" — YES: the payload becomes a
real link the sim carries, and the joints whose limits are ours get stronger motors so the BOM/mass rise.

What this file did NOT cover, and now does. ``set_payload`` used to scale joint torques, stamp a rating, and add
NOTHING — measured through ``call_tool`` on a real Menagerie Go2, ``payload_kg: 25`` returned
``added_mass_kg: 0``. Every assertion here passed throughout, because they all asked whether torque and total
mass went UP and both did: from re-deriving the customer's own 15 links (+30.654 kg), which is the opposite of
the request. "Mass rose" is not "the payload is on the robot", and only the second one is what was asked for.

Offline (AGENTS.md); MuJoCo not required (pure gene mutation + grounding).
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
        # a payload whose torque genuinely exceeds the strongest real motor must be REPORTED, not silently
        # maxed-out-and-called-fine. Use a high-torque-joint arm (an industrial-class joint) so a big payload
        # actually clears the 520 Nm catalog ceiling.
        g = _arm()
        for s in g.segments:
            if s.joint_type in ("revolute", "prismatic"):
                s.torque_req_nm = 240.0                          # a heavy industrial joint (base requirement)
                s.actuator_torque_nm = 240.0
        _, diff = set_payload(g, payload_kg=40.0)                # 240 x large load_factor -> well over 520 Nm
        self.assertIn("undersized_joints", diff)
        self.assertIn("warning", diff)
        self.assertGreater(len(diff["undersized_joints"]), 0)
        # ...and a modest payload a small arm CAN carry must not be falsely flagged
        _, ok = set_payload(_arm(), payload_kg=3.0)
        self.assertNotIn("warning", ok)

    def test_is_registered(self):
        from virturoid.services.edit_operators import OPERATORS, op_specs
        self.assertIn("set_payload", OPERATORS)
        self.assertTrue(any(s["op"] == "set_payload" for s in op_specs()))

    # ----------------------------------------------------------------- the payload is ON the robot
    def test_the_payload_is_added_as_a_real_link_with_exactly_that_mass(self):
        """`added_mass_kg: 0` for a 10 kg request was the whole defect. The load has to be carryable."""
        g = _arm()
        g2, diff = set_payload(g, payload_kg=10.0)
        link = g2.segment(diff["payload_link"])
        self.assertIsNotNone(link, "no payload link on the robot after a payload request")
        self.assertAlmostEqual(float(link.mass_kg), 10.0, places=4)
        self.assertAlmostEqual(diff["mass"]["added_mass_kg"], 10.0, places=2)
        self.assertEqual(link.joint_type, "fixed")               # a carried load, not a new degree of freedom
        self.assertEqual(link.parent, diff["carried_by"])
        # ...and the sim carries it: the body's own total goes up by the load, on top of any motor mass.
        self.assertGreaterEqual(_mass(g2) - _mass(g), 10.0 - 1e-3)

    def test_the_stated_mass_survives_grounding(self):
        """Grounding derives masses from geometry. 25 kg asked for must not become (box volume x density)."""
        g2, diff = set_payload(_arm(), payload_kg=25.0)
        self.assertAlmostEqual(float(g2.segment(diff["payload_link"]).mass_kg), 25.0, places=4)

    def test_an_arm_carries_the_load_at_its_tool_not_on_its_base(self):
        """Where the mass sits is inertia, not bookkeeping: an arm's payload hangs off the end effector."""
        g = _arm()
        g2, diff = set_payload(g, payload_kg=3.0)
        tip = g.end_effector()
        if tip is not None:
            self.assertEqual(diff["carried_by"], tip.name)
        self.assertNotEqual(diff["carried_by"], (g.root().name if g.root() else ""))

    def test_a_second_payload_does_not_collide_with_the_first(self):
        """Two amends must not silently overwrite each other's load."""
        g2, d1 = set_payload(_arm(), payload_kg=2.0)
        g3, d2 = set_payload(g2, payload_kg=4.0)
        self.assertNotEqual(d1["payload_link"], d2["payload_link"])
        self.assertEqual(len([s for s in g3.segments if s.name.startswith("payload")]), 2)

    # ----------------------------------------------------------------- the ledger is a measurement
    def test_source_masses_preserved_is_the_measurement_not_a_constant(self):
        """It read `true` beside `n_existing_links_remassed: 15`. The two now cannot disagree."""
        from virturoid.services.edit_operators import _mass_ledger
        g = _arm()
        before = {s.name: float(s.mass_kg or 0.0) for s in g.segments}
        g.segments[0].mass_kg = float(g.segments[0].mass_kg) + 1.0      # one link re-massed behind the flag
        led = _mass_ledger(before, g, added=set(), preserved=True)
        self.assertEqual(led["n_existing_links_remassed"], 1)
        self.assertFalse(led["source_masses_preserved"])
        self.assertEqual(led["mass_authority"], "source_model")         # ...and the other fact is still stated

    def test_a_dropped_link_cannot_read_as_preserved(self):
        """A body-replacing op keeps no per-link mass, so 'nothing changed' must not be the answer."""
        from virturoid.services.edit_operators import _mass_ledger
        g = _arm()
        before = {s.name: float(s.mass_kg or 0.0) for s in g.segments}
        before["a_link_this_edit_deleted"] = 4.0
        led = _mass_ledger(before, g, added=set(), preserved=True)
        self.assertEqual(led["n_existing_links_dropped"], 1)
        self.assertFalse(led["source_masses_preserved"])

    def test_upsize_actuators_false_touches_no_joint(self):
        g = _arm()
        t0 = _torques(g)
        g2, diff = set_payload(g, payload_kg=10.0, upsize_actuators=False)
        self.assertEqual(diff["n_joints_upsized"], 0)
        self.assertEqual(_torques(g2)[:len(t0)], t0)
        self.assertIsNotNone(g2.segment(diff["payload_link"]))          # the payload still lands

    def test_the_payload_is_not_billed_as_raw_material(self):
        """Cargo is not stock. ``bom_builder`` bills a material line from ``structural_link_masses``, so a load
        left out of the embodiment ledger would appear on the parts list as 25 kg of metal to buy."""
        from virturoid.services.grounded_physics import structural_link_masses
        g2, diff = set_payload(_arm(), payload_kg=25.0)
        struct = structural_link_masses(g2)
        self.assertAlmostEqual(struct[diff["payload_link"]], 0.0, places=3)
        self.assertEqual((g2.metadata["embodied_mass"]["payload_kg"]), {diff["payload_link"]: 25.0})

    def test_a_later_edit_does_not_re_derive_the_cargo(self):
        """The load survives the NEXT amend too: scaling the robot must not re-weigh what it is carrying."""
        from virturoid.services.edit_operators import scale_robot
        g2, diff = set_payload(_arm(), payload_kg=25.0)
        g3, _ = scale_robot(g2, factor=1.3)
        self.assertAlmostEqual(float(g3.segment(diff["payload_link"]).mass_kg), 25.0, places=3)

    def test_a_bad_upsize_mode_teaches(self):
        with self.assertRaises(EditError) as ctx:
            set_payload(_arm(), payload_kg=5.0, upsize_actuators="maybe")
        self.assertIn("upsize_actuators", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
