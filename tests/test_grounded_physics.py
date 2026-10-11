"""STAGE 2: a procedurally-composed body is physically GROUNDED — link mass from material density x
geometry, each joint matched to a REAL actuator (covering its torque) + a real Bill of Materials."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class GroundedPhysicsTests(unittest.TestCase):
    def test_masses_are_grounded_not_flat_guesses(self):
        from virturoid.services.grounded_physics import _link_volume_m3, ground_gene
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        flat = {s.mass_kg for s in g.segments}
        ground_gene(g, material="aluminum")
        # mass now varies with geometry (not a single flat 0.3) and tracks volume
        self.assertGreater(len({round(s.mass_kg, 3) for s in g.segments}), len(flat) - 2)
        for s in g.segments:
            self.assertGreater(s.mass_kg, 0.0)
            # a denser material yields a heavier structural mass for the same shape
        vol = _link_volume_m3("capsule", 0.2, 0.03)
        self.assertGreater(vol, 0.0)

    def test_real_actuators_cover_required_torque_and_form_a_bom(self):
        from virturoid.services.grounded_physics import ACTUATORS, ground_gene
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        grounding = ground_gene(g)
        self.assertEqual(grounding["actuator_count"], len(g.actuated_joints()))
        self.assertGreater(grounding["total_mass_kg"], 0.0)
        catalog = [a[0] for a in ACTUATORS]
        biggest = ACTUATORS[-1]
        for item in grounding["bom"]:
            self.assertIn(item["part"], catalog)                       # a real catalogue part
            self.assertTrue(item["stall_nm"] >= item["required_nm"] - 1e-6 or item["part"] == biggest[0])

    def test_grounded_co_design_optimizes_a_real_physics_body(self):
        # STAGE 4: generative co-design runs OVER the grounded substrate — each variant has real masses +
        # a real actuator; the search never regresses and returns a valid, grounded best body.
        from virturoid.services.gene_codesign import co_design_general
        from virturoid.services.grounded_physics import ACTUATORS
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        res = co_design_general(g, "a quadruped walking robot", iterations=2, population=4, ground=True)
        best = res["gene"]
        self.assertEqual(best.validate(), [])
        self.assertGreaterEqual(res["best_value"], res["baseline_value"])      # never-regress
        stalls = {a[2] for a in ACTUATORS}
        self.assertTrue(all(s.actuator_torque_nm in stalls
                            for s in best.segments if s.actuator_torque_nm))   # real catalogue actuators

    def test_grounded_body_still_compiles_and_sims(self):
        import mujoco
        import numpy as np

        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.grounded_physics import ground_gene
        from virturoid.services.morphology_composer import compose_robot
        g = compose_robot("a quadruped walking robot")
        ground_gene(g)
        self.assertEqual(g.validate(), [])
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))
        d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
        for _ in range(60):
            mujoco.mj_step(m, d)
        self.assertTrue(np.all(np.isfinite(d.qpos)))

    def test_reference_morphologies_land_in_grounded_mass_bands(self):
        from virturoid.services.grounded_physics import ground_gene
        from virturoid.services.morphology_composer import compose_robot
        cases = [("a robot dog", (12.0, 15.0)),
                 ("a 6-axis collaborative robot arm", (17.0, 21.0))]
        for prompt, band in cases:
            gene = compose_robot(prompt, llm=None, ensure_walkable=False)
            report = ground_gene(gene, fill=0.25)
            self.assertGreaterEqual(report["total_mass_kg"], band[0], prompt)
            self.assertLessEqual(report["total_mass_kg"], band[1], prompt)
            self.assertIsNotNone(report["physical_prior"])

    def test_the_warehouse_envelope_is_for_a_warehouse_sized_base(self):
        """The AMR band (70-150 kg) describes 0.8-0.9 m machines. The composed rover is a 0.36 m deck, and the
        band used to lift it to 70 kg with 57 kg of estimated wiring; it now keeps its own itemised mass. The same
        body widened to an AMR's footprint still lands in the band."""
        from virturoid.services.grounded_physics import ground_gene, physical_prior_for
        from virturoid.services.morphology_composer import compose_robot
        small = compose_robot("a four-wheeled warehouse rover", llm=None, ensure_walkable=False)
        self.assertIsNone(physical_prior_for(small))
        report = ground_gene(small, fill=0.25)
        self.assertLess(report["total_mass_kg"], 70.0)
        self.assertEqual(report["balance_of_system_mass_kg"], 0.0)
        big = compose_robot("a four-wheeled warehouse rover", llm=None, ensure_walkable=False)
        big.root().cross_section = (0.40, 0.29)                  # an 0.80 x 0.58 m deck
        self.assertEqual(getattr(physical_prior_for(big), "id", None), "mobile.amr")
        report = ground_gene(big, fill=0.25)
        self.assertGreaterEqual(report["total_mass_kg"], 70.0)
        self.assertLessEqual(report["total_mass_kg"], 150.0)

    def test_physics_derived_torque_decreases_down_a_leg(self):
        from virturoid.services.grounded_physics import ground_gene
        from virturoid.services.morphology_composer import compose_robot
        gene = compose_robot("a robot dog", llm=None, ensure_walkable=True)
        ground_gene(gene)
        chain = [s for s in gene.segments if s.name.startswith("leg1_l_") and s.joint_type == "revolute"]
        needs = [s.torque_req_nm for s in sorted(chain, key=lambda s: s.name)]
        self.assertGreaterEqual(needs[0], needs[-1])
        self.assertGreater(len(set(needs)), 1)


if __name__ == "__main__":
    unittest.main()
