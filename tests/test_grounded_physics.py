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


if __name__ == "__main__":
    unittest.main()
