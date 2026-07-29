"""Real off-the-shelf actuator grounding: pick the real part by torque, render it at its datasheet size as a
visual-only housing, and emit a citable BOM — without copying any real robot shell, and without touching
physics."""

import importlib.util
import unittest
from pathlib import Path

from virturoid.services.component_geometry import (
    actuator_for_joint, actuator_housing_xml, bill_of_materials, select_actuator)
from virturoid.services.gene_compiler import compile_gene_to_mjcf
from virturoid.services.morphology_composer import compose_robot

_MUJOCO = importlib.util.find_spec("mujoco") is not None


class ComponentGeometryTests(unittest.TestCase):
    def test_select_smallest_actuator_that_fits(self):
        # The render selects from the SAME single-source-of-truth catalog as the BOM (component_catalog), so the
        # motor shown on a joint is the exact part the BOM cites; part names are the catalog's (vendor is a
        # separate field).
        a = select_actuator(3.0)
        self.assertGreaterEqual(a["stall_nm"], 3.0)
        self.assertEqual(a["part"], "Dynamixel XM430-W350-T")             # smallest covering 3 Nm
        big = select_actuator(999.0)
        self.assertEqual(big["part"], "Harmonic Drive FHA-40C-160")       # nothing covers it -> the largest (520 Nm)

    def test_actuator_only_for_revolute_joints(self):
        g = compose_robot("a quadruped walking robot")
        rev = next(s for s in g.segments if s.joint_type == "revolute")
        weld = next(s for s in g.segments if s.joint_type not in ("revolute", "prismatic"))
        self.assertIsNotNone(actuator_for_joint(rev))
        self.assertIsNone(actuator_for_joint(weld))

    def test_housing_is_visual_only_and_sized_to_envelope(self):
        g = compose_robot("a quadruped walking robot")
        rev = next(s for s in g.segments if s.joint_type == "revolute")
        xml = actuator_housing_xml(rev, "")
        self.assertIn('mass="0"', xml)
        self.assertIn('contype="0"', xml)
        self.assertIn('conaffinity="0"', xml)
        self.assertTrue(xml.strip().startswith("<geom"))

    def test_bom_cites_real_parts(self):
        g = compose_robot("a quadruped walking robot")
        bom = bill_of_materials(g)
        self.assertEqual(len(bom), len(g.actuated_joints()))
        for row in bom:
            self.assertTrue(row["part"])
            self.assertTrue(row["supplier"])                              # a real, named supplier (vendor)
            self.assertGreaterEqual(row["stall_nm"], 0.0)

    def test_grounded_bom_render_and_cad_resolve_the_same_motor(self):
        from virturoid.services.grounded_physics import ground_gene
        g = compose_robot("a 6-axis robot arm with a gripper", llm=None)
        grounded = ground_gene(g)
        by_role = {row["role"]: row["part"] for row in grounded["bom"]}
        for segment in g.segments:
            if segment.joint_type == "revolute":
                self.assertEqual(by_role[segment.name], actuator_for_joint(segment)["part"])

    def test_humanoid_hips_are_now_grounded_not_under_spec(self):
        # The catalog must cover humanoid-class joint torques (a hip ~42 Nm). Before adding the high-torque
        # geared actuator, the largest part topped out ~34 Nm and the hip was honestly under-spec.
        from virturoid.services.humanoid_anatomy import build_anthropometric_humanoid
        bom = bill_of_materials(build_anthropometric_humanoid())
        self.assertTrue(bom)
        self.assertFalse([r for r in bom if r["under_spec"]], "humanoid joints should be covered by the catalog")

    def test_under_spec_joint_is_flagged(self):
        # An absurd torque beyond every catalog part is honestly flagged, not silently shipped.
        from virturoid.schemas.gene import GeneSegment, RobotGene
        seg = GeneSegment(name="j", parent="b", joint_type="revolute", actuator_torque_nm=500.0,
                          is_end_effector=True)
        base = GeneSegment(name="b", parent=None)
        g = RobotGene(id="x", species="x", robot_class="manipulator", segments=[base, seg],
                      base_mount="table", end_effector_type="none")
        row = bill_of_materials(g)[0]
        self.assertTrue(row["under_spec"])

    def test_bright_line_no_part_catalog_import(self):
        # Components are servos/fasteners, never a real robot's body shell — this module must not IMPORT the
        # real-mesh kit-bash catalog. Enforced structurally so the boundary can't silently erode. (The
        # docstring may *name* part_catalog to explain the rule; what's forbidden is importing it.)
        for line in Path("src/virturoid/services/component_geometry.py").read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith(("import ", "from ")):
                self.assertNotIn("part_catalog", s)

    @unittest.skipUnless(_MUJOCO, "mujoco not installed")
    def test_actuators_are_physics_byte_identical(self):
        import mujoco

        g = compose_robot("a quadruped walking robot")
        base = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, show_actuators=False))
        act = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, show_actuators=True))
        ncol = lambda m: sum(1 for i in range(m.ngeom) if m.geom_contype[i] != 0)
        self.assertEqual(ncol(base), ncol(act))            # same collision geoms -> identical contacts

        def act_geoms(m):
            return [i for i in range(m.ngeom)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").endswith("_act")]
        self.assertTrue(act_geoms(act))                    # real actuator housings present in the shown model
        self.assertFalse(act_geoms(base))                  # and absent when not shown
        for i in act_geoms(act):                           # every housing is visual-only (no collisions)
            self.assertEqual(int(act.geom_contype[i]), 0)
        # stepping identically: the visual housings carry no mass, so the dynamics match byte-for-byte
        db, da = mujoco.MjData(base), mujoco.MjData(act)
        for _ in range(50):
            mujoco.mj_step(base, db)
            mujoco.mj_step(act, da)
        self.assertEqual(base.nq, act.nq)
        self.assertLess(float(abs(db.qpos - da.qpos).max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
