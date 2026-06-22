"""Robot import (roadmap N5.1): URDF/MJCF -> RobotGene + honest warnings.

Validated by ROUND-TRIP: compile a known gene to MJCF, import it back, and check the structure is
recovered (segments, the actuated joints, a single end-effector) — no external robot files needed.
Plus a missing-mass case to confirm we WARN rather than silently fix (startup plan §28.2)."""

import importlib.util
import unittest

_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "MuJoCo not installed.")
class RobotImportTests(unittest.TestCase):
    def _import_arm(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.robot_import import import_robot

        gene = tabletop_arm_gene()
        xml = compile_gene_to_mjcf(gene, include_floor=False)   # no floor body to keep the body set clean
        return gene, import_robot(xml, robot_id="ra", species="manipulator.test.imported")

    def test_roundtrip_recovers_structure(self):
        gene, out = self._import_arm()
        imp = out["gene"]
        # one gene segment per MJCF body -> same count
        self.assertEqual(len(imp.segments), len(gene.segments))
        # the imported gene is a valid kinematic tree
        self.assertTrue(out["valid"], out["warnings"])
        self.assertEqual(out["gene"].validate(), [])
        # exactly one end-effector, and the actuated revolute joints round-trip
        self.assertIsNotNone(imp.end_effector())
        n_rev_in = sum(1 for s in gene.segments if s.joint_type == "revolute")
        n_rev_out = sum(1 for s in imp.segments if s.joint_type == "revolute")
        self.assertEqual(n_rev_out, n_rev_in)
        self.assertEqual(out["robot_class"], "manipulator")

    def test_actuator_torque_is_recovered(self):
        _, out = self._import_arm()
        # the compiler emits motor forcerange; import should pull a torque onto actuated joints
        torques = [s.actuator_torque_nm for s in out["gene"].segments if s.joint_type == "revolute"]
        self.assertTrue(any(t and t > 0 for t in torques))

    def test_backend_support_reported(self):
        _, out = self._import_arm()
        self.assertEqual(out["backend_support"]["mujoco"]["status"], "supported")
        self.assertIn("isaac", out["backend_support"])

    def test_urdf_fixed_base_imports_to_a_valid_welded_root(self):
        """§24 'bring your own robot': a normal URDF has a fixed base_link + an actuated first joint.
        MuJoCo merges that base into the world, leaving an actuated root — which a RobotGene can't use.
        The importer must synthesize a welded base so the gene is VALID and flows into the pipeline."""
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        from virturoid.services.robot_import import import_robot

        urdf = """<?xml version="1.0"?>
        <robot name="arm2">
          <link name="base_link"><inertial><mass value="1.0"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision><geometry><box size="0.1 0.1 0.1"/></geometry></collision></link>
          <link name="upper"><inertial><mass value="0.6"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision><geometry><box size="0.04 0.04 0.3"/></geometry></collision></link>
          <link name="forearm"><inertial><mass value="0.4"/>
            <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
            <collision><geometry><box size="0.03 0.03 0.25"/></geometry></collision></link>
          <joint name="pan" type="revolute"><parent link="base_link"/><child link="upper"/>
            <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
            <limit lower="-3.14" upper="3.14" effort="20" velocity="2"/></joint>
          <joint name="elbow" type="revolute"><parent link="upper"/><child link="forearm"/>
            <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
            <limit lower="-2" upper="2" effort="15" velocity="2"/></joint>
        </robot>"""
        out = import_robot(urdf, robot_id="arm2")
        self.assertTrue(out["valid"], out["warnings"])              # VALID gene (was invalid before the fix)
        gene = out["gene"]
        roots = [s for s in gene.segments if s.parent is None]
        self.assertEqual(len(roots), 1)
        self.assertIn(roots[0].joint_type, (None, "fixed"))         # welded root
        self.assertEqual(sum(1 for s in gene.segments if s.joint_type == "revolute"), 2)  # both DOFs kept
        self.assertTrue(any("synthesized a welded base" in w for w in out["warnings"]))
        # and it compiles into the pipeline
        import mujoco
        mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))

    def test_unlimited_joint_is_warned_not_silently_fixed(self):
        # A continuous (no-range) hinge is valid MuJoCo but under-specified for a buildable design:
        # the importer must WARN (startup plan §28.2: warn, don't silently fix), and use ee_site.
        from virturoid.services.robot_import import import_robot

        xml = """
        <mujoco model="m">
          <worldbody>
            <body name="base" pos="0 0 0">
              <geom name="b" type="box" size="0.05 0.05 0.05" mass="1"/>
              <body name="link" pos="0 0 0.1">
                <joint name="j" type="hinge" axis="0 0 1"/>
                <geom name="l" type="box" size="0.02 0.02 0.1" mass="1"/>
                <site name="ee_site" pos="0 0 0.1"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
        out = import_robot(xml)
        self.assertTrue(any("limit" in w.lower() for w in out["warnings"]), out["warnings"])
        # ee_site sits on 'link' -> that body is the end-effector
        ee = out["gene"].end_effector()
        self.assertIsNotNone(ee)
        self.assertEqual(ee.name, "link")


if __name__ == "__main__":
    unittest.main()
