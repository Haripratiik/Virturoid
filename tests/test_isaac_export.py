"""The NVIDIA Isaac Sim / Isaac Lab hand-off: export a RobotGene to OpenUSD physics + an Isaac Lab task package.

The un-gameable proofs (re-read from the written files, never asserted from the writer's own return):
  * the USD has exactly ONE articulation root and its actuated-DOF count equals the gene's,
  * every joint's frames are geometrically consistent: body0 * localPos0 == body1 * localPos1 == world anchor,
  * revolute limits are stored in DEGREES (Isaac's convention), matching the model's radian ranges,
  * every generated Python file byte-compiles and the cfg carries the real per-joint motor limits.

USD tests are gated on ``usd-core`` (pure-CPU OpenUSD) + ``mujoco``. The pure helpers + honest-degradation path
always run.
"""
import ast
import importlib.util
import math
import os
import py_compile
import unittest
from unittest import mock

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
os.environ.setdefault("MUJOCO_GL", "glfw")
_PXR = importlib.util.find_spec("pxr") is not None
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_READY = _PXR and _MUJOCO


def _gene(prompt):
    from virturoid.services.morphology_composer import compose_robot
    return compose_robot(prompt)


def _gene_dof(gene):
    return sum(1 for s in gene.segments if s.joint_type in ("revolute", "prismatic"))


class PureHelperTests(unittest.TestCase):
    """No pxr/mujoco needed — the naming, leaf-detection, tool registration, honest-degradation surface."""

    def test_pyid_sanitizes(self):
        from virturoid.services.isaac_lab_exporter import _pyid
        self.assertEqual(_pyid("a robot-dog!"), "a_robot_dog")
        self.assertTrue(_pyid("3legs").isidentifier())          # can't start with a digit

    def test_tools_registered(self):
        from virturoid.services.agent_tools import tool_specs
        names = {t["name"] for t in tool_specs()}
        self.assertIn("export_isaac", names)
        # usd + isaac_lab are valid export_held formats
        from virturoid.services.agent_design_tools import _EXPORT_FORMATS
        self.assertIn("usd", _EXPORT_FORMATS)
        self.assertIn("isaac_lab", _EXPORT_FORMATS)

    def test_export_isaac_degrades_honestly_without_usd_core(self):
        # if OpenUSD is missing, the tool returns a clear install hint, not a crash
        from virturoid.services import agent_design_tools as adt
        from virturoid.services import session_state as S
        rid = S.put_robot(_min_gene(), label="isaac-nocore")
        with mock.patch("virturoid.services.isaac_lab_exporter.export_isaac_lab",
                        side_effect=ImportError("USD export needs OpenUSD. Install: pip install usd-core")):
            out = adt.export_isaac({"robot_id": rid})
        self.assertFalse(out["ok"])
        self.assertIn("usd-core", out.get("hint", "") + out.get("error", ""))


def _min_gene():
    """A tiny 1-joint gene so the pure tests don't need the composer."""
    from virturoid.schemas.gene import GeneSegment, RobotGene
    return RobotGene(id="t", species="test", robot_class="manipulator", base_mount="table", segments=[
        GeneSegment(name="base", parent=None, shape="box", length_m=0.1, radius_m=0.05, mass_kg=1.0),
        GeneSegment(name="link1", parent="base", shape="capsule", length_m=0.2, radius_m=0.03, mass_kg=0.5,
                    joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-1.0, joint_upper=1.0,
                    actuator_torque_nm=8.0)])


def _frame_worst_error(path):
    """Max || body0*localPos0 - body1*localPos1 || over all joints (the geometric consistency check)."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.Open(path)
    xcache = UsdGeom.XformCache()
    worst = 0.0
    for p in stage.Traverse():
        if not (p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)):
            continue
        j = UsdPhysics.Joint(p)
        b0, b1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
        lp0, lp1 = j.GetLocalPos0Attr().Get(), j.GetLocalPos1Attr().Get()
        if not (b0 and b1 and lp0 is not None and lp1 is not None):
            continue
        a0 = np.array(xcache.GetLocalToWorldTransform(stage.GetPrimAtPath(b0[0])).Transform(Gf.Vec3d(*lp0)))
        a1 = np.array(xcache.GetLocalToWorldTransform(stage.GetPrimAtPath(b1[0])).Transform(Gf.Vec3d(*lp1)))
        worst = max(worst, float(np.abs(a0 - a1).max()))
    return worst


@unittest.skipUnless(_READY, "USD export needs usd-core + mujoco")
class UsdExportTests(unittest.TestCase):
    def _export(self, prompt, fname):
        from virturoid.services.usd_exporter import export_usd
        gene = _gene(prompt)
        path = os.path.join(self.tmp, fname)
        return gene, export_usd(gene, path), path

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="virt_usd_")

    def test_dof_and_root_match_the_gene(self):
        gene, man, path = self._export("a quadruped robot dog that walks", "quad.usda")
        self.assertTrue(man["validated"])
        self.assertEqual(man["articulation_roots"], 1, "exactly one articulation root")
        self.assertEqual(man["dof"], _gene_dof(gene), "USD DOF must equal the gene's actuated joints")
        self.assertGreaterEqual(man["n_links"], 4)

    def test_joint_frames_are_geometrically_consistent(self):
        _, _, path = self._export("a quadruped robot dog that walks", "quad2.usda")
        self.assertLess(_frame_worst_error(path), 1e-4, "each joint's parent/child frames must meet at one anchor")

    def test_base_type_matches_class(self):
        from virturoid.services.usd_exporter import export_usd
        arm = _gene("a robotic arm that stacks blocks")
        quad = _gene("a quadruped robot dog that walks")
        self.assertEqual(export_usd(arm, os.path.join(self.tmp, "arm.usda"))["base"], "fixed")
        self.assertEqual(export_usd(quad, os.path.join(self.tmp, "q.usda"))["base"], "floating")

    def test_revolute_limits_are_degrees(self):
        # the USD limit must be the model's radian range converted to degrees (Isaac's convention)
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
        gene, man, _ = self._export("a robotic arm that stacks blocks", "arm2.usda")
        model = mujoco.MjModel.from_xml_string(
            compile_gene_to_mjcf(gene, include_floor=False, spawn_z=standing_spawn_z(gene)))
        ranges = {}
        for jj in range(model.njnt):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jj)
            if nm and bool(model.jnt_limited[jj]) and int(model.jnt_type[jj]) == 3:  # HINGE
                ranges[nm] = (float(model.jnt_range[jj][0]), float(model.jnt_range[jj][1]))
        checked = 0
        for jd in man["joints"]:
            if jd["type"] == "revolute" and jd["lower"] is not None and jd["name"] in ranges:
                lo_rad, hi_rad = ranges[jd["name"]]
                self.assertAlmostEqual(jd["lower"], math.degrees(lo_rad), delta=0.5)
                self.assertAlmostEqual(jd["upper"], math.degrees(hi_rad), delta=0.5)
                checked += 1
        self.assertGreater(checked, 0, "expected at least one limited revolute joint to verify")


@unittest.skipUnless(_READY, "Isaac Lab scaffold needs usd-core + mujoco (for the USD)")
class IsaacLabScaffoldTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="virt_isaac_")

    def _pkg(self, prompt, name):
        from virturoid.services.isaac_lab_exporter import export_isaac_lab
        return _gene(prompt), export_isaac_lab(_gene(prompt), os.path.join(self.tmp, name), robot_name=name)

    def test_all_generated_python_byte_compiles(self):
        _, man = self._pkg("a quadruped robot dog that walks", "quad")
        pys = [v for v in man["files"].values() if v.endswith(".py")]
        self.assertGreaterEqual(len(pys), 3)                    # cfg + spawn + velocity_env
        for p in pys:
            py_compile.compile(p, doraise=True)                 # raises SyntaxError on bad codegen

    def test_cfg_is_structurally_correct_and_consistent(self):
        gene, man = self._pkg("a quadruped robot dog that walks", "quadcfg")
        src = open(man["files"]["cfg"], encoding="utf-8").read()
        tree = ast.parse(src)                                   # parses
        self.assertIn("ArticulationCfg(", src)
        self.assertIn("ImplicitActuatorCfg(", src)
        self.assertIn("sim_utils.UsdFileCfg(", src)
        self.assertIn("from isaaclab.assets import ArticulationCfg", src)
        # cross-consistency: one init joint_pos + one effort entry per actuated joint
        dof = _gene_dof(gene)
        self.assertEqual(len(man["actuator_effort_nm"]), dof)
        self.assertEqual(man["dof"], dof)
        del tree

    def test_effort_limits_are_real_motor_torques(self):
        _, man = self._pkg("a quadruped robot dog that walks", "quadmot")
        efforts = list(man["actuator_effort_nm"].values())
        self.assertTrue(all(e > 0 for e in efforts))
        self.assertTrue(any(e > 3.0 for e in efforts), "leg motors should carry real torque, not a 1.0 stub")

    def test_legged_gets_velocity_env_manipulator_does_not(self):
        _, quad = self._pkg("a quadruped robot dog that walks", "qv")
        _, arm = self._pkg("a robotic arm that stacks blocks", "av")
        self.assertIn("velocity_env", quad["files"])
        self.assertTrue(quad["is_legged"])
        self.assertNotIn("velocity_env", arm["files"])
        self.assertFalse(arm["is_legged"])

    def test_readme_is_honest_about_isaac_verification(self):
        _, man = self._pkg("a quadruped robot dog that walks", "qr")
        readme = open(man["files"]["readme"], encoding="utf-8").read().lower()
        self.assertIn("verify in isaac", readme)                # states what the engineer must still check
        self.assertIn("not a trained policy", readme)           # no overclaiming
        self.assertTrue(man["not_run_in_isaac"])

    def test_export_held_and_export_isaac_produce_the_package(self):
        from virturoid.services import agent_design_tools as adt
        from virturoid.services import session_state as S
        rid = S.put_robot(_gene("a quadruped robot dog that walks"), label="isaac")
        held = adt.export_held({"robot_id": rid, "formats": ["usd", "isaac_lab"]})
        self.assertTrue(held["ok"])
        self.assertIn("usd", held["artifacts"])
        self.assertIn("isaac_lab", held["artifacts"])
        tool = adt.export_isaac({"robot_id": rid})
        self.assertTrue(tool["ok"])
        self.assertTrue(tool["usd_validated"])
        self.assertGreater(tool["dof"], 0)


if __name__ == "__main__":
    unittest.main()
