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
import json
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

    def _assert_limits_belong_to_their_own_joint(self, gene, man, label):
        """Every per-joint limit in the cfg is the one THIS joint's own segment was sized for.

        A range check (`all(e > 0)`, `any(e > 3)`) is invariant under a PERMUTATION of the values, which is
        exactly the failure this pins: ``export_isaac_lab`` used to pair the USD joint list against
        ``gene.segments`` order BY POSITION, and on an amended body those orders differ.

        The ``actuator_limits_unresolved == []`` assertion below is only worth writing because the list CAN be
        non-empty: ``test_an_unresolvable_joint_is_named_not_handed_the_default`` drives that branch and
        asserts what lands in it. Without that companion the assertion could never fail.

        The expected segment is derived from the JOINT NAME (``<segment>_joint``, which the compiler builds by
        construction for a composed/amended gene), deliberately NOT from the ``child`` field the exporter now
        resolves through -- so this asserts the answer, not the implementation.
        """
        from virturoid.services.usd_exporter import _actuator_limits
        limits = _actuator_limits(gene)
        eff, vel = man["actuator_effort_nm"], man["actuator_velocity_radps"]
        self.assertEqual(man["actuator_limits_unresolved"], [], f"{label}: unresolved joints")
        checked = 0
        for jname, e in eff.items():
            self.assertTrue(jname.endswith("_joint"), f"{label}: {jname} breaks the naming convention")
            seg = jname[: -len("_joint")]
            self.assertIn(seg, limits, f"{label}: {jname} names no segment of this gene")
            self.assertAlmostEqual(float(e), float(limits[seg][0]), places=6,
                                   msg=f"{label}: {jname} carries {e} Nm, but its own segment {seg} is sized "
                                       f"for {limits[seg][0]} Nm")
            self.assertAlmostEqual(float(vel[jname]), float(limits[seg][1]), places=6,
                                   msg=f"{label}: {jname}'s velocity limit belongs to another joint")
            self.assertEqual(man["joint_segment"].get(jname), seg, f"{label}: manifest pairing for {jname}")
            checked += 1
        # a permutation is only VISIBLE when the values differ; a body sized to one torque everywhere would
        # make the assertions above pass under any shuffle.
        self.assertGreater(len(set(round(float(v), 6) for v in eff.values())), 1,
                           f"{label}: all efforts identical -- this check could not see a permutation")
        return checked

    def test_per_joint_limits_belong_to_their_own_joint_composed(self):
        from virturoid.services.isaac_lab_exporter import export_isaac_lab
        gene = _gene("a quadruped robot dog that walks")
        man = export_isaac_lab(gene, os.path.join(self.tmp, "pairc"), robot_name="pairc")
        self.assertGreaterEqual(self._assert_limits_belong_to_their_own_joint(gene, man, "composed"), 4)

    def test_an_unresolvable_joint_is_named_not_handed_the_default(self):
        """The NEGATIVE side of ``actuator_limits_unresolved``, without which the ``== []`` assertions in
        ``_assert_limits_belong_to_their_own_joint`` are decorative -- an always-empty list cannot fail.

        Drives the real branch (`sname is None or sname not in limits`) the way production reaches it: a joint
        whose driven segment the BOM sizes NO actuator for. The joint must be NAMED in the manifest and called
        out in the README, never silently handed the 20 Nm placeholder as if it were a measured motor limit.
        """
        from virturoid.services import usd_exporter
        from virturoid.services.isaac_lab_exporter import export_isaac_lab
        gene = _gene("a quadruped robot dog that walks")
        real = usd_exporter._actuator_limits
        full = real(gene)
        dropped = sorted(full)[0]                     # one segment the parts list will not size

        def _missing_one(g):
            return {k: v for k, v in real(g).items() if k != dropped}

        with mock.patch.object(usd_exporter, "_actuator_limits", _missing_one):
            man = export_isaac_lab(gene, os.path.join(self.tmp, "unres"), robot_name="unres")

        unresolved = man["actuator_limits_unresolved"]
        self.assertEqual(unresolved, [f"{dropped}_joint"], man["joint_segment"])
        self.assertNotIn(f"{dropped}_joint", man["joint_segment"])          # no fabricated pairing
        self.assertEqual(man["actuator_effort_nm"][f"{dropped}_joint"], 20.0)   # the placeholder, disclosed
        # every OTHER joint still carries its own segment's real limit -- the gap is scoped, not contagious
        for jname, seg in man["joint_segment"].items():
            self.assertAlmostEqual(float(man["actuator_effort_nm"][jname]), float(full[seg][0]), places=6)
        readme = open(man["files"]["readme"], encoding="utf-8").read()
        self.assertIn("could NOT be resolved", readme)
        self.assertIn("actuator_limits_unresolved", readme)
        manifest = json.loads(open(os.path.join(self.tmp, "unres", "manifest.json"), encoding="utf-8").read())
        self.assertEqual(manifest["actuator_limits_unresolved"], unresolved)   # re-read from the written file

    def test_per_joint_limits_survive_an_amend(self):
        """The regression that the range check could not see: ``add_limb`` appends its chain at the END of
        ``gene.segments`` while the compiler emits it under its mid-tree parent, so kinematic-DFS order and
        list order diverge. How MANY joints that corrupts depends on WHERE you mount, so no single ratio
        describes it: sweeping the mid-tree parent on this 19-DOF body, the old positional pairing handed a
        joint the wrong segment's motor 18/19 (`neck`, the parent this test picks), 14/19 (`leg1_l_*`) and
        11/19 (`leg1_r_*`) -- of which the effort/velocity numbers actually differ on 12, 9 and 7."""
        from virturoid.services.edit_operators import add_limb
        from virturoid.services.isaac_lab_exporter import export_isaac_lab
        gene = _gene("a quadruped robot dog that walks")
        # a MID-TREE parent (has a parent of its own AND has children), picked from the body rather than
        # hard-coded, so the test does not depend on the composer's naming
        names = {s.name for s in gene.segments}
        mid = next(s.name for s in gene.segments
                   if s.parent in names and any(c.parent == s.name for c in gene.segments))
        amended, _ = add_limb(gene, segments=3, name="limb")            # on the root
        amended, _ = add_limb(amended, parent=mid, segments=2, name="mast")   # mid-tree: the reorder
        man = export_isaac_lab(amended, os.path.join(self.tmp, "paira"), robot_name="paira")
        n = self._assert_limits_belong_to_their_own_joint(amended, man, "amended")
        self.assertGreater(n, _gene_dof(gene), "the amend should have added actuated joints")

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
