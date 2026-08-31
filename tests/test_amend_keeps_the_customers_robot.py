"""ONE AMEND MUST NOT UNDO THE INGEST GUARANTEE.

Measured through ``agent_tools.call_tool`` on a real Menagerie Unitree Go2, ingested with
``mass_provenance {preserved: true, delta_kg: 0.0}`` — every link verbatim from the customer's file — and then
ONE ``add_limb``:

    total 15.206 -> 32.683 kg, of which the added arm is 2.733 — so +14.7 kg was the customer's own Go2 being
    silently re-weighed (base 6.921 -> 10.306, each thigh 1.152 -> 2.050, each calf 0.241 -> 2.177, a 9x move)

while ``metadata['mass_source']`` went on reading ``source_model``, so every door downstream would keep citing
Unitree for our number. Two more from the same session:

  * the same ``add_limb`` was REFUSED, with ``before {high_or_fatal: 0, weighted_findings: 0} / after {0, 2}``
    and no other information — zero fatal findings on either side, the #1 use case blocked by two findings the
    payload would not name, on parts it would not name, with no mention that ``gate_non_regression`` exists;
  * ``attach: "top"`` put the arm 1.07 m FORWARD of the base, 0.26 m off centreline and 0.00 m UP — a
    horizontal broomstick out of the dog's nose — because the attach table was read in the PARENT LINK's frame
    and an imported trunk's local +z is horizontal.

Offline and hermetic by default; the Menagerie case runs when the real cache is present.
"""
from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_MENAGERIE = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie"))
_GO2 = _MENAGERIE / "unitree_go2"
_HAVE_GO2 = (_GO2 / "go2.xml").is_file()


def _imported_quadruped(*, root_euler=(0.0, 0.0, 0.0)):
    """A four-legged body stamped exactly as ``robot_import`` stamps a customer's model: per-link masses that
    are the MANUFACTURER'S (``mass_source='source_model'``), not derived from geometry.

    ``root_euler`` rotates the root the way a reconstructed import frame is rotated, which is the condition the
    attach bug needed: with the trunk's local +z pointing along world +x, "top" resolved parent-locally is
    forward, not up.
    """
    from virturoid.schemas.gene import GeneSegment, RobotGene
    segs = [GeneSegment(name="base", parent=None, shape="box", length_m=0.38, radius_m=0.10,
                        mass_kg=6.921, mount_euler=tuple(root_euler))]
    for side, sy in (("L", 1.0), ("R", -1.0)):
        for end, sx in (("F", 1.0), ("H", -1.0)):
            hip = f"{end}{side}_hip"
            segs.append(GeneSegment(name=hip, parent="base", shape="capsule", length_m=0.08, radius_m=0.04,
                                    mass_kg=0.678, joint_type="revolute", joint_axis=(1.0, 0.0, 0.0),
                                    joint_lower=-0.8, joint_upper=0.8, actuator_torque_nm=23.7,
                                    mount_offset=(sx * 0.16, sy * 0.09, -0.38)))
            segs.append(GeneSegment(name=f"{end}{side}_thigh", parent=hip, shape="capsule", length_m=0.21,
                                    radius_m=0.03, mass_kg=1.152, joint_type="revolute",
                                    joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.5, joint_upper=1.5,
                                    actuator_torque_nm=23.7))
            segs.append(GeneSegment(name=f"{end}{side}_calf", parent=f"{end}{side}_thigh", shape="capsule",
                                    length_m=0.20, radius_m=0.02, mass_kg=0.241, joint_type="fixed",
                                    is_end_effector=(side == "L" and end == "F")))  # gene.validate wants one
    g = RobotGene(id="cust_quad", species="imported.quadruped", robot_class="quadruped", segments=segs,
                  base_mount="free")
    g.metadata = {"mass_source": "source_model", "imported_from": "customer_model.xml",
                  "torque_source": "source_model"}
    return g


def _mass(gene):
    return round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 4)


class AmendPreservesTheCustomersMassTests(unittest.TestCase):
    """DO1 — an added limb adds ITS mass and nothing else."""

    def test_add_limb_adds_only_its_own_mass(self):
        from virturoid.services.edit_operators import add_limb
        gene = _imported_quadruped()
        before = {s.name: s.mass_kg for s in gene.segments}
        new, diff = add_limb(gene, segments=3, length_m=0.25, radius_m=0.03, attach="top", name="arm")
        added = set(diff["segments_added"])
        moved = {n: (before[n], s.mass_kg) for s in new.segments
                 if (n := s.name) in before and abs(s.mass_kg - before[n]) > 1e-3}
        self.assertEqual(moved, {}, f"the customer's own links were re-massed by an add_limb: {moved}")
        arm_kg = sum(float(s.mass_kg or 0.0) for s in new.segments if s.name in added)
        self.assertAlmostEqual(_mass(new) - _mass(gene), arm_kg, places=3,
                               msg="the robot gained mass the added limb does not account for")

    def test_the_diff_states_the_mass_it_moved(self):
        """DO1's second half: an unavoidable change must be DISCLOSED with the number."""
        from virturoid.services.edit_operators import add_limb
        gene = _imported_quadruped()
        new, diff = add_limb(gene, segments=2, length_m=0.2, radius_m=0.02, attach="back", name="mast")
        led = diff["mass"]
        self.assertEqual(led["total_mass_kg"], [_mass(gene), _mass(new)])
        self.assertEqual(led["n_existing_links_remassed"], 0)
        self.assertEqual(led["existing_mass_changed_kg"], 0.0)
        self.assertTrue(led["source_masses_preserved"])
        self.assertIn(f"{led['added_mass_kg']:.3f} kg", diff["note"])

    def test_a_body_we_generated_is_still_re_derived(self):
        """The protection is keyed on PROVENANCE, not on the op: a body with no manufacturer mass to keep
        must still have its masses tracked to its geometry, or 'make the legs longer' would leave a longer
        leg weighing what the shorter one did."""
        from virturoid.services.edit_operators import scale_group
        gene = _imported_quadruped()
        gene.metadata = {}                                  # composed, not imported
        from virturoid.services.grounded_physics import ground_gene
        ground_gene(gene, material="aluminum", fill=0.25)
        before = {s.name: s.mass_kg for s in gene.segments}
        new, _ = scale_group(gene, group="legs", dims="both", factor=1.5)
        grew = [s.name for s in new.segments if s.name in before and s.mass_kg > before[s.name] + 1e-3]
        self.assertTrue(grew, "a generated body's links must re-derive their mass when they are resized")

    def test_a_resized_link_on_an_imported_body_is_re_derived_and_named(self):
        """...and on an IMPORTED body only the links the customer actually asked to change may move."""
        from virturoid.services.edit_operators import scale_group, segments_for_group
        gene = _imported_quadruped()
        before = {s.name: s.mass_kg for s in gene.segments}
        touched = {s.name for s in segments_for_group(gene, "legs")}
        new, diff = scale_group(gene, group="legs", dims="both", factor=1.5)
        moved = {s.name for s in new.segments if s.name in before and abs(s.mass_kg - before[s.name]) > 1e-3}
        self.assertTrue(moved, "precondition: resizing a link must move its mass")
        self.assertTrue(moved <= touched, f"links nobody resized were re-massed: {sorted(moved - touched)}")
        self.assertEqual(diff["mass"]["n_existing_links_remassed"], len(moved))
        self.assertTrue(diff["mass"]["remassed"], "a mass that moved must be named with its number")

    def test_ground_gene_derive_mass_links_is_scoped(self):
        """The mechanism under it: ``preserve_mass`` releases exactly the named links and no others."""
        from virturoid.services.grounded_physics import ground_gene
        gene = _imported_quadruped()
        before = {s.name: s.mass_kg for s in gene.segments}
        rep = ground_gene(gene, preserve_mass=True, derive_mass_links={"FL_thigh"})
        self.assertEqual(rep["mass_derived_links"], ["FL_thigh"])
        moved = {s.name for s in gene.segments if abs(s.mass_kg - before[s.name]) > 1e-3}
        self.assertEqual(moved, {"FL_thigh"}, f"preservation leaked to {moved - {'FL_thigh'}}")


@unittest.skipUnless(_MUJOCO, "attach placement is measured on the compiled body")
class AttachPlacesTheLimbWhereItSaysTests(unittest.TestCase):
    """DO3 — 'top' is the top of the ROBOT, on any body, in any reconstructed root frame."""

    #: the imported Go2's own reconstructed root rotation: local +z lands on world (0.972, 0.234, 0.000)
    GO2_ROOT_EULER = (-1.5707963267948957, 1.3348407086313339, 1.3348407086313332)

    def _tip_offset(self, gene, tip_name):
        import mujoco
        from virturoid.services.gene_compiler import compile_gene_to_mjcf
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False, spawn_z=0.5))
        d = mujoco.MjData(mj)
        if mj.nkey:                                   # the rest posture is the pose the customer is shown
            mujoco.mj_resetDataKeyframe(mj, d, 0)
        mujoco.mj_forward(mj, d)
        bid = {mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(mj.nbody)}
        return d.xpos[bid[tip_name]] - d.xpos[bid[gene.root().name]]

    def test_top_goes_up_even_when_the_root_frame_is_rotated(self):
        from virturoid.services.edit_operators import add_limb
        gene = _imported_quadruped(root_euler=self.GO2_ROOT_EULER)
        new, diff = add_limb(gene, segments=3, length_m=0.25, radius_m=0.03, attach="top", name="arm")
        p = self._tip_offset(new, "arm_2")
        self.assertGreater(float(p[2]), 0.35, f"'top' did not go up: arm tip at {p}")
        self.assertLess(abs(float(p[0])), 0.30, f"'top' drifted forward: arm tip at {p}")
        self.assertLess(abs(float(p[1])), 0.20, f"'top' drifted off centreline: arm tip at {p}")
        self.assertEqual(diff["placement"]["grows_toward"], [0.0, 0.0, 1.0])

    def test_every_face_grows_along_its_own_axis(self):
        from virturoid.services.edit_operators import add_limb
        want = {"top": (2, +1), "bottom": (2, -1), "front": (0, +1),
                "back": (0, -1), "left": (1, +1), "right": (1, -1)}
        for face, (axis, sign) in want.items():
            with self.subTest(attach=face):
                gene = _imported_quadruped(root_euler=self.GO2_ROOT_EULER)
                new, _ = add_limb(gene, segments=2, length_m=0.2, radius_m=0.02, attach=face, name="p",
                                  end_effector="none")
                p = self._tip_offset(new, "p_1")
                dom = max(range(3), key=lambda i: abs(float(p[i])))
                self.assertEqual(dom, axis, f"attach '{face}' grew along axis {dom}, not {axis} ({p})")
                self.assertEqual(math.copysign(1.0, float(p[axis])), float(sign),
                                 f"attach '{face}' grew the wrong way ({p})")

    def test_rest_angles_bend_the_chain(self):
        """A chain at all-zero joints is a straight mast; an arm needs a shoulder and an elbow to read as one."""
        from virturoid.services.edit_operators import add_limb
        gene = _imported_quadruped(root_euler=self.GO2_ROOT_EULER)
        straight, _ = add_limb(gene, segments=3, length_m=0.25, attach="top", name="arm")
        bent, _ = add_limb(gene, segments=3, length_m=0.25, attach="top", name="arm",
                           rest_angles=[0.0, -0.75, 1.5])
        self.assertEqual((bent.metadata or {}).get("rest_pose", {}).get("arm_1_joint"), -0.75)
        import numpy as np
        self.assertGreater(float(np.linalg.norm(self._tip_offset(bent, "arm_2")
                                                - self._tip_offset(straight, "arm_2"))), 0.05)


class RefusalIsActionableTests(unittest.TestCase):
    """DO2 — a refusal names the findings, the parts and the way forward."""

    def test_symmetry_is_not_a_reason_to_refuse_an_added_limb(self):
        from virturoid.services.edit_operators import expected_findings
        self.assertEqual(expected_findings([{"op": "add_limb"}]), {"symmetry"})
        self.assertEqual(expected_findings([{"op": "scale_group"}]), set())

    def test_findings_are_named_not_counted(self):
        from virturoid.services.edit_operators import add_limb, explain_findings
        gene = _imported_quadruped()
        # a deliberately absurd boom: heavy enough to break real checks, so there IS something to name
        new, _ = add_limb(gene, segments=6, length_m=1.6, radius_m=0.3, attach="front", name="boom")
        out = explain_findings(gene, new, ops=[{"op": "add_limb"}])
        self.assertTrue(out["new"], "precondition: this edit does introduce findings")
        for f in out["new"]:
            self.assertTrue(f["check"] and f["detail"], f"an unnamed finding: {f}")
        self.assertEqual(out["expected_checks"], ["symmetry"])
        self.assertNotIn("symmetry", {f["check"] for f in out["blocking"]})

    @unittest.skipUnless(_MUJOCO, "the gate runs the compiled verifiers")
    def test_the_tool_refusal_quotes_the_findings_and_the_override(self):
        from virturoid.services import session_state as S
        from virturoid.services.ai_native_tools import edit_robot
        env = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = tempfile.mkdtemp(prefix="amend_refusal_")
        try:
            rid = S.put_robot(_imported_quadruped(), label="imported")
            out = edit_robot({"robot_id": rid, "ops": [
                {"op": "add_limb", "args": {"segments": 6, "length_m": 1.6, "radius_m": 0.3,
                                            "attach": "front", "name": "boom"}}]})
            self.assertFalse(out["ok"], "precondition: this absurd boom is refused")
            msg = out["error"]
            self.assertTrue(out["blocking_findings"], "a refusal with no named finding is the old refusal")
            for f in out["blocking_findings"]:
                self.assertIn(f["check"], msg, "the message must name every blocking finding")
            self.assertIn("gate_non_regression", msg, "the way forward must be in the message, not the schema")
            self.assertEqual(out["override"]["arg"], "gate_non_regression")
            self.assertNotEqual(msg, "edit auto-reverted because deterministic design findings regressed")
        finally:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
            if env is not None:
                os.environ["VIRTUROID_SESSIONS_DIR"] = env


@unittest.skipUnless(_MUJOCO and _HAVE_GO2, "needs the MuJoCo Menagerie cache (a real robot, not a fixture)")
class TheRealGo2JourneyTests(unittest.TestCase):
    """DO4 — the whole thing through ``call_tool`` on the customer's actual robot. A fixture cannot show this:
    the Go2's real per-link masses already contain its motors, and its root frame is really rotated."""

    def test_ingest_then_add_an_arm_and_the_mass_is_still_theirs(self):
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        env = os.environ.get("VIRTUROID_SESSIONS_DIR")
        os.environ["VIRTUROID_SESSIONS_DIR"] = tempfile.mkdtemp(prefix="go2_amend_")
        try:
            ing = call_tool("ingest_project", {"path": str(_GO2), "description": "our Unitree Go2"})
            self.assertTrue(ing["ok"], ing.get("error"))
            rid = ing["result"]["robot_id"]
            self.assertTrue(ing["result"]["mass_provenance"]["preserved"], "precondition: ingest preserved")
            m0 = _mass(S.get_robot(rid))

            out = call_tool("edit_robot", {"robot_id": rid, "ops": [
                {"op": "add_limb", "args": {"segments": 3, "length_m": 0.25, "radius_m": 0.03,
                                            "attach": "top", "name": "arm", "end_effector": "gripper"}}]})
            self.assertTrue(out["ok"], f"the #1 amend was refused: {out.get('error')}")
            diff = out["result"]["diffs"][0]
            gene = S.get_robot(rid)
            m1 = _mass(gene)
            arm = sum(float(s.mass_kg or 0.0) for s in gene.segments if s.name.startswith("arm"))
            self.assertAlmostEqual(m1 - m0, arm, places=2,
                                   msg=f"{m0} -> {m1} kg but the arm is only {arm} kg")
            self.assertEqual(diff["mass"]["n_existing_links_remassed"], 0)
            self.assertEqual(gene.metadata.get("mass_source"), "source_model")
            self.assertIsNone(gene.metadata.get("mass_source_replaced"))
            # ...and the arm is ON TOP of the dog, not out of its nose
            self.assertEqual(diff["placement"]["grows_toward"], [0.0, 0.0, 1.0])
            self.assertGreater(diff["placement"]["anchor_m_from_root"][2], 0.0)
        finally:
            os.environ.pop("VIRTUROID_SESSIONS_DIR", None)
            if env is not None:
                os.environ["VIRTUROID_SESSIONS_DIR"] = env


if __name__ == "__main__":
    unittest.main()
