"""Enterprise ingestion hardening: a customer drops a folder (model + BOM + CAD + NLP + their control script) and
our sim parses + improves it. Covers the fixes that made a real drop-folder work:

  * URDF import is ROBUST to missing meshes (the #1 real-world problem) -> imports the structure, doesn't hard-fail
  * a multi-limb URDF is classified LEGGED (quadruped), not defaulted to 'manipulator'
  * a gene -> URDF export carries the real JOINT AXES + primitive collision (was flattening every axis to +z)
  * control_adopter maps an imported controller's params into the gait-search space
  * ingest_project surfaces detected control scripts; adopt_control_script runs + improves one

MuJoCo-gated where a real compile/sim is needed; the pure-Python bits always run.
"""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None

# a proper single-root quadruped URDF whose visual+collision are MESHES that do NOT exist next to the file
_QUAD_URDF = """<?xml version="1.0"?>
<robot name="cust_quad">
  <link name="torso"><inertial><mass value="2.0"/>
    <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><mesh filename="meshes/torso.stl"/></geometry></visual>
    <collision><geometry><mesh filename="meshes/torso.stl"/></geometry></collision></link>
  {legs}
</robot>"""


def _leg(side, dx, dy):
    links, joints = [], []
    parent = "torso"
    for k in range(3):
        name = f"leg_{side}_{k}"
        links.append(
            f'<link name="{name}"><inertial><mass value="0.3"/>'
            f'<inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>'
            f'<visual><geometry><mesh filename="meshes/{name}.stl"/></geometry></visual>'
            f'<collision><geometry><mesh filename="meshes/{name}.stl"/></geometry></collision></link>')
        axis = "0 1 0" if k == 0 else "1 0 0"
        origin = f"{dx} {dy} 0" if k == 0 else "0 0 -0.12"
        joints.append(
            f'<joint name="{name}_j" type="revolute"><parent link="{parent}"/><child link="{name}"/>'
            f'<origin xyz="{origin}"/><axis xyz="{axis}"/>'
            f'<limit lower="-1.5" upper="1.5" effort="20" velocity="10"/></joint>')
        parent = name
    return "\n".join(links + joints)


def _quad_urdf() -> str:
    legs = "\n".join(_leg(s, dx, dy) for s, dx, dy in
                     [("fl", 0.15, 0.1), ("fr", 0.15, -0.1), ("bl", -0.15, 0.1), ("br", -0.15, -0.1)])
    return _QUAD_URDF.format(legs=legs)


class ControlAdopterUnitTests(unittest.TestCase):
    def test_maps_our_exported_cpg_format(self):
        from virturoid.services.control_adopter import map_control_params
        p = map_control_params({"frequency_hz": 1.8, "amplitude": [0.0, 0.6, 0.9, 0.0, 0.6, 0.9]})
        self.assertAlmostEqual(p["freq"], 1.8, places=3)
        self.assertLess(p["hip_amp"], p["knee_amp"])              # smaller nonzero = hip, larger = knee
        # `duty` is deliberately absent: it was removed from the gait-search space (task #265) because the
        # crawl controller never read it, so adopting an imported one would only pretend to honor it.
        self.assertTrue(all(k in p for k in ("freq", "hip_amp", "knee_amp", "kp", "kd")))
        self.assertNotIn("duty", p)

    def test_explicit_gait_keys_win_and_clamp(self):
        from virturoid.services.control_adopter import map_control_params
        p = map_control_params({"freq": 99.0, "hip_amp": 0.7})    # 99 is out of range -> clamped, not passed through
        self.assertLessEqual(p["freq"], 3.2)
        self.assertGreaterEqual(p["freq"], 0.8)


@unittest.skipUnless(_MUJOCO, "URDF import/compile needs MuJoCo")
class ImportRobustnessTests(unittest.TestCase):
    def _write(self):
        d = tempfile.mkdtemp(prefix="cust_")
        Path(d, "quad.urdf").write_text(_quad_urdf(), encoding="utf-8")   # meshes/ intentionally absent
        return os.path.join(d, "quad.urdf")

    def test_missing_meshes_still_import(self):
        from virturoid.services.robot_import import import_robot
        imp = import_robot(self._write())
        self.assertTrue(imp["valid"], imp.get("warnings"))
        self.assertIsNotNone(imp["gene"])
        self.assertTrue(any("mesh" in w.lower() and "missing" in w.lower() for w in imp["warnings"]),
                        "a missing-mesh import must be flagged, not silent")

    def test_multi_limb_classifies_legged(self):
        from virturoid.services.robot_import import import_robot
        self.assertEqual(import_robot(self._write())["robot_class"], "quadruped")


@unittest.skipUnless(_MUJOCO, "export round-trip needs MuJoCo")
class ExportFidelityTests(unittest.TestCase):
    def test_exported_urdf_carries_real_axes_and_box_collision(self):
        from virturoid.services.gene_build import _gene_to_genome
        from virturoid.schemas.robot import RobotGenome
        from virturoid.services.morphology_composer import compose_robot
        from virturoid.services.anatomy_compiler import ensure_walkable_quad
        from virturoid.services.urdf_exporter import robot_genome_to_urdf
        gene = ensure_walkable_quad(compose_robot("a quadruped robot dog"), "a quadruped robot dog")
        urdf = robot_genome_to_urdf(RobotGenome.from_dict(_gene_to_genome(gene)))
        self.assertIn("<box", urdf)                                # self-contained collision (not mesh-only)
        # at least one joint axis is NOT the default +z (the bug flattened every axis to 0 0 1)
        import re
        axes = re.findall(r'<axis xyz="([^"]+)"', urdf)
        self.assertTrue(any(a.strip() not in ("0.0 0.0 1.0", "0 0 1") for a in axes),
                        "exported joints must keep their real axes, not default to +z")


@unittest.skipUnless(_MUJOCO, "ingest + adopt run real sims")
class IngestAndAdoptE2ETests(unittest.TestCase):
    def setUp(self):
        os.environ["VIRTUROID_SESSION_DIR"] = tempfile.mkdtemp(prefix="cust_sess_")

    def test_ingest_surfaces_control_scripts(self):
        from virturoid.services.input_training_tools import _ingest_project
        d = tempfile.mkdtemp(prefix="proj_")
        Path(d, "robot").mkdir()
        Path(d, "robot", "quad.urdf").write_text(_quad_urdf(), encoding="utf-8")
        Path(d, "gait_controller.py").write_text("class GaitController:\n    def act(self, obs): return obs\n", encoding="utf-8")
        Path(d, "policy_params.json").write_text('{"frequency_hz": 1.5, "amplitude": [0,0.6,0.8]}', encoding="utf-8")
        r = _ingest_project({"project_path": d, "description": "aluminum body, carbon-fiber legs, carries 5 kg"})
        self.assertTrue(r["robot_id"])
        self.assertTrue(r.get("control_scripts"), "a dropped control script must be surfaced for adopt_control_script")

    def test_ingested_quadruped_is_honestly_verified_and_never_silently_swapped(self):
        # B2 (2026-07-24 audit): the ingest must NOT silently swap the customer's body for a walkable template and
        # report the template's walk as the customer's. It KEEPS the imported geometry, classifies it legged, and
        # reports ITS honest verdict. (Was: auto-swap -> "the ingested quadruped walks" -- a walk that wasn't the
        # customer's.)
        #
        # THE SECOND HALF WAS RE-ANCHORED 2026-08-12 ON A MEASUREMENT, NOT WEAKENED TO GO GREEN. It used to
        # assert that opting into the walkable template "then walks". MEASURED on THIS fixture: the imported body
        # ALREADY walks -- 1.604 m, and ``gait_quality.classify`` on that same rollout returns CREDIBLE WALK
        # (straightness 0.825, upright_frac 1.0, cadence 33.33, survived). So there is nothing for a template to
        # improve: ``walkable_template_offer`` is correctly WITHHELD, and ``ensure_walkable_quad`` correctly
        # returns the body UNCHANGED (its documented "sleek-when-possible" rule). The old assertion demanded a
        # substitution the product is right to refuse -- it was asserting the B2 DEFECT, not the B2 fix, and it
        # only ever passed back when the swap was automatic.
        #
        # What replaces it is STRICTLY STRONGER on the property B2 exists to protect: on a body that already
        # walks, the customer's geometry survives the opt-in BYTE-IDENTICAL. If a future change makes
        # ``adopt_walkable_template`` swap a walking customer body, or makes the ingest offer a template to one,
        # this test fails.
        #
        # KNOWN COVERAGE GAP, stated rather than papered over: the opt-in's ability to actually HELP a body that
        # genuinely cannot walk is not covered anywhere. `test_import_verify_honesty
        # ::test_walkable_template_is_opt_in_and_undoable` looks like it does and does not -- it asserts only
        # that `undo` restores the original body, and never asserts the adopt changed anything, so it passes
        # whether or not the template was ever applied. Closing that needs a fixture quadruped that fails the
        # gait gate, which this file does not have.
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        from virturoid.services.input_training_tools import _ingest_project
        d = tempfile.mkdtemp(prefix="proj_")
        Path(d, "robot").mkdir()
        Path(d, "robot", "quad.urdf").write_text(_quad_urdf(), encoding="utf-8")
        r = _ingest_project({"project_path": d, "description": "a quadruped robot"})
        self.assertTrue(r["robot_id"])
        # the held robot is the customer's OWN geometry, classified legged, honestly verified
        v = call_tool("verify_robot", {"robot_id": r["robot_id"], "mode": "quick"}).get("result", {})
        self.assertEqual(v.get("kind"), "legged")
        iv = r.get("imported_verdict")
        self.assertIsNotNone(iv)
        held = [s.name for s in S.get_robot(r["robot_id"]).segments]
        self.assertNotIn("neck", held)                       # not a composed template body

        # the distance is attributed to the controller that produced it. verify_robot withholds a locomotion
        # verdict for this robot precisely because the gait is ours; the ingest report must not quietly imply
        # otherwise by calling it the customer's walk.
        self.assertIn("walks_under_our_scripted_gait", iv,
                      "the imported distance must not be reported as an unattributed 'walks_as_imported'")
        self.assertIn("OURS, not yours", iv.get("controller", ""),
                      f"imported_verdict must name whose controller produced {iv.get('distance_m')} m")
        self.assertIsNone(v.get("locomotion_verdict"),
                          "verify_robot must still withhold a locomotion verdict without the customer's controller")

        # this body already walks credibly under our gait, so no template is offered...
        self.assertIsNone((r.get("walkable_template_offer") or {}).get("available"),
                          "a template must not be offered to a body that already walks")
        # ...and opting in anyway must NOT cost the customer their geometry.
        before = [s.name for s in S.get_robot(r["robot_id"]).segments]
        call_tool("edit_robot", {"robot_id": r["robot_id"], "ops": [{"op": "adopt_walkable_template"}]})
        after = [s.name for s in S.get_robot(r["robot_id"]).segments]
        self.assertEqual(before, after,
                         "adopt_walkable_template replaced a body that already walks -- that is the B2 defect")

    def test_adopt_control_script_utilises_and_improves(self):
        # the user's controller RUNS in our sim and the sim tunes it -> credible walk that beats it (on a body our
        # stack drives). Small budget = smoke; asserts the improve loop produces a credible, non-worse result.
        from virturoid.services.agent_tools import call_tool
        rid = call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]["robot_id"]
        out = call_tool("adopt_control_script", {"robot_id": rid, "generations": 4, "pop": 10, "steps": 600,
                                                 "params": {"frequency_hz": 1.5, "amplitude": [0, 0.6, 0.8]}}).get("result", {})
        self.assertNotIn("error", out)
        self.assertIn("utilised", out)
        self.assertIn("improved", out)
        self.assertTrue(out["improved"]["credible"], "the tuned gait should be a credible walk")


if __name__ == "__main__":
    unittest.main()
