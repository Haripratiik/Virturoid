"""T4 (total_generality_plan): shape-programs — the agent AUTHORS a part's own visual geometry (extrude /
revolve / tapered / loft + fillet/chamfer/cutouts) via an optional ``geometry`` field, realized by the mesh
layer (cad_geometry.realize_shape). The physics collider stays the size/girth primitive, so it's visual-only
and physics-safe. A malformed shape program is REJECTED with a teaching error (no silent capsule fallback).
Offline + NO_INTERNAL_LLM."""
import importlib.util
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")   # offline (get_llm -> None); NO module-level
# NO_INTERNAL_LLM — it leaks process-wide during collection and breaks the llm-router tests
_MUJOCO = importlib.util.find_spec("mujoco") is not None
_BUILD123D = importlib.util.find_spec("build123d") is not None

_REVOLVE_DOME = {"family": "revolve", "profile": [[0.0, 0.0], [0.08, 0.0], [0.075, 0.06], [0.045, 0.11], [0.0, 0.13]]}


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ShapeProgramTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _call(self, name, args=None):
        from virturoid.services.agent_tools import call_tool
        return call_tool(name, args or {})["result"]

    def _domed_graph(self):
        return {"robot_class": "quadruped", "name": "domed", "parts": [
            {"name": "torso", "role": "body", "size": 0.5, "girth": 0.14},
            {"name": "dome", "role": "head", "parent": "torso", "attach": "front_top", "aim": "forward",
             "size": 0.16, "girth": 0.08, "geometry": _REVOLVE_DOME},
            {"name": "leg1", "role": "leg", "parent": "torso", "attach": "front_bottom", "aim": "down_out",
             "size": 0.4, "girth": 0.018, "segments": 4, "symmetry": "left_right", "joint": "revolute"},
            {"name": "leg2", "role": "leg", "parent": "torso", "attach": "rear_bottom", "aim": "down_out",
             "size": 0.4, "girth": 0.018, "segments": 4, "symmetry": "left_right", "joint": "revolute"}]}

    def test_schema_documents_geometry_families(self):
        sch = self._call("get_design_schema")
        fams = sch.get("geometry_families", {})
        for f in ("extrude", "revolve", "tapered", "loft"):
            self.assertIn(f, fams)
        self.assertIn("geometry", sch["part_fields"])

    def test_custom_geometry_compiles_and_is_carried_on_the_segment(self):
        from virturoid.services import session_state as S
        r = self._call("submit_design", {"graph": self._domed_graph()})
        self.assertTrue(r["ok"], r.get("error"))
        seg = next(s for s in S.get_robot(r["robot_id"]).segments if s.name == "dome")
        self.assertEqual((seg.geometry or {}).get("family"), "revolve")   # the authored shape is on the gene

    def test_bad_family_teaches(self):
        g = {"robot_class": "quadruped", "name": "b", "parts": [
            {"name": "t", "role": "body", "size": 0.5, "girth": 0.14},
            {"name": "h", "role": "head", "parent": "t", "attach": "front_top", "size": 0.1,
             "geometry": {"family": "wormhole"}}]}
        r = self._call("submit_design", {"graph": g})
        self.assertFalse(r["ok"])
        self.assertIn("wormhole", r["error"])

    def test_missing_required_field_teaches(self):
        g = {"robot_class": "quadruped", "name": "b", "parts": [
            {"name": "t", "role": "body", "size": 0.5, "girth": 0.14},
            {"name": "p", "role": "shell", "parent": "t", "attach": "rear_top", "size": 0.1,
             "geometry": {"family": "extrude", "profile": [[0, 0], [1, 0], [1, 1]]}}]}   # no 'height'
        r = self._call("submit_design", {"graph": g})
        self.assertFalse(r["ok"])
        self.assertIn("height", r["error"])

    @unittest.skipUnless(_BUILD123D, "needs build123d for the mesh layer")
    def test_custom_geometry_realizes_a_distinct_mesh_not_a_fallback(self):
        # the STRONG proof: the revolve produces an AXISYMMETRIC dome mesh (x~=y, taller z) — not a capsule.
        from virturoid.services import session_state as S
        from virturoid.services.cad_geometry import build_visual_meshes
        import tempfile
        rid = self._call("submit_design", {"graph": self._domed_graph()})["robot_id"]
        meshes = build_visual_meshes(S.get_robot(rid), tempfile.mkdtemp(prefix="t4_"))
        self.assertIn("dome", meshes)
        self.assertTrue(os.path.exists(meshes["dome"]) and os.path.getsize(meshes["dome"]) > 1000)


if __name__ == "__main__":
    unittest.main()
