"""AI-native layer (docs/ai_native_plan.md) — the MCP server + in-app assistant + incremental edit operators.
Guards the two capabilities the user asked for: "make the robot a little taller" (a LOCALIZED, undoable edit
on the EXISTING gene, not a rebuild) and "change the scene to a house instead of a warehouse" (a theme swap
keeping task/robot). Plus the shared registry (MCP == in-app) and the honest-verdict / teaching-error contract.
Offline (llm=None / no local env) so the acceptance conversations are deterministic in CI.
"""
import importlib.util
import io
import json
import os
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class EditOperatorTests(unittest.TestCase):
    def _dog(self):
        from virturoid.services.morphology_composer import compose_robot
        return compose_robot("a quadruped robot dog", llm=None)

    def test_scale_group_is_localized_and_taller(self):
        from virturoid.services import edit_operators as EO
        g = self._dog()
        before = {s.name: (round(s.length_m, 5), round(s.radius_m, 5)) for s in g.segments}
        g2, diff = EO.apply_op(g, "scale_group", {"group": "legs", "dims": "length", "factor": 1.2})
        h0, h1 = diff["standing_height_m"]
        self.assertGreater(h1, h0 * 1.1, "lengthening the legs must make it taller")
        # LOCALIZED: non-leg segments unchanged; leg segments changed
        after = {s.name: (round(s.length_m, 5), round(s.radius_m, 5)) for s in g2.segments}
        self.assertEqual(set(before), set(after), "segment identity is preserved (no rebuild)")
        for name in before:
            if "leg" not in name.lower():
                self.assertEqual(before[name], after[name], f"non-leg segment {name} must be untouched")

    def test_set_height_hits_target(self):
        from virturoid.services import edit_operators as EO
        g2, diff = EO.apply_op(self._dog(), "set_height", {"target_m": 0.5})
        self.assertAlmostEqual(diff["standing_height_m"][1], 0.5, delta=0.06)

    def test_unknown_group_teaches(self):
        from virturoid.services import edit_operators as EO
        with self.assertRaises(EO.EditError) as cm:
            EO.apply_op(self._dog(), "scale_group", {"group": "wings", "dims": "length", "factor": 1.2})
        self.assertIn("valid groups", str(cm.exception))       # the error tells the agent how to fix it


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class SessionAndToolTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def test_create_edit_undo_round_trip(self):
        from virturoid.services.agent_tools import call_tool
        rid = call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]["robot_id"]
        h0 = call_tool("get_robot", {"robot_id": rid})["result"]["standing_height_m"]
        e = call_tool("edit_robot", {"robot_id": rid,
                                     "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": 1.3}}]})
        self.assertTrue(e["result"]["ok"])
        h1 = call_tool("get_robot", {"robot_id": rid})["result"]["standing_height_m"]
        self.assertGreater(h1, h0)
        self.assertTrue(call_tool("undo_robot", {"robot_id": rid})["result"]["ok"])
        self.assertAlmostEqual(call_tool("get_robot", {"robot_id": rid})["result"]["standing_height_m"], h0, delta=1e-4)

    def test_scene_theme_swap_house_over_warehouse(self):
        from virturoid.services import session_state as S
        from virturoid.services.agent_tools import call_tool
        sid = call_tool("create_scene", {"task": "navigation", "theme": "warehouse"})["result"]["scene_id"]
        r = call_tool("edit_scene", {"scene_id": sid, "ops": [{"op": "swap_theme", "args": {"theme": "house"}}]})
        self.assertTrue(r["result"]["ok"])
        self.assertEqual(r["result"]["theme"], "house")
        d = S.get_scene(sid)
        floor = next(o for o in d["objects"] if o.get("category") == "floor")
        self.assertEqual(floor["material"], "wood", "a house has a wood floor, not concrete")
        self.assertTrue(any(o.get("category") == "goal" for o in d["objects"]), "the task (goal) is preserved")


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class McpServerTests(unittest.TestCase):
    def test_initialize_tools_list_and_call(self):
        from virturoid import mcp_server
        from virturoid.services.agent_tools import tool_specs
        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "create_robot", "arguments": {"prompt": "a quadruped robot dog"}}},
        ]
        out = io.StringIO()
        mcp_server.serve(io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
        resp = {json.loads(l)["id"]: json.loads(l) for l in out.getvalue().strip().splitlines()}
        self.assertEqual(resp[1]["result"]["serverInfo"]["name"], "virturoid")
        # G-G: MCP advertises the CONSOLIDATED <=15-tool view, not the whole registry (Cursor's ~40-tool cap).
        self.assertEqual(len(resp[2]["result"]["tools"]), len(tool_specs(view="mcp")), "MCP lists the consolidated view")
        self.assertLessEqual(len(resp[2]["result"]["tools"]), 17)
        self.assertTrue(all("inputSchema" in t for t in resp[2]["result"]["tools"]))
        self.assertFalse(resp[3]["result"]["isError"])
        self.assertIn("robot_id", resp[3]["result"]["structuredContent"])


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class AssistantConversationTests(unittest.TestCase):
    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _new_dog(self):
        from virturoid.services.agent_tools import call_tool
        return call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]["robot_id"]

    def test_make_it_taller(self):
        from virturoid.services.assistant_core import handle_turn
        rid = self._new_dog()
        t = handle_turn("make the robot a little taller", robot_id=rid, llm=None)
        self.assertEqual(t["scope"], "parameter")
        self.assertTrue(t["applied"])
        h = t["diffs"][0]["standing_height_m"]
        self.assertGreater(h[1], h[0])

    def test_make_it_carry_heavier(self):
        # the user's own question: "if i ask the robot to lift heavier stuff, does it amend the robot?" -> yes
        from virturoid.services.assistant_core import handle_turn
        rid = self._new_dog()
        t = handle_turn("make it carry 15 kg of payload", robot_id=rid, llm=None)
        self.assertEqual(t["scope"], "feature")
        self.assertTrue(t["applied"], "a capability amend (heavier payload) must land as a real edit")
        d = t["diffs"][0]
        self.assertEqual(d["op"], "set_payload")
        self.assertGreater(d["load_factor"], 1.0)
        self.assertGreater(d["total_mass_kg"][1], d["total_mass_kg"][0])   # stronger motors -> heavier robot (honest cost)

    def test_house_instead_of_warehouse(self):
        from virturoid.services.agent_tools import call_tool
        from virturoid.services.assistant_core import handle_turn
        sid = call_tool("create_scene", {"task": "navigation", "theme": "warehouse"})["result"]["scene_id"]
        t = handle_turn("change the scene to a house instead of a warehouse", scene_id=sid, llm=None)
        self.assertEqual(t["scope"], "scene")
        self.assertTrue(t["applied"])
        self.assertIn("house", t["reply"].lower())            # the TARGET theme, not the one it replaced

    def test_structural_change_asks_confirmation(self):
        from virturoid.services.assistant_core import handle_turn
        t = handle_turn("make it a hexapod", robot_id=self._new_dog(), llm=None)
        self.assertEqual(t["scope"], "structure")
        self.assertTrue(t["needs_confirmation"])
        self.assertFalse(t["applied"], "a structural rebuild must NOT auto-apply; it confirms first")

    def test_undo_in_conversation(self):
        from virturoid.services.agent_tools import call_tool
        from virturoid.services.assistant_core import handle_turn
        rid = self._new_dog()
        h0 = call_tool("get_robot", {"robot_id": rid})["result"]["standing_height_m"]
        handle_turn("make it taller", robot_id=rid, llm=None)
        t = handle_turn("undo", robot_id=rid, llm=None)
        self.assertTrue(t["applied"])
        self.assertAlmostEqual(call_tool("get_robot", {"robot_id": rid})["result"]["standing_height_m"], h0, delta=1e-4)


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class RenderViewPathTests(unittest.TestCase):
    """``render_view`` is the tool an engineer calls to SEE the robot. It must hand back a path that OPENS.

    Measured before the fix, through ``call_tool`` on an imported Go2: the PNG was written, and the result
    carried no ``path`` key at all (only ``artifacts``), holding a CWD-relative string
    ``build\\agent_renders\\<id>_view.png``. An MCP host launched from a different working directory cannot
    resolve that, so the picture existed and the caller had nothing that opened it.
    """

    def setUp(self):
        from virturoid.services import session_state as S
        S.reset()

    def _rid(self):
        from virturoid.services.agent_tools import call_tool
        return call_tool("create_robot", {"prompt": "a quadruped robot dog"})["result"]["robot_id"]

    def test_render_view_returns_an_absolute_path_to_a_file_that_exists(self):
        import os

        from virturoid.services.agent_tools import call_tool
        env = call_tool("render_view", {"robot_id": self._rid()})
        self.assertTrue(env["ok"], env)
        r = env["result"]
        self.assertIn("path", r, "the tool for SEEING the robot must report the picture under 'path'")
        self.assertTrue(os.path.isabs(r["path"]), f"a relative path does not resolve for an MCP host: {r['path']}")
        self.assertTrue(os.path.isfile(r["path"]), f"reported a path with no file behind it: {r['path']}")
        self.assertGreater(r["bytes"], 0)
        self.assertEqual(r["artifacts"], [r["path"]], "artifacts must carry the same absolute path")

    def test_the_reported_path_resolves_from_any_working_directory(self):
        """The actual failure mode: the string is handed to a process whose CWD is not the server's."""
        import os
        import tempfile

        from virturoid.services.agent_tools import call_tool
        path = call_tool("render_view", {"robot_id": self._rid()})["result"]["path"]
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            try:                                     # restore BEFORE the dir is removed, or Windows can't rmdir
                os.chdir(elsewhere)
                self.assertTrue(os.path.isfile(path), f"{path} does not open from {elsewhere}")
            finally:
                os.chdir(cwd)

    def test_a_render_that_cannot_be_produced_fails_loudly_with_a_reason(self):
        """No silent ``None``: a render that did not happen says why, and never reports a path."""
        from virturoid.services import ai_native_tools as A
        from virturoid.services.agent_tools import call_tool
        rid = self._rid()
        real = A._render_gene_detail
        A._render_gene_detail = lambda *a, **k: (None, "MUJOCO_GL=osmesa: libOSMesa.so not found")
        try:
            env = call_tool("render_view", {"robot_id": rid})
        finally:
            A._render_gene_detail = real
        self.assertFalse(env["ok"], f"a render that produced nothing must not report ok=True: {env}")
        self.assertIn("libOSMesa", env["error"], "the reason must reach the caller, not be replaced by a phrase")
        self.assertIsNone(env["result"]["path"])
        self.assertEqual(env["result"]["artifacts"], [])

    def test_an_unknown_view_is_an_honest_failure_through_the_dispatcher(self):
        from virturoid.services.agent_tools import call_tool
        env = call_tool("render_view", {"robot_id": self._rid(), "view": "xray"})
        self.assertFalse(env["ok"], env)
        self.assertIn("collision", env["error"], "the refusal names the accepted values")

    def test_the_collision_view_of_an_imported_robot_actually_draws_the_colliders(self):
        """``view='collision'`` promises "the EXACT bodies the physics verdict is computed on".

        On an imported Menagerie robot it drew an EMPTY FLOOR: the source parks its collision geoms in geom
        group 3 (visuals in 0/2), MjvOption defaults to geomgroup [1,1,1,0,0,0], so all 13 of the Go2's
        colliders were in a group the renderer had switched off -- while the cosmetics that WERE visible had
        just been set to alpha 0. A path to a picture of nothing is the same defect as no path at all.
        """
        import os
        from pathlib import Path

        men = Path(os.path.expanduser("~/.cache/robot_descriptions/mujoco_menagerie/unitree_go2"))
        if not men.is_dir():
            self.skipTest("unitree_go2 is not cached locally (robot_descriptions fetches on demand)")
        import numpy as np
        from PIL import Image

        from virturoid.services.agent_tools import call_tool
        rid = call_tool("ingest_project", {"path": str(men)})["result"]["robot_id"]
        env = call_tool("render_view", {"robot_id": rid, "view": "collision"})
        self.assertTrue(env["ok"], env)
        a = np.asarray(Image.open(env["result"]["path"]).convert("RGB")).astype(int)
        # the colliders are drawn in an unmistakable orange (0.95, 0.45, 0.15); the floor and sky are grey-blue,
        # so "red channel well above blue" separates body from scene without depending on exact shading.
        warm = float(((a[..., 0] - a[..., 2]) > 40).mean())
        self.assertGreater(warm, 0.02, f"the collision view drew no colliders (only {warm:.4%} of the frame is "
                                       f"collider-coloured) — this is a picture of an empty floor")


if __name__ == "__main__":
    unittest.main()
