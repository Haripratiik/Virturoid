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
        self.assertLessEqual(len(resp[2]["result"]["tools"]), 15)
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


if __name__ == "__main__":
    unittest.main()
