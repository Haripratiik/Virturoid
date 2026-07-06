"""Scripted agent-eval (docs/agent_first_plan.md P4: G-E) — the proof a connected frontier agent (Claude
Code/Codex over MCP) completes the whole canonical loop through the REAL JSON-RPC/stdio transport, not just
the in-process registry. Each step goes through ``mcp_server.serve`` (stdin line -> response line), exactly as
a client speaks to us. Runs with the internal LLM FORCED OFF, so the passing loop is also the measured
zero-our-tokens proof (llm_spend reads internal_calls == 0). This is the CI half of G-E's acceptance; the live
``claude -p`` / ``codex exec`` checklist is manual.
"""
import importlib.util
import io
import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(_MUJOCO, "needs MuJoCo")
class ScriptedAgentEvalTests(unittest.TestCase):
    """Drive the MCP server over stdio JSON-RPC through the full loop, asserting each step's contract."""

    def setUp(self):
        from virturoid.services import session_state as S
        from virturoid.services.llm_client import reset_spend_ledger
        self._tmp = tempfile.mkdtemp(prefix="virt_eval_")
        self._prev_dir = os.environ.get("VIRTUROID_SESSIONS_DIR")
        self._prev_off = os.environ.get("VIRTUROID_NO_INTERNAL_LLM")
        os.environ["VIRTUROID_SESSIONS_DIR"] = self._tmp
        os.environ["VIRTUROID_NO_INTERNAL_LLM"] = "1"          # the zero-token switch: this loop must spend nothing
        S.reset(wipe_disk=True)
        reset_spend_ledger()
        self._id = 0

    def tearDown(self):
        from virturoid.services import session_state as S
        S.reset(wipe_disk=True)
        for k, prev in (("VIRTUROID_SESSIONS_DIR", self._prev_dir), ("VIRTUROID_NO_INTERNAL_LLM", self._prev_off)):
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _rpc(self, method, params=None):
        """One real JSON-RPC round-trip through the stdio serve loop (what a client actually does)."""
        from virturoid import mcp_server
        self._id += 1
        line = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}})
        out = io.StringIO()
        mcp_server.serve(io.StringIO(line + "\n"), out)
        resp = json.loads(out.getvalue().strip())
        self.assertNotIn("error", resp, f"{method} JSON-RPC error: {resp.get('error')}")
        return resp["result"]

    def _call(self, name, arguments):
        """tools/call -> the structured payload (what the agent reads back)."""
        r = self._rpc("tools/call", {"name": name, "arguments": arguments})
        self.assertFalse(r.get("isError"), f"{name} isError: {r.get('content')}")
        return r.get("structuredContent") or {}

    def test_full_canonical_loop_over_jsonrpc_zero_spend(self):
        # 0. initialize — the handshake carries the loop contract + honesty rule for floor clients (Codex).
        init = self._rpc("initialize", {})
        self.assertEqual(init["serverInfo"]["name"], "virturoid")
        for cap in ("tools", "resources", "prompts"):
            self.assertIn(cap, init["capabilities"])
        self.assertIn("verify_robot", init["instructions"])
        self.assertIn("llm_spend", init["instructions"])

        # 1. prompts teach the loop on the CLIENT's tokens
        prompts = {p["name"] for p in self._rpc("prompts/list")["prompts"]}
        self.assertIn("design_robot_workflow", prompts)
        got = self._rpc("prompts/get", {"name": "design_robot_workflow", "arguments": {"goal": "a walking dog"}})
        text = got["messages"][0]["content"]["text"]
        self.assertIn("a walking dog", text)
        self.assertIn("verify_robot", text)                    # the honesty step is taught

        # 2. tools/list is the CONSOLIDATED <=15 view (G-G), and a folded tool is absent from the menu
        tools = [t["name"] for t in self._rpc("tools/list")["tools"]]
        self.assertLessEqual(len(tools), 15)
        self.assertIn("submit_design", tools)
        self.assertNotIn("simulate_gait", tools)               # folded into verify_robot

        # 3-8. the loop, all via tools/call over stdio
        schema = self._call("get_design_schema", {})           # learn the language
        design = self._call("submit_design", {"graph": schema["examples"]["quadruped"]})  # AUTHOR
        self.assertTrue(design["ok"]); rid = design["robot_id"]
        self.assertEqual(self._call("get_robot", {"robot_id": rid})["robot_class"], "quadruped")
        vq = self._call("verify_robot", {"robot_id": rid, "mode": "quick"})               # honest quick verdict
        self.assertEqual(vq["mode"], "quick"); self.assertIn("verdict", vq)
        edit = self._call("edit_robot", {"robot_id": rid,                                  # localized edit
                          "ops": [{"op": "scale_group", "args": {"group": "legs", "dims": "length", "factor": 1.2}}]})
        self.assertTrue(edit["ok"])
        exp = self._call("export_held", {"robot_id": rid, "formats": ["mjcf"]})            # export
        self.assertTrue(exp["ok"] and os.path.exists(exp["artifacts"]["mjcf"]))

        # 9. live STATE resource for the held robot (a client that auto-attaches resources grounds itself)
        res_uris = {r["uri"] for r in self._rpc("resources/list")["resources"]}
        uri = f"virturoid://robot/{rid}/summary"
        self.assertIn(uri, res_uris)
        body = self._rpc("resources/read", {"uri": uri})["contents"][0]["text"]
        self.assertEqual(json.loads(body)["robot_class"], "quadruped")

        # 10. THE PROOF: the whole loop spent zero internal tokens
        spend = self._call("llm_spend", {})
        self.assertTrue(spend["zero_internal_spend"], f"loop must be zero-spend; totals={spend['totals']}")
        self.assertEqual(spend["totals"]["internal_calls"], 0)
        self.assertGreaterEqual(spend["totals"]["blocked"], 0)


if __name__ == "__main__":
    unittest.main()
