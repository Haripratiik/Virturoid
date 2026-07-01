"""Agentic tool surface: discoverable schema'd tools + a structured-error dispatcher an agent can drive.

Pure-Python (the fast tools need no MuJoCo); build_robot/evaluate_robot are heavy real-physics tools,
exercised elsewhere, so here we test the registry mechanics + the fast tools.
"""

import tempfile
import unittest
from pathlib import Path

from virturoid.services.agent_tools import call_tool, tool_specs


class AgentToolsTests(unittest.TestCase):
    def test_tool_specs_are_discoverable_and_schemad(self):
        specs = tool_specs()
        names = {s["name"] for s in specs}
        self.assertLessEqual({"list_tools", "capabilities", "search_memory", "design_brain",
                              "build_robot", "evaluate_robot"}, names)
        for s in specs:
            self.assertIn("description", s)
            self.assertEqual(s["parameters"]["type"], "object")   # JSON-Schema object an LLM/MCP can fill
        heavy = {s["name"] for s in specs if s["heavy"]}
        self.assertIn("build_robot", heavy)                       # real-physics tools flagged for latency
        self.assertNotIn("search_memory", heavy)

    def test_list_tools_is_self_describing(self):
        r = call_tool("list_tools")
        self.assertTrue(r["ok"])
        self.assertTrue(r["result"]["tools"])

    def test_unknown_tool_returns_structured_error(self):
        r = call_tool("does_not_exist")
        self.assertFalse(r["ok"])
        self.assertIn("unknown tool", r["error"])

    def test_missing_required_argument_is_structured(self):
        r = call_tool("search_memory", {})                        # missing 'query'
        self.assertFalse(r["ok"])
        self.assertIn("query", r["error"])

    def test_design_brain_tool_on_empty_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = call_tool("design_brain", {"memory_dir": tmp})
            self.assertTrue(r["ok"])
            self.assertEqual(r["result"]["archive_coverage"], 0)

    def test_search_memory_tool_retrieves_prior_work(self):
        from virturoid.services.memory_db import MemoryDB
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryDB(Path(tmp) / "virturoid_memory.db") as db:
                db.record_run("sort red and blue blocks", "manipulator", "pick_place_sort", None, 0.9)
            r = call_tool("search_memory", {"query": "sort colored blocks", "memory_dir": tmp})
            self.assertTrue(r["ok"])
            self.assertTrue(r["result"]["hits"])
            self.assertEqual(r["result"]["hits"][0]["task_type"], "pick_place_sort")

    def test_capabilities_tool(self):
        r = call_tool("capabilities")
        self.assertTrue(r["ok"])
        self.assertTrue(r["result"])   # registry summary or the known-tasks fallback


if __name__ == "__main__":
    unittest.main()
