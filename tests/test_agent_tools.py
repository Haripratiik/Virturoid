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

    def test_understanding_tools_are_registered(self):
        names = {s["name"] for s in tool_specs()}
        self.assertLessEqual({"describe_robot", "diagnose_body", "nearest_bodies", "recall_knowledge"}, names)

    def test_diagnose_body_tool(self):
        r = call_tool("diagnose_body", {"prompt": "a tabletop arm that grasps a cube"})
        self.assertTrue(r["ok"], r)
        self.assertIn("observations", r["result"])
        self.assertIn("stats", r["result"])

    def test_recall_knowledge_tool_retrieves_seeded_tip(self):
        from virturoid.fixtures.gene_library import tabletop_arm_gene
        from virturoid.services.memory_db import MemoryDB
        from virturoid.services.species_discovery import auto_place_species
        with tempfile.TemporaryDirectory() as tmp:
            with MemoryDB(Path(tmp) / "virturoid_memory.db") as db:
                sp = auto_place_species(tabletop_arm_gene(), db)["species_pattern"]
                db.add_species_tip(sp, "keep the wrist DOF wide for top-down grasps", audience="builder")
            r = call_tool("recall_knowledge", {"prompt": "a tabletop arm that grasps",
                                               "memory_dir": tmp, "task_type": "grasp"})
            self.assertTrue(r["ok"], r)
            self.assertTrue(r["result"]["tips"])

    def test_recall_knowledge_tool_without_memory_is_graceful(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = call_tool("recall_knowledge", {"prompt": "an arm", "memory_dir": tmp})
            self.assertTrue(r["ok"])
            self.assertIn("note", r["result"])

    def test_design_search_tool_runs_a_verified_multistep_search(self):
        # the agentic-surface brick: an external agent can run a bounded verified search (real MuJoCo, kept tiny)
        r = call_tool("design_search", {"prompt": "a quadruped robot that walks", "max_evals": 2, "steps": 120})
        self.assertTrue(r["ok"], r)
        self.assertIn("tree", r["result"])
        self.assertEqual(len(r["result"]["tree"]), r["result"]["n_evals"])   # honest tree of what it tried
        self.assertLessEqual(r["result"]["n_evals"], 2)
        for row in r["result"]["tree"]:
            self.assertIn(row["verdict"], ("pass", "fail"))


class EnvelopeHonestyTests(unittest.TestCase):
    """``ok`` must mean THE CALL DID WHAT WAS ASKED.

    Measured before the fix: 28 of the 28 registry tools that can be driven to fail with a junk required
    argument returned ``{"ok": true, "result": {"error": "..."}}``. The MCP wire sets ``isError`` straight from
    that ``ok`` (mcp_server._tools_call), so a client was told "path not found" was a success.
    """

    MISSING = str(Path(tempfile.gettempdir()) / "virturoid_no_such_project_dir_9f3a")

    def test_a_bad_path_reports_failure_on_the_envelope(self):
        if "ingest_project" not in {s["name"] for s in tool_specs()}:
            self.skipTest("ingest_project is not registered in this environment")
        r = call_tool("ingest_project", {"path": self.MISSING})
        self.assertFalse(r["ok"], f"a call that could not find its input must not report ok=True: {r}")
        self.assertIn("path not found", r["error"])
        self.assertIn("error", r["result"], "the failing payload is still handed back, not thrown away")

    def test_an_unknown_robot_reports_failure_on_the_envelope(self):
        r = call_tool("get_robot", {"robot_id": "robot_does_not_exist_9f3a"})
        self.assertFalse(r["ok"], r)
        self.assertIn("no robot", r["error"])

    def test_a_bad_verdict_is_still_a_successful_call(self):
        """The other half of the contract: ``ok`` is about the CALL, not the answer. An empty memory is a real
        answer to a real question, so it stays ok=True — the flip must not swallow honest negative results."""
        with tempfile.TemporaryDirectory() as tmp:
            r = call_tool("recall_knowledge", {"prompt": "an arm", "memory_dir": tmp})
            self.assertTrue(r["ok"], r)
            self.assertIsNone(r["result"].get("error"))


if __name__ == "__main__":
    unittest.main()
