"""Phase 3: the MCP server (scripts/virturoid_mcp.py) exposes the agent tools over MCP stdio.

Tests the pure JSON-RPC handler (no client/subprocess needed): initialize, tools/list, tools/call, and
notification/unknown-method handling. The script is stdlib-only (no `mcp` SDK), so this always runs.
"""

import importlib.util
import unittest
from pathlib import Path

_MCP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "virturoid_mcp.py"
_spec = importlib.util.spec_from_file_location("virturoid_mcp", _MCP_PATH)
mcp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcp)


class VirturoidMcpTests(unittest.TestCase):
    def test_initialize_advertises_the_server(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(resp["result"]["serverInfo"]["name"], "virturoid")
        self.assertIn("tools", resp["result"]["capabilities"])

    def test_tools_list_maps_the_registry(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        self.assertLessEqual({"describe_robot", "recall_knowledge", "build_robot"}, names)
        for t in tools:                                   # MCP shape: name + description + inputSchema
            self.assertEqual(t["inputSchema"]["type"], "object")

    def test_tools_call_dispatches_to_the_registry(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "list_tools", "arguments": {}}})
        self.assertFalse(resp["result"]["isError"])
        self.assertEqual(resp["result"]["content"][0]["type"], "text")

    def test_tools_call_reports_tool_errors_without_crashing(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                   "params": {"name": "does_not_exist", "arguments": {}}})
        self.assertTrue(resp["result"]["isError"])       # structured tool error surfaced, not a crash

    def test_notification_gets_no_response(self):
        self.assertIsNone(mcp.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}))

    def test_unknown_method_is_a_jsonrpc_error(self):
        resp = mcp.handle_request({"jsonrpc": "2.0", "id": 5, "method": "bogus/method"})
        self.assertEqual(resp["error"]["code"], -32601)


if __name__ == "__main__":
    unittest.main()
