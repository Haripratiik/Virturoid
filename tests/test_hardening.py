"""V5-P1 hardening (plan_v5 W-H): resources/read is confined to our artifact roots (H1), agent-supplied
write paths can't escape build/ (H2), and every tool result carries a latency field (H3). These gate any
future HTTP transport / registry listing (OWASP MCP, Filesystem-MCP CVE-2025-53109/53110)."""
import os
import unittest
from pathlib import Path


class ResourceReadAllowlistTests(unittest.TestCase):
    """H1: file:// resources outside build/agent_{renders,exports}/sessions are refused."""

    def test_escape_paths_are_refused(self):
        from virturoid import mcp_server
        for bad in ("file:///etc/passwd", "file://../../../../etc/passwd",
                    f"file://{Path.cwd() / 'src' / 'virturoid' / 'mcp_server.py'}"):
            with self.assertRaises(PermissionError, msg=f"{bad} must be refused"):
                mcp_server._resources_read({"uri": bad})

    def test_allowed_root_passes_the_gate(self):
        # a path INSIDE an allowed root passes the allowlist (then fails on existence, not permission)
        from virturoid import mcp_server
        inside = Path("build/agent_renders/does_not_exist_xyz.png").resolve()
        self.assertTrue(mcp_server._read_allowed(inside))
        with self.assertRaises(FileNotFoundError):     # allowed root, but missing file -> not-found, NOT permission
            mcp_server._resources_read({"uri": f"file://{inside}"})

    def test_virturoid_state_uri_still_works(self):
        from virturoid import mcp_server
        out = mcp_server._resources_read({"uri": "virturoid://robot/nope/summary"})
        self.assertIn("contents", out)                 # state URIs are unaffected by the file allowlist


class PathClampTests(unittest.TestCase):
    """H2: agent-supplied out_dir/build_root are confined under build/."""

    def test_escape_falls_back_to_default(self):
        from virturoid.services.agent_tools import safe_build_path
        build = (Path.cwd() / "build").resolve()
        # absolute escape, parent-traversal, and a home path all fall back to the default under build/
        for bad in ("/etc", "../../secrets", os.path.expanduser("~"), "C:/Windows"):
            p = safe_build_path(bad, "agent_exports")
            self.assertTrue(p == build / "agent_exports" or build in p.parents or p == build,
                            f"{bad} -> {p} escaped build/")

    def test_relative_is_kept_under_build(self):
        from virturoid.services.agent_tools import safe_build_path
        build = (Path.cwd() / "build").resolve()
        p = safe_build_path("agent_exports/myrobot", "agent_exports")
        self.assertTrue(build in p.parents or p == build / "agent_exports" / "myrobot")


class LatencyTelemetryTests(unittest.TestCase):
    """H3: every dispatched tool result carries took_s."""

    def test_took_s_present(self):
        from virturoid.services.agent_tools import call_tool
        r = call_tool("get_design_schema", {})
        self.assertIn("took_s", r["result"])
        self.assertIsInstance(r["result"]["took_s"], float)

    def test_took_s_on_error_envelope(self):
        from virturoid.services.agent_tools import call_tool
        r = call_tool("get_robot", {"robot_id": "nonexistent_zzz"})
        # get_robot returns ok=True with an inner ok=False; took_s is on the result dict
        self.assertIn("took_s", r["result"])


if __name__ == "__main__":
    unittest.main()
