"""Virturoid MCP server — expose the agent tool registry over MCP stdio (stdlib-only, no `mcp` SDK needed).

Phase 3 of the robotics-native-AI plan (``docs/robotics_native_ai_plan.md``): the doorway *through our helper*.
A thin wrapper over ``services.agent_tools`` (``tool_specs`` + ``call_tool``) that speaks MCP's newline-delimited
JSON-RPC 2.0 over stdin/stdout, so an external agent — Claude Desktop, any MCP client, a robotics-tuned model —
drives the whole prompt -> body -> sim -> train -> export -> memory chain (describe_robot, recall_knowledge,
nearest_bodies, diagnose_body, build_robot, ...). Registering it (Claude Desktop config, say):

    { "mcpServers": { "virturoid": { "command": "python", "args": ["scripts/virturoid_mcp.py"] } } }

``handle_request`` is a pure function (dict -> dict|None) so the protocol is unit-testable without a client.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from virturoid.services.agent_tools import call_tool, tool_specs   # noqa: E402

_PROTOCOL_VERSION = "2024-11-05"


def _mcp_tools() -> list[dict]:
    """The registry mapped to MCP's tool shape (``inputSchema`` is our JSON-Schema ``parameters``)."""
    return [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]}
            for t in tool_specs()]


def handle_request(req: dict) -> dict | None:
    """Handle one JSON-RPC request; return the response dict, or ``None`` for notifications (no reply)."""
    mid = req.get("id")
    method = req.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "virturoid", "version": "0.1.0"}}}
    if method in ("notifications/initialized", "initialized"):
        return None                                    # notification — no response
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": _mcp_tools()}}
    if method == "tools/call":
        params = req.get("params") or {}
        res = call_tool(params.get("name"), params.get("arguments") or {})
        payload = res.get("result") if res.get("ok") else {"error": res.get("error")}
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": json.dumps(payload, default=str)}],
            "isError": not res.get("ok")}}
    if mid is None:
        return None                                    # unknown notification — ignore
    return {"jsonrpc": "2.0", "id": mid,
            "error": {"code": -32601, "message": f"method not found: {method}"}}


def main() -> int:
    """Newline-delimited JSON-RPC loop over stdio (the MCP stdio transport)."""
    os.environ.setdefault("VIRTUROID_BACKEND", "off")  # tools stay offline-safe unless the host enables a backend
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle_request(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
