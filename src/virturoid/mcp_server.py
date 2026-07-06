"""Virturoid MCP server (docs/ai_native_plan.md P2) — expose the ENTIRE platform to external AI agents (Claude
Code/Desktop, any MCP client) as tools, over the same ``agent_tools`` registry the in-app assistant uses (one
surface to harden). Stdlib-only JSON-RPC 2.0 over stdio (newline-delimited), no SDK dependency, so it runs as:

    claude mcp add virturoid -- python -m virturoid.mcp_server

Implements the MCP essentials: ``initialize``, ``tools/list`` (from ``tool_specs()``), ``tools/call`` (to
``call_tool()`` -> structured content), ``resources/list``/``resources/read`` (the renders/GIFs agents produce,
so they can SEE what they built — the Blender-MCP lesson). Long jobs use the tool-side handle pattern
(``start_training`` returns a job_id; ``get_job`` polls). Errors are returned as JSON-RPC errors, never crashes.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "virturoid", "version": "0.1.0"}


def _tools_list() -> dict:
    from virturoid.services.agent_tools import tool_specs
    # G-G: advertise the CONSOLIDATED <=15-tool workflow view (Cursor caps ~40 tools across servers and silently
    # drops the rest; a lean menu keeps every client — incl. Codex/Cursor — working). Every other registry tool
    # stays callable by name via tools/call. MCP uses `inputSchema`; our registry stores it under `parameters`.
    return {"tools": [{"name": t["name"], "description": t["description"], "inputSchema": t["parameters"]}
                      for t in tool_specs(view="mcp")]}


def _tools_call(params: dict) -> dict:
    from virturoid.services.agent_tools import call_tool
    name = params.get("name")
    res = call_tool(name, params.get("arguments") or {})
    ok = bool(res.get("ok", True))
    payload = res.get("result", res)
    content = [{"type": "text", "text": json.dumps(payload, default=str)}]
    # surface any render/GIF the tool produced as a resource link so the agent can fetch + SEE it
    arts = (payload or {}).get("artifacts") if isinstance(payload, dict) else None
    for a in (arts or []):
        content.append({"type": "resource_link", "uri": f"file://{Path(a).resolve()}", "name": Path(a).name,
                        "mimeType": "image/gif" if str(a).endswith(".gif") else "image/png"})
    return {"content": content, "isError": not ok, "structuredContent": payload if isinstance(payload, dict) else {}}


def _resources_list() -> dict:
    render_dir = Path("build/agent_renders")
    items = []
    if render_dir.exists():
        for p in sorted(render_dir.glob("*"))[:100]:
            if p.suffix.lower() in (".png", ".gif"):
                items.append({"uri": f"file://{p.resolve()}", "name": p.name,
                              "mimeType": "image/gif" if p.suffix == ".gif" else "image/png"})
    return {"resources": items}


def _resources_read(params: dict) -> dict:
    import base64
    uri = str(params.get("uri", "")); path = Path(uri.replace("file://", ""))
    if not path.exists():
        raise FileNotFoundError(f"no resource {uri}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    return {"contents": [{"uri": uri, "mimeType": mime, "blob": data}]}


def _handle(method: str, params: dict):
    if method == "initialize":
        return {"protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False}},
                "serverInfo": SERVER_INFO,
                "instructions": (
                    "Virturoid is a grounded text-to-robot substrate: YOU (the connected agent) are the "
                    "designer, engineer, and assistant; it compiles, simulates, gates, and tells the truth. "
                    "It never spends its own LLM tokens — verify that any time with llm_spend.\n"
                    "LOOP: create_robot (from a prompt) OR get_design_schema -> submit_design (author your own "
                    "anatomy graph) to hold a robot_id. Then get_robot to inspect, edit_robot for LOCALIZED "
                    "edits (op:'list' for the catalog, op:'undo' to revert; e.g. taller = scale_group legs "
                    "length 1.2 — never regenerate), render_view to SEE it, verify_robot (mode:'quick' while "
                    "iterating, 'full' for the definitive verdict+GIF) and evaluate_held for the task score. "
                    "Scenes: create_scene + edit_scene (e.g. 'house instead of warehouse' keeps the task/robot). "
                    "Long runs: train_held returns a job_id -> poll get_job. export_held writes real MJCF/CAD.\n"
                    "HONESTY: never claim a walk without verify_robot's traces (survived + cadence + forward "
                    "displacement). Every edit is localized to the held gene — you keep the robot across turns.")}
    if method == "tools/list":
        return _tools_list()
    if method == "tools/call":
        return _tools_call(params)
    if method == "resources/list":
        return _resources_list()
    if method == "resources/read":
        return _resources_read(params)
    if method in ("ping",):
        return {}
    raise ValueError(f"method not found: {method}")


def serve(stdin=None, stdout=None) -> None:
    """Run the newline-delimited JSON-RPC loop until stdin closes. Notifications (no ``id``) get no response."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        mid = msg.get("id")
        method = msg.get("method", "")
        if method.startswith("notifications/") or mid is None:
            continue                                            # notification: no reply
        try:
            result = _handle(method, msg.get("params") or {})
            resp = {"jsonrpc": "2.0", "id": mid, "result": result}
        except Exception as exc:  # noqa: BLE001 - JSON-RPC error object, never crash the server
            resp = {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}",
                              "data": traceback.format_exc()[-500:]}}
        stdout.write(json.dumps(resp) + "\n"); stdout.flush()


if __name__ == "__main__":
    serve()
