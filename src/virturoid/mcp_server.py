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


# G-E: workflow PROMPTS. A server teaches multi-step use through prompts; the reasoning runs on the CLIENT's
# tokens (Claude Code slash-commands /mcp__virturoid__*, Cursor). Codex ignores prompts -> `instructions` (above)
# carries the same contract for the floor client. Each returns a user message the client LLM then executes.
_PROMPTS = {
    "design_robot_workflow": {
        "description": "The full design loop: author -> simulate -> read the HONEST verdict -> localized-edit -> "
                       "repeat until a credible walk -> export. Use for any 'build me a robot that ...' goal.",
        "arguments": [{"name": "goal", "description": "what the robot must be/do", "required": True}],
        "text": (
            "Goal: {goal}\n\nDrive the Virturoid substrate (you are the designer; it never spends its own LLM "
            "tokens — check with llm_spend):\n"
            "1. get_design_schema to learn the anatomy-graph language.\n"
            "2. submit_design {{graph}} to author + hold the robot (or create_robot {{prompt}} for a quick start).\n"
            "3. get_robot to read class + discovered appendages; render_view to SEE it.\n"
            "4. verify_robot {{mode:'quick'}} while iterating — read survived + cadence + FORWARD DISPLACEMENT.\n"
            "5. If not a credible walk, edit_robot with LOCALIZED ops (e.g. taller = scale_group legs length 1.2; "
            "op:'list' for the catalog, op:'undo' to revert) — NEVER regenerate. Go to 4.\n"
            "6. verify_robot {{mode:'full'}} for the definitive verdict + GIF, then export_held.\n"
            "HONESTY: never claim a walk without verify_robot's traces (forward displacement, not abs-masked).")},
    "diagnose_gait": {
        "description": "Read a held robot's honest gait verdict and map the failure mode to the localized edit "
                       "that fixes it (no regeneration).",
        "arguments": [{"name": "robot_id", "description": "the held robot", "required": True}],
        "text": (
            "Diagnose robot_id={robot_id}:\n"
            "1. verify_robot {{robot_id:'{robot_id}', mode:'full'}} — read verdict, survived, cadence, support_frac, "
            "height_ratio, forward_m.\n"
            "2. Map: falls over (upright low) -> widen stance (scale_group legs girth, or set stance out); no "
            "propulsion (cadence low, forward~0) -> lengthen/strengthen legs (scale_group legs length); topples "
            "forward (height_ratio drops) -> lower the body (set_height down).\n"
            "3. Apply ONE edit_robot op, re-verify. Change one thing at a time; keep what improves forward_m.")},
    "scene_workflow": {
        "description": "Author or re-theme a task scene (e.g. 'house instead of warehouse') keeping the task/robot.",
        "arguments": [{"name": "task", "description": "the task, e.g. navigation", "required": False},
                      {"name": "theme", "description": "warehouse|house|kitchen|lab|yard", "required": False}],
        "text": (
            "Scene for task='{task}', theme='{theme}':\n"
            "1. create_scene {{task:'{task}', theme:'{theme}'}} to build + hold it.\n"
            "2. To re-theme later (keeping task/robot/layout): edit_scene {{ops:[{{op:'swap_theme', "
            "args:{{theme:'house'}}}}]}}.\n"
            "3. submit_scene_spec {{objects}} instead if you want to author exact object placements yourself.")},
}


def _prompts_list() -> dict:
    return {"prompts": [{"name": n, "description": p["description"], "arguments": p["arguments"]}
                        for n, p in _PROMPTS.items()]}


def _prompts_get(params: dict) -> dict:
    name = params.get("name")
    p = _PROMPTS.get(name)
    if p is None:
        raise ValueError(f"no prompt '{name}'")
    args = params.get("arguments") or {}
    filled = {a["name"]: args.get(a["name"], f"<{a['name']}>") for a in p["arguments"]}
    return {"description": p["description"],
            "messages": [{"role": "user", "content": {"type": "text", "text": p["text"].format(**filled)}}]}


def _resources_list() -> dict:
    items = []
    # G-E: live STATE resources — clients that auto-attach resources ground themselves in what the agent holds.
    # Bounded (Claude Code caps responses ~25k tokens): the 40 most-recent robot sessions, not every one ever held.
    try:
        from virturoid.services import session_state
        for r in session_state.list_robots()[-40:]:
            rid = r["robot_id"]
            items.append({"uri": f"virturoid://robot/{rid}/summary", "name": f"{rid} summary",
                          "description": f"{r.get('robot_class') or '?'} — {r.get('label')}", "mimeType": "application/json"})
    except Exception:  # noqa: BLE001 - resources are progressive enhancement
        pass
    render_dir = Path("build/agent_renders")
    if render_dir.exists():
        for p in sorted(render_dir.glob("*"))[:100]:
            if p.suffix.lower() in (".png", ".gif"):
                items.append({"uri": f"file://{p.resolve()}", "name": p.name,
                              "mimeType": "image/gif" if p.suffix == ".gif" else "image/png"})
    return {"resources": items}


# H1 (plan_v5): the ONLY filesystem roots resources/read may serve. Canonicalize + confine, so a
# `file://../../etc/passwd` or a symlink can't escape — the Equixly 22%-of-servers arbitrary-read class
# (Filesystem-MCP CVE-2025-53109/53110 remedy: resolve then allowlist). Artifacts we produce live here.
_ALLOWED_READ_ROOTS = ("build/agent_renders", "build/agent_exports", "build/sessions")


def _read_allowed(path: Path) -> bool:
    try:
        target = path.resolve()
    except (OSError, RuntimeError):
        return False
    for root in _ALLOWED_READ_ROOTS:
        r = Path(root).resolve()
        if r == target or r in target.parents:
            return True
    return False


def _resources_read(params: dict) -> dict:
    import base64
    uri = str(params.get("uri", ""))
    if uri.startswith("virturoid://robot/"):                    # live state: return the compact summary JSON
        rid = uri[len("virturoid://robot/"):].split("/")[0]
        from virturoid.services.agent_tools import call_tool
        payload = call_tool("get_robot", {"robot_id": rid}).get("result", {})
        return {"contents": [{"uri": uri, "mimeType": "application/json", "text": json.dumps(payload, default=str)}]}
    path = Path(uri.replace("file://", ""))
    if not _read_allowed(path):                                 # H1: confine to our artifact roots
        raise PermissionError(f"resource outside the allowed roots {_ALLOWED_READ_ROOTS}: {uri}")
    if not path.exists() or path.is_dir():
        raise FileNotFoundError(f"no resource {uri}")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    mime = "image/gif" if path.suffix == ".gif" else "image/png"
    return {"contents": [{"uri": uri, "mimeType": mime, "blob": data}]}


def _handle(method: str, params: dict):
    if method == "initialize":
        return {"protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}, "resources": {"listChanged": False},
                                 "prompts": {"listChanged": False}},
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
                    "Tasks: run_task {robot_id, goal} gives the robot a goal in plain language — it plans a "
                    "skill sequence, VERIFIES it against the morphology (honest infeasible on a mismatch), runs "
                    "the real skills, and scores it (list_skills for the vocabulary; submit_task to author the "
                    "sequence yourself). Long runs: train_held returns a job_id -> poll get_job. export_held "
                    "writes real MJCF/CAD.\n"
                    "INPUT & TRAINING (folded — call by name, not in the menu): interpret_prompt (provenance for "
                    "a build prompt — what was stated vs inferred), inspect_project_bundle {path} (classify a "
                    "dropped enterprise folder/zip into a project graph), import_robot_model {path} (a .urdf/.mjcf "
                    "faithful + inferred-RobotGene import report with fixable warnings), plan_training {task} (the "
                    "three-phase trainer ladder), check_perception_leakage (reject a policy that cheats with "
                    "privileged sim truth), amplify_demonstrations {prompt} (one walking body -> many "
                    "physics-validated gait demos), data_dividends (the flywheel 'what did we improve?' ledger).\n"
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
    if method == "prompts/list":
        return _prompts_list()
    if method == "prompts/get":
        return _prompts_get(params)
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
