"""Agentic tool surface — the platform's capabilities as discoverable, schema'd, agent-callable tools.

Virturoid's Pillar 1 is *agentic* robot creation. The autonomous build loop is agentic INTERNALLY, but for
an EXTERNAL agent (an LLM, an MCP client, a user's own agent) to DRIVE the platform, its capabilities must be
exposed as TOOLS: each with a name, a JSON-schema input, a one-line description, and a structured (JSON-able)
result. This module is that surface — one ``tool_specs()`` an agent discovers, one ``call_tool(name, args)``
dispatcher — so a thin MCP server (``scripts/virturoid_mcp.py``) or the Build Assistant wraps it unchanged and
the whole "prompt -> body -> sim -> train -> evaluate -> export -> memory" value chain becomes agent-driveable.

Each tool is a thin, honest wrapper over an existing service that returns agent-friendly structured data (never
a live object). Heavy tools (build/evaluate run real MuJoCo) say so in their description; anything GPU-gated
degrades gracefully. Pure dispatch — no MuJoCo/torch imported until a tool that needs it is actually called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------- tool handlers (thin service wrappers)

def _list_tools(_args: dict) -> dict:
    """Self-describing: the tools an agent can call, with their input schemas."""
    return {"tools": tool_specs()}


def _search_memory(args: dict) -> dict:
    """Retrieve the most similar prior designs/runs from a memory dir (the design flywheel's read side)."""
    query = args["query"]                                    # required (accessed first -> clean missing-arg error)
    from virturoid.services.memory_db import MemoryDB
    memory_dir = Path(args.get("memory_dir") or "build/memory")
    db_path = memory_dir / "virturoid_memory.db"
    if not db_path.exists():
        return {"hits": [], "note": f"no memory at {db_path}"}
    with MemoryDB(db_path) as db:
        hits = db.similar_runs(query, robot_class=args.get("robot_class"),
                               limit=int(args.get("limit", 5)))
    return {"hits": [{"prompt": h.get("prompt"), "robot_class": h.get("robot_class"),
                      "task_type": h.get("task_type"), "success_rate": h.get("success_rate"),
                      "similarity": h.get("similarity")} for h in hits]}


def _design_brain(args: dict) -> dict:
    """The moat measured: MAP-Elites coverage/QD + provenance compounding for a memory dir."""
    from virturoid.services.design_brain import design_brain_summary
    return design_brain_summary(args.get("memory_dir") or "build/memory")


def _describe_robot(args: dict) -> dict:
    """Make a robot legible to an LLM: compose a body from the prompt, serialize it to tokens + a grounded
    natural-language summary (the robotics-native read side — an agent reasons about *this* body, not blind)."""
    prompt = args["prompt"]                                   # required (accessed first -> clean missing-arg error)
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.robot_serializer import describe_robot
    return describe_robot(compose_robot(prompt))


def _build_robot(args: dict) -> dict:
    """Compose + physics-build a robot from a prompt (REAL MuJoCo; slow). Returns an honest summary."""
    from virturoid.services.autonomous_build import autonomous_build
    out = Path(args.get("out_dir") or "build/agent_builds") / _slug(args["prompt"])
    memory_dir = Path(args.get("memory_dir") or "build/memory")
    r = autonomous_build(args["prompt"], out, target_success_rate=float(args.get("target", 0.8)),
                         memory_dir=memory_dir, train=bool(args.get("train", False)))
    return {"robot_class": getattr(r, "robot_class", None), "species": getattr(r, "species", None),
            "task_type": getattr(r, "task_type", None),
            "success_rate": float(getattr(r, "final_success_rate", 0.0) or 0.0),
            "succeeded": bool(getattr(r, "succeeded", False)),
            "memory_reused": bool(getattr(r, "memory_reused", False)),
            "package_dir": str(out), "decisions": [d.stage for d in getattr(r, "decisions", [])]}


def _evaluate_robot(args: dict) -> dict:
    """Compose a robot for a prompt and evaluate it on its morphology-implied task (REAL MuJoCo; slow)."""
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.task_matched_eval import evaluate_robot, robot_kind
    gene = compose_robot(args["prompt"])
    res = evaluate_robot(gene, prompt=args["prompt"])
    return {"robot_class": gene.robot_class, "kind": robot_kind(gene), "dof": len(gene.actuated_joints()),
            "task": res.get("task"), "metric": res.get("metric"), "value": res.get("value")}


def _capabilities(_args: dict) -> dict:
    """The task types + skills the platform can build/evaluate (the capability registry)."""
    try:
        from virturoid.services.capability_registry import capability_summary
        return capability_summary()
    except Exception:  # noqa: BLE001 - registry shape varies; fall back to the known task types
        return {"tasks": ["pick_place_sort", "pick_place", "stack", "shelf", "push", "transport",
                          "grasp", "locomotion", "navigation", "spray_coverage"],
                "morphologies": ["manipulator", "quadruped", "legged", "mobile_base", "humanoid", "spray"]}


# ---------------------------------------------------------------- the registry

# name -> {description, parameters (JSON schema), handler}. `parameters` follows JSON-Schema so an LLM/MCP can
# validate + fill inputs; `heavy` flags real-physics tools an agent should expect to take time.
TOOLS: dict[str, dict] = {
    "list_tools": {
        "description": "List the Virturoid tools an agent can call, with their input schemas.",
        "parameters": {"type": "object", "properties": {}},
        "handler": _list_tools, "heavy": False,
    },
    "capabilities": {
        "description": "The task types + morphologies the platform can build and evaluate.",
        "parameters": {"type": "object", "properties": {}},
        "handler": _capabilities, "heavy": False,
    },
    "search_memory": {
        "description": "Retrieve the most similar prior designs/runs from the design flywheel memory.",
        "parameters": {"type": "object", "required": ["query"], "properties": {
            "query": {"type": "string", "description": "the design/task to find prior work for"},
            "robot_class": {"type": "string"}, "limit": {"type": "integer", "default": 5},
            "memory_dir": {"type": "string", "description": "shared memory dir (default build/memory)"}}},
        "handler": _search_memory, "heavy": False,
    },
    "design_brain": {
        "description": "The moat measured: MAP-Elites niche coverage/QD-score + provenance compounding.",
        "parameters": {"type": "object", "properties": {
            "memory_dir": {"type": "string", "description": "shared memory dir (default build/memory)"}}},
        "handler": _design_brain, "heavy": False,
    },
    "describe_robot": {
        "description": "Serialize a robot (composed from a prompt) into LLM-readable tokens + a grounded "
                       "natural-language summary — the robotics-native read side so an agent reasons about "
                       "the specific body, not blind. No physics; composes a body only.",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string", "description": "what robot to describe (e.g. 'a dog that walks')"}}},
        "handler": _describe_robot, "heavy": False,
    },
    "evaluate_robot": {
        "description": "Compose a robot for a prompt and score it on its morphology-implied task (real MuJoCo).",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}}},
        "handler": _evaluate_robot, "heavy": True,
    },
    "build_robot": {
        "description": "Compose + physics-build a buildable robot (body + BOM + CAD + honest eval) from a "
                       "prompt, warm-started from memory; runs the self-improving keystone (real MuJoCo, slow).",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}, "out_dir": {"type": "string"}, "memory_dir": {"type": "string"},
            "target": {"type": "number", "default": 0.8},
            "train": {"type": "boolean", "default": False, "description": "also train a controller (slower)"}}},
        "handler": _build_robot, "heavy": True,
    },
}


def tool_specs() -> list[dict]:
    """The agent/MCP-discoverable tool list: ``[{name, description, parameters, heavy}]``."""
    return [{"name": n, "description": t["description"], "parameters": t["parameters"],
             "heavy": t.get("heavy", False)} for n, t in TOOLS.items()]


def call_tool(name: str, args: dict | None = None) -> dict:
    """Dispatch a tool by name. Returns ``{ok, tool, result}`` or ``{ok: False, tool, error}`` — never raises,
    so an agent gets a structured error instead of a crash."""
    spec = TOOLS.get(name)
    if spec is None:
        return {"ok": False, "tool": name, "error": f"unknown tool '{name}'; call list_tools to discover"}
    try:
        return {"ok": True, "tool": name, "result": spec["handler"](args or {})}
    except KeyError as exc:
        return {"ok": False, "tool": name, "error": f"missing required argument: {exc}"}
    except Exception as exc:  # noqa: BLE001 - agent-facing: return the error, don't crash the caller
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}"}


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", (text or "robot").lower()).strip("_")[:60] or "robot"
