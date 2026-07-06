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


def _diagnose_body(args: dict) -> dict:
    """Fast, physics-free structural read of a body composed from the prompt (DOF, reach, end effector,
    limb count) — the cheap "understand what this robot IS + its structural limits" tool."""
    prompt = args["prompt"]
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.robot_serializer import structural_diagnosis
    return structural_diagnosis(compose_robot(prompt))


def _nearest_bodies(args: dict) -> dict:
    """Prior bodies most morphologically similar to the prompt's body (the design flywheel's read side —
    "what have we built that's shaped like this?")."""
    prompt = args["prompt"]
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot(prompt)
    vm, closer = _open_vector_memory(args.get("memory_dir"))
    if vm is None:
        return {"bodies": [], "note": "no memory yet"}
    try:
        vm.index_species_bodies()
        return {"bodies": vm.nearest_bodies(gene, k=int(args.get("k", 5)))}
    finally:
        closer()


def _recall_knowledge(args: dict) -> dict:
    """Retrieve prior tips + lessons + skills for the prompt's body, keyed by its morphology embedding —
    the recall organ ("what worked on robots like this?"), so an agent reasons with prior experience."""
    prompt = args["prompt"]
    from virturoid.services.morphology_composer import compose_robot
    gene = compose_robot(prompt)
    vm, closer = _open_vector_memory(args.get("memory_dir"))
    if vm is None:
        return {"tips": [], "lessons": [], "skills": [], "note": "no memory yet"}
    try:
        return vm.recall_knowledge(gene, args.get("task_type"), failure_code=args.get("failure_code"),
                                   k=int(args.get("k", 3)))
    finally:
        closer()


def _open_vector_memory(memory_dir):
    """Open the vector memory for a memory dir. Returns ``(vm, closer)``; ``(None, noop)`` if absent."""
    from pathlib import Path
    memory_dir = Path(memory_dir or "build/memory")
    db_path = memory_dir / "virturoid_memory.db"
    if not db_path.exists():
        return None, (lambda: None)
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.robotics_vector_memory import RoboticsVectorMemory
    db = MemoryDB(db_path)
    return RoboticsVectorMemory(db), db.close


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


def _design_search(args: dict) -> dict:
    """Run a bounded, verified DESIGN SEARCH: compose a body from the prompt, then let the harness sweep its
    CPG gait parameters — each candidate physics-evaluated, selected by the honesty gate, diagnosed — and return
    the best config + the honest tree of what was tried. This is the multi-step search bare Claude+MCP can't do
    (no training loop, no verified selection); the CPU CPG rung is the cheapest tier (a GPU burst is the next)."""
    prompt = args["prompt"]                                    # required (accessed first -> clean missing-arg error)
    from virturoid.services.design_search import run_design_search
    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.search_adapters import cpg_grid_proposer, make_cpg_evaluate
    gene = compose_robot(prompt)
    evaluate = make_cpg_evaluate(gene, steps=int(args.get("steps", 600)))
    gates = {"forward_m": float(args.get("forward_target", 0.12)), "cadence": 3.0, "upright": 0.5}
    rep = run_design_search(propose=cpg_grid_proposer(), evaluate=evaluate, task_type="locomotion",
                            max_evals=int(args.get("max_evals", 8)), gates=gates)
    b = rep.best
    return {"robot_class": gene.robot_class, "dof": len(gene.actuated_joints()), "solved": rep.solved,
            "n_evals": rep.n_evals, "stopped_reason": rep.stopped_reason,
            "best": ({"params": b.spec.get("params"), "forward_m": round(float(b.result.get("forward", 0)), 3),
                      "cadence": round(float(b.result.get("cadence", 0)), 1),
                      "failure_mode": b.artifact["failure_mode"], "fitness": b.fitness} if b else None),
            "tree": rep.tree()}


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
    "diagnose_body": {
        "description": "Fast, physics-free structural read of a body (DOF, reach, end effector, limb count) "
                       "— understand what a robot IS and its structural limits. No physics.",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}}},
        "handler": _diagnose_body, "heavy": False,
    },
    "nearest_bodies": {
        "description": "Prior bodies most morphologically similar to the prompt's body (the design flywheel "
                       "read: 'what have we built shaped like this?').",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}, "k": {"type": "integer", "default": 5},
            "memory_dir": {"type": "string", "description": "shared memory dir (default build/memory)"}}},
        "handler": _nearest_bodies, "heavy": False,
    },
    "recall_knowledge": {
        "description": "Retrieve prior tips + lessons + reusable skills for the prompt's body, keyed by its "
                       "morphology embedding — the recall organ ('what worked on robots like this?').",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}, "task_type": {"type": "string"},
            "failure_code": {"type": "string"}, "k": {"type": "integer", "default": 3},
            "memory_dir": {"type": "string", "description": "shared memory dir (default build/memory)"}}},
        "handler": _recall_knowledge, "heavy": False,
    },
    "evaluate_robot": {
        "description": "Compose a robot for a prompt and score it on its morphology-implied task (real MuJoCo).",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}}},
        "handler": _evaluate_robot, "heavy": True,
    },
    "design_search": {
        "description": "Run a bounded VERIFIED design search over a body's gait parameters (compose -> sweep "
                       "-> physics-evaluate each -> honesty-gate select -> diagnose), returning the best config "
                       "+ an honest tree. The multi-step search a bare LLM+simulator cannot do. Real MuJoCo, slow.",
        "parameters": {"type": "object", "required": ["prompt"], "properties": {
            "prompt": {"type": "string"}, "max_evals": {"type": "integer", "default": 8},
            "steps": {"type": "integer", "default": 600},
            "forward_target": {"type": "number", "default": 0.12}}},
        "handler": _design_search, "heavy": True,
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


# AI-native stateful tools (session-held robot/scene, incremental edits, jobs) — the surface the MCP server +
# in-app assistant drive. Merged here so ``tool_specs``/``call_tool`` expose everything through one registry.
try:
    from virturoid.services.ai_native_tools import AI_NATIVE_TOOLS
    TOOLS.update(AI_NATIVE_TOOLS)
except Exception:  # noqa: BLE001 - the stateless tools stay available even if the AI-native module has an issue
    pass
try:
    from virturoid.services.agent_design_tools import AGENT_DESIGN_TOOLS
    TOOLS.update(AGENT_DESIGN_TOOLS)                          # G-A/G-C: the external agent as the designer
except Exception:  # noqa: BLE001
    pass


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
