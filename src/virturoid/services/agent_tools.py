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
    """The moat measured: MAP-Elites coverage/QD + provenance compounding + the P1-P3 brain layers (transfer
    ledger, gated-metric state, episodes, per-kind compounding deltas) — every number traceable to a table."""
    from virturoid.services.design_brain import design_brain_summary
    from virturoid.services.flywheel_status import moat_status
    memdir = args.get("memory_dir") or "build/memory"
    out = design_brain_summary(memdir)
    try:
        out["brain"] = moat_status(memdir).get("brain")
    except Exception:  # noqa: BLE001 - brain layers are additive; never break the panel
        out["brain"] = None
    return out


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
try:
    from virturoid.services.input_training_tools import INPUT_TRAINING_TOOLS
    TOOLS.update(INPUT_TRAINING_TOOLS)                       # input-ingestion + training-improvement plan tools
except Exception:  # noqa: BLE001
    pass
try:
    from virturoid.services.episode_moments import EPISODE_TOOLS
    TOOLS.update(EPISODE_TOOLS)                              # Thread C: semantic moment search over episodes
except Exception:  # noqa: BLE001
    pass
try:
    from virturoid.services.shape_flywheel import SHAPE_TOOLS
    TOOLS.update(SHAPE_TOOLS)                                # Thesis A: shape-word corpus as runtime grounding
except Exception:  # noqa: BLE001
    pass
try:
    from virturoid.services.gait_hints import GAIT_HINT_TOOLS
    TOOLS.update(GAIT_HINT_TOOLS)                            # flywheel = auto-mined HINTS + per-body adaptation
except Exception:  # noqa: BLE001
    pass

try:
    from virturoid.services.reward_loop import REWARD_LOOP_TOOLS
    TOOLS.update(REWARD_LOOP_TOOLS)                          # R1: LLM-authored-reward training from an NLP task
except Exception:  # noqa: BLE001
    pass


# ---------------------------------------------------------------- the ADVERTISED schema must be TRUE
#
# ``parameters`` is not documentation, it is the CONTRACT a strict MCP client validates ``tools/call`` arguments
# against — an argument the schema does not declare is REJECTED before it ever reaches the handler. So a
# parameter a handler honours but the schema hides is not a doc nit; it is a call the customer's agent CANNOT
# MAKE. Measured at HEAD, 21 of 67 registered tools hid 39 accepted keys between them, and the worst was the one
# we instruct agents to use most: ``verify_robot`` advertised ``{robot_id}`` alone while accepting ``mode``,
# ``steps`` and ``scene_id`` — and both the MCP ``initialize`` instructions and the ``design_robot_workflow``
# prompt told agents to send ``mode:'quick'``. A strict client rejected the exact call the server taught it.
#
# The repair DERIVES instead of restating: ``_accepted_params`` reads a handler's own body for the keys it pulls
# out of its args dict, and ``_publish_accepted_params`` publishes every one the schema is missing. ``_PARAM_DOCS``
# adds the enum/units/meaning an agent needs where the derived shape is too thin; ``_NOT_ADVERTISED`` is the one
# place a key can be kept off the contract, and it must say why. Because the source of truth is the handler
# itself, a parameter added tomorrow cannot go quietly unadvertised — ``tests/test_tool_registration.py`` fails.
#
# Applied HERE, in the aggregator, rather than in each sub-registry: this module is the single surface both
# ``call_tool`` and the MCP server read, so one pass covers ai_native/design/input-training/reward tools alike
# and there is exactly one schema per tool in the process (the specs are shared dicts, corrected in place).

_COERCE = {"int": "integer", "float": "number", "bool": "boolean", "str": "string"}
_LITERAL_TYPE = {bool: "boolean", int: "integer", float: "number", str: "string"}
_ACCEPTED_CACHE: dict = {}


def _accepted_params(handler) -> dict[str, dict]:
    """Every key ``handler`` pulls out of its args dict, with the type/default it is read with.

    Reads the handler's own AST for ``args["k"]``, ``args.get("k")`` and ``args.pop("k")`` (including the
    ``(args or {}).get("k")`` guard idiom), and infers the JSON-Schema type from the coercion wrapped around the
    read (``int(...)``/``float(...)``/``bool(...)``/``str(...)``) or from a literal default. Deliberately
    one-sided: a key forwarded wholesale to another function (``probe(gene, args)``) is not detected, so this
    UNDER-reports rather than inventing parameters no handler honours.

    Memoized per process. ``inspect.getsource`` reads the file by LINE NUMBER, so a source file edited while
    this process is alive makes a later call disagree with the one that built the schema; the first read — at
    import, the same moment the schema is published — is the one that counts."""
    import ast
    import inspect
    import textwrap
    try:
        return _ACCEPTED_CACHE[handler]
    except (KeyError, TypeError):
        pass
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(handler)))
    except (OSError, TypeError, SyntaxError, IndentationError, ValueError):
        return {}
    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or not fn.args.args:
        return {}
    argname = fn.args.args[0].arg
    parent: dict[int, ast.AST] = {}
    for node in ast.walk(fn):
        for child in ast.iter_child_nodes(node):
            parent[id(child)] = node

    def is_args(node) -> bool:
        if isinstance(node, ast.Name):
            return node.id == argname
        return (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
                and bool(node.values) and is_args(node.values[0]))

    def coerced_type(node):
        """Walk out through ``.lower()``/``or``/parens to the ``int(...)``-style call wrapping the read."""
        cur = parent.get(id(node))
        for _ in range(3):
            if isinstance(cur, ast.Call) and isinstance(cur.func, ast.Name) and cur.func.id in _COERCE:
                return _COERCE[cur.func.id]
            if isinstance(cur, (ast.Call, ast.BoolOp, ast.Attribute)):
                cur = parent.get(id(cur))
                continue
            return None
        return None

    found: dict[str, dict] = {}

    def record(key: str, node, default=None) -> None:
        prop = found.setdefault(key, {})
        kind = coerced_type(node)
        if kind:
            prop.setdefault("type", kind)
        if isinstance(default, ast.Constant) and type(default.value) in _LITERAL_TYPE:
            prop.setdefault("type", _LITERAL_TYPE[type(default.value)])
            prop.setdefault("default", default.value)

    for node in ast.walk(fn):
        if (isinstance(node, ast.Subscript) and is_args(node.value)
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            record(node.slice.value, node)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("get", "pop") and is_args(node.func.value) and node.args
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            record(node.args[0].value, node, node.args[1] if len(node.args) > 1 else None)
    try:
        _ACCEPTED_CACHE[handler] = found
    except TypeError:                                          # unhashable handler; recompute next time
        pass
    return found


# Keys a handler reads that are deliberately NOT part of the public contract. Empty on purpose: every key the
# audit found was a real, useful lever an agent was simply unable to pull. An entry here must carry its reason,
# because it is the one way to keep something off the advertised schema without the guard failing.
_NOT_ADVERTISED: dict[str, frozenset[str]] = {}

# Where the derived ``{type, default}`` is too thin for an agent to use the parameter correctly. Only consulted
# for keys the schema was MISSING; a stale entry (naming a key the handler no longer reads) fails the guard.
_PARAM_DOCS: dict[str, dict[str, dict]] = {
    "verify_robot": {
        "mode": {"type": "string", "enum": ["quick", "full"], "default": "full",
                 "description": "'quick' = the fast iterate check (legged: 800 steps, no GIF, reports "
                                "settled:false); 'full' = the definitive verdict over the settling horizon, "
                                "with a GIF for a legged body. Iterate on quick, claim on full."},
        "steps": {"type": "integer", "description": "override the simulated horizon; default is chosen per "
                                                    "robot kind and mode (legged full = the settling horizon)"},
        "scene_id": {"type": "string", "description": "run the verdict inside a HELD scene (create_scene/"
                                                      "submit_scene_spec). Matters for drive/reach; a legged "
                                                      "gait verdict stays obstacle-free and says so."},
    },
    "edit_robot": {
        "gate_non_regression": {"type": "boolean", "default": True,
                                "description": "auto-revert the edit if deterministic design findings get worse"},
        "gate_connectivity": {"type": "boolean", "default": True,
                              "description": "reject an edit that newly DETACHES a link from the body"},
    },
    "export_held": {
        "task": {"type": "string", "description": "task the BOM/spec should be sized for (default: the robot's "
                                                  "own prompt)"},
        "out_dir": {"type": "string", "description": "output dir, confined under build/ (default agent_exports)"},
        "certificate_margins": {"type": "boolean", "default": True,
                                "description": "run the margin checks behind the 'certificate' format"},
        "dr_sweep": {"type": "boolean", "default": False,
                     "description": "add a frozen-policy domain-randomization sweep with a Clopper-Pearson bound"},
        "dr_draws": {"type": "integer", "default": 12, "description": "draws in that DR sweep"},
    },
    "export_isaac": {"out_dir": {"type": "string",
                                 "description": "output dir, confined under build/ (default agent_exports)"}},
    "train_held": {
        "iters": {"type": "integer", "default": 200, "description": "PPO iterations when mode='gpu_rl'"},
        "build_root": {"type": "string", "description": "job workspace, confined under build/"},
    },
    "start_training": {
        "target": {"type": "number", "default": 0.8, "description": "target success rate to build toward"},
        "build_root": {"type": "string", "description": "job workspace, confined under build/"},
    },
    "create_robot": {"tune_gait": {"type": "boolean", "default": True,
                                   "description": "fit a per-body gait after composing (legged only); false is "
                                                  "much faster but the body arrives untuned"}},
    "assert_design": {"list": {"type": "boolean",
                               "description": "shortcut for kind:'list' — return the assertion vocabulary"}},
    "ingest_project": {
        "path": {"type": "string", "description": "alias for project_path"},
        "nlp": {"type": "string", "description": "alias for description (the plain-language spec to parse)"},
    },
    "submit_scene_spec": {
        "name": {"type": "string", "default": "agent_scene", "description": "name for the held scene"},
        "theme": {"type": "string", "description": "theme tag recorded on the scene (default 'custom')"},
    },
    "submit_task": {
        "name": {"type": "string", "default": "agent_task", "description": "name for the authored task"},
        "goal_text": {"type": "string", "description": "plain-language restatement of the goal, kept with the task"},
    },
    "pin_part": {"name": {"type": "string", "description": "alias for part"}},
    "part_specs": {"name": {"type": "string", "description": "alias for part"}},
    "bank_shape_word": {"program": {"type": "object", "description": "alias for shape_program"}},
    "ask_episode": {"prompt": {"type": "string", "description": "alias for question"}},
    "train_reward": {
        "steps": {"type": "integer", "default": 800, "description": "physics steps per candidate rollout"},
        "seed": {"type": "integer", "default": 0, "description": "search seed (set it to reproduce a run)"},
    },
    "learn_gait": {
        "workers": {"type": "integer", "default": 1, "description": "parallel rollout workers"},
        "db_path": {"type": "string", "description": "explicit memory DB path (default the shared memory dir)"},
    },
    "plan_training": {"scene_set_refs": {"type": "array", "items": {"type": "string"},
                                         "description": "scene sets the ladder should plan against"}},
    "check_perception_leakage": {
        "id": {"type": "string", "default": "contract_adhoc", "description": "id for this observation contract"},
        "perception_rung": {"type": "string",
                            "description": "perception rung of the contract (default rung 0, privileged)"},
        "task_graph_id": {"type": "string", "default": "task"},
        "scene_set_id": {"type": "string", "default": "scenes"},
        "robot_genome_id": {"type": "string", "default": "body"},
    },
}


def _publish_accepted_params() -> dict[str, list[str]]:
    """Add every handler-accepted key the tool's schema was hiding. Returns ``{tool: [keys added]}``."""
    added: dict[str, list[str]] = {}
    for name, spec in TOOLS.items():
        params = spec.get("parameters")
        if not isinstance(params, dict) or params.get("type") != "object":
            continue
        props = params.setdefault("properties", {})
        hidden = _NOT_ADVERTISED.get(name, frozenset())
        docs = _PARAM_DOCS.get(name, {})
        for key, derived in sorted(_accepted_params(spec.get("handler")).items()):
            if key in props or key in hidden:
                continue
            props[key] = dict(docs.get(key) or derived)
            added.setdefault(name, []).append(key)
    return added


def _derive_registry_backed_docs() -> None:
    """Where a description RESTATES a registry, derive it instead — restating is how it drifts.

    Two shipped instances, both found by diffing the prose against the registry behind it:
      * ``edit_ops`` named 4 of the 8 ``edit_operators.OPERATORS``, and
      * ``export_held`` named 8 of the 9 ``_EXPORT_FORMATS`` — the missing one being ``certificate``, the
        verification certificate, which is the export an honest product most wants an agent to ask for.
    """
    _derive_operator_catalog()
    _derive_export_formats()


def _derive_export_formats() -> None:
    """``export_held``'s format list, from ``agent_design_tools._EXPORT_FORMATS``, as prose AND as an enum."""
    spec = TOOLS.get("export_held")
    if spec is None:
        return
    try:
        from virturoid.services.agent_design_tools import _EXPORT_FORMATS
    except Exception:  # noqa: BLE001
        return
    fmts = list(_EXPORT_FORMATS)
    missing = [f for f in fmts if f not in spec["description"]]
    if missing:
        spec["description"] = spec["description"].rstrip() + " Also accepts: " + ", ".join(missing) + "."
    props = spec["parameters"]["properties"]
    props["formats"] = {**(props.get("formats") or {}), "type": "array",
                        "items": {"type": "string", "enum": fmts},
                        "description": f"any of {', '.join(fmts)}; omit for all {len(fmts)}"}


def _derive_operator_catalog() -> None:
    """``edit_ops``/``edit_robot`` must name EVERY operator ``edit_operators.OPERATORS`` can apply.

    Restating the list in prose let it drift: the shipped description named 4 of 8 (scale_group/set_height/
    set_material/set_leg_count), so scale_robot, set_payload, add_limb and adopt_walkable_template — four real
    capabilities, one of them the ONLY way to grow a new limb — were invisible to every agent reading
    tools/list. Derived from the registry it cannot drift again, and the same names go into ``ops[].op`` as an
    enum so a client that validates schemas discovers them without reading prose at all."""
    try:
        from virturoid.services.edit_operators import OPERATORS
    except Exception:  # noqa: BLE001 - description enrichment is never worth breaking the registry for
        return
    ops = sorted(OPERATORS)
    if "edit_ops" in TOOLS:
        TOOLS["edit_ops"]["description"] = (
            f"Discover the typed LOCALIZED edit operators and their args — all {len(ops)} of them: "
            + " | ".join(ops) + ". Feed one to edit_robot as ops:[{op, args}].")
    spec = TOOLS.get("edit_robot")
    if spec:
        props = spec["parameters"]["properties"]
        # `{op:'undo'}` / `{op:'list'}` are accepted inside `ops` too (the single-verb shortcut), so the enum
        # a strict client validates against has to admit them or it would reject a documented call.
        verbs = sorted(set(ops) | {"undo", "list"})
        prev = props.get("ops") or {}
        props["ops"] = {**prev, "type": "array", "description": prev.get(
            "description", "list of {op, args} localized edits"),
            "items": {"type": "object", "required": ["op"],
                      "properties": {"op": {"type": "string", "enum": verbs},
                                     "args": {"type": "object", "description": "operator args; see edit_ops"}}}}


SCHEMA_REPAIRS: dict[str, list[str]] = _publish_accepted_params()
_derive_registry_backed_docs()


# The CONSOLIDATED MCP surface (agent_first_plan.md G-G). Research: Cursor caps ~40 active tools ACROSS all
# servers and SILENTLY drops the rest; Codex/weaker clients degrade with a big flat menu. So the MCP server
# advertises this small, workflow-shaped view (<=15) instead of all ~30 registry tools. Every folded tool is
# still callable by name (call_tool dispatches the full registry) — we just don't crowd the menu with it:
#   describe_robot/diagnose_body -> get_robot ; simulate_gait -> verify_robot(mode) ; undo_robot+edit_ops ->
#   edit_robot(op) ; search_memory/nearest_bodies -> recall_knowledge ; list_tools/capabilities/design_brain/
#   build_robot/evaluate_robot/design_search/start_training/submit_scene_spec -> covered by the loop tools +
#   the server `instructions`. Ordered by the canonical loop so the menu reads as a workflow.
MCP_TOOL_VIEW: tuple[str, ...] = (
    "get_design_schema", "submit_design", "critique_design",  # author / see-measure-critique
    "ingest_project",                                          # INGEST an existing robot folder (gateway, below)
    "get_robot", "edit_robot", "render_view",                  # inspect / localized-edit / see
    "verify_robot", "run_task",                                # honest verdict / any-goal task
    "create_scene",                                             # themed scene (edit_scene stays callable)
    "train_held", "get_job", "export_held",                    # train (job) / poll / export
    "recall_knowledge", "llm_spend",                           # memory recall / zero-token proof
)

# ADVANCED AUTHORING (M5, 2026-07-24 audit): the reward-loop / sensor-fusion / control-script compilers are
# callable-by-name but kept OUT of the lean core menu (which must stay within the cross-client budget). Like the
# ingestion siblings, they are advertised by name in the anchor tool's description so an MCP client can discover
# and call them without bloating tools/list.
_ADVANCED_SIBLINGS: tuple[str, ...] = ("train_reward", "generate_fusion", "generate_control_scripts")
_DESIGN_SIBLINGS: tuple[str, ...] = ("create_robot", "evaluate_held")
_SCENE_SIBLINGS: tuple[str, ...] = ("edit_scene", "submit_scene_spec")

# MEASURE / DECLARE-INTENT and DRY-RUN-THE-AMEND. These three shipped fully implemented and were registered
# NOWHERE — so `call_tool`, `tools/list` and `tools/call` all rejected them, while `get_design_schema` was
# telling the customer's own model "Check it with probe_robot rather than reasoning about frames": we advertised
# a tool that did not exist on the wire. They are registered in AI_NATIVE_TOOLS now (so tools/call dispatches
# them), and advertised here on the anchor they belong to rather than added to MCP_TOOL_VIEW, because that view
# is at its documented cross-client cap of 15 and test_agent_first asserts it. Same treatment as the ingest and
# advanced-authoring siblings: named in the tools/list payload, callable by name.
_INSPECT_SIBLINGS: tuple[str, ...] = ("probe_robot", "assert_design")
_AMEND_SIBLINGS: tuple[str, ...] = ("scope_amend",)

# INGESTION GATEWAY (P0, agentic-ingestion plan): the plan found ZERO ingestion tools in the MCP view, so a
# customer's own agent (the BYOK model) could not DISCOVER ingestion from tools/list at all. Rather than crowd
# the lean menu with every importer, ``ingest_project`` is advertised and its description names the sibling
# importers — each callable by name via call_tool (the registry dispatches all of them). This is the single
# discovery entry point for "I already have a robot / policy / BOM / dataset; bring it in."
_INGEST_SIBLINGS: tuple[str, ...] = (
    "import_robot_model", "import_bom", "import_onnx_policy", "import_dataset",
    "import_cad", "adopt_control_script", "sandbox_policy",
)


def tool_specs(view: str | None = None) -> list[dict]:
    """The agent/MCP-discoverable tool list: ``[{name, description, parameters, heavy}]``. ``view='mcp'`` returns
    only the consolidated <=15-tool MCP surface (G-G), in workflow order; default returns the full registry."""
    if view == "mcp":
        out = []
        for n in MCP_TOOL_VIEW:
            if n not in TOOLS:
                continue
            desc = TOOLS[n]["description"]
            if n == "ingest_project":                          # advertise the sibling importers by name
                avail = [s for s in _INGEST_SIBLINGS if s in TOOLS]
                desc = (desc + " Companion importers, each callable by name via a tools/call: "
                        + ", ".join(avail) + ".")
            if n == "submit_design":
                avail = [s for s in _DESIGN_SIBLINGS if s in TOOLS]
                if avail:
                    desc = desc + " Callable companions: " + ", ".join(avail) + "."
            if n == "get_robot":                               # MEASURE the model / declare intent as checks
                avail = [s for s in _INSPECT_SIBLINGS if s in TOOLS]
                if avail:
                    desc = (desc + " Callable companions: " + ", ".join(avail)
                            + " (measure the COMPILED model — reach/clearance/mass/per-joint torque margin/"
                              "swept self-collision through each joint's range; and state design intent as "
                              "assertions the harness checks).")
            if n == "edit_robot":
                avail = [s for s in _AMEND_SIBLINGS if s in TOOLS]
                if avail:
                    desc = (desc + " Callable companion: " + ", ".join(avail)
                            + " — DRY-RUN the same ops first (blast radius + what it invalidates, nothing edited).")
            if n == "create_scene":
                avail = [s for s in _SCENE_SIBLINGS if s in TOOLS]
                if avail:
                    desc = desc + " Callable companions: " + ", ".join(avail) + "."
            if n == "train_held":                              # M5: advertise the advanced authoring compilers
                adv = [s for s in _ADVANCED_SIBLINGS if s in TOOLS]
                if adv:
                    desc = (desc + " Advanced authoring companions, each callable by name via a tools/call: "
                            + ", ".join(adv) + " (reward-as-code loop / sensor-fusion config / control scripts).")
            out.append({"name": n, "description": desc, "parameters": TOOLS[n]["parameters"],
                        "heavy": TOOLS[n].get("heavy", False)})
        return out
    return [{"name": n, "description": t["description"], "parameters": t["parameters"],
             "heavy": t.get("heavy", False)} for n, t in TOOLS.items()]


def call_tool(name: str, args: dict | None = None) -> dict:
    """Dispatch a tool by name. Returns ``{ok, tool, result}`` or ``{ok: False, tool, error}`` — never raises,
    so an agent gets a structured error instead of a crash. H3: every result carries ``took_s`` so an agent can
    budget (verify_full ~14 s vs quick ~0.1 s)."""
    import time
    spec = TOOLS.get(name)
    if spec is None:
        return {"ok": False, "tool": name, "error": f"unknown tool '{name}'; call list_tools to discover"}
    t0 = time.monotonic()
    try:
        result = spec["handler"](args or {})
        if isinstance(result, dict):
            result.setdefault("took_s", round(time.monotonic() - t0, 3))
        return {"ok": True, "tool": name, "result": result}
    except KeyError as exc:
        return {"ok": False, "tool": name, "error": f"missing required argument: {exc}", "took_s": round(time.monotonic() - t0, 3)}
    except Exception as exc:  # noqa: BLE001 - agent-facing: return the error, don't crash the caller
        return {"ok": False, "tool": name, "error": f"{type(exc).__name__}: {exc}", "took_s": round(time.monotonic() - t0, 3)}


# H2 (plan_v5): confine agent-supplied write paths (out_dir/build_root) under build/ so a tool can't be
# steered to write outside the workspace. Resolve + verify containment; fall back to the default on escape.
def safe_build_path(value, default_subdir: str):
    """Return an absolute path under ``build/`` for an agent-supplied dir arg, or the default if it escapes."""
    build_root = (Path.cwd() / "build").resolve()
    default = build_root / default_subdir
    if not value:
        return default
    try:
        cand = Path(value)
        if not cand.is_absolute():
            cand = build_root / cand
        cand = cand.resolve()
    except (OSError, RuntimeError, ValueError):
        return default
    return cand if (cand == build_root or build_root in cand.parents) else default


def _slug(text: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", (text or "robot").lower()).strip("_")[:60] or "robot"
