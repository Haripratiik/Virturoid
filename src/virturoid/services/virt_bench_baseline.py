"""VIRT-Bench Arm 0 — the LITERAL "Claude + MCP" baseline (plan v3 M1, the credibility keystone).

The whole thesis is "a self-improving robot-design brain that out-performs Claude + MCP." Arms A/A+/B measure
OUR methods against OUR fixed-pipeline proxy (Arm A) — but until we run the *actual* control (an LLM agent
driving a generic robotics MCP on the same frozen tasks, scored by the SAME independent verifier), "beats
Claude+MCP" is a claim without an experiment. This module is that control.

Arm 0 is deliberately given only the PRIMITIVE MCP tools a generic MuJoCo integration would expose — inspect a
body and simulate it (``describe_robot``, ``diagnose_body``, ``evaluate_robot``, ``capabilities``,
``list_tools``). It is NOT given our differentiators: the verified ``design_search`` / ``build_robot`` harness
(the multi-step search + training loop) or the memory tools (the flywheel moat). That is the honest boundary:
whatever Arm 0 achieves is what "an LLM with a simulator but no verified search loop or memory" achieves, and
``B_solved − A0_solved`` is the measured value of everything we add on top.

The agent runs a bounded ReAct loop (inspect → reason → submit a controller); the controller it submits (a CPG
gait for locomotion, skill params for manipulation) is re-run by ``verify_submission`` at the frozen horizon.
Its ``claimed_pass`` (the agent's own belief) vs the verifier's verdict is the honesty axis (agents fabricate
success — the benchmark measures the over-claim). The LLM is injected, so this unit-tests with a mock; with no
backend configured Arm 0 honestly reports ``no_llm_backend`` (there is no "Claude+MCP" without a model)."""

from __future__ import annotations

from virturoid.services.agent_tools import TOOLS, call_tool
from virturoid.services.virt_bench import get_task, verify_submission

# The PRIMITIVE MCP surface Arm 0 is allowed: inspect + simulate only. Excludes design_search/build_robot (our
# verified harness) and search_memory/nearest_bodies/recall_knowledge/design_brain (the flywheel moat) — giving
# those to the baseline would collapse the very delta we are measuring.
_ARM0_TOOLS = ("list_tools", "capabilities", "describe_robot", "diagnose_body", "evaluate_robot")

_ARM0_STEP_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["tool", "submit"]},
        "tool": {"type": "string"},
        "args": {"type": "object"},
        "controller": {"type": "object", "description": "on submit: the controller to score"},
        "expect_pass": {"type": "boolean", "description": "on submit: does the agent believe it solves the task?"},
    },
    "required": ["action"],
    "additionalProperties": True,
}

_ARM0_SYSTEM = (
    "You are an autonomous robotics agent with an MCP tool surface over a MuJoCo simulator. Given a task, design "
    "a CONTROLLER for the provided robot body and submit it to be scored. Work step by step: each turn output ONE "
    "JSON action. To inspect or simulate, use action='tool' with a tool name + args (call list_tools to see the "
    "surface; describe_robot/diagnose_body read a body, evaluate_robot simulates it). When ready, use "
    "action='submit' with a 'controller' object and 'expect_pass'. For a LOCOMOTION task the controller is a CPG "
    "gait: {calf_phase (radians, controls step direction/timing), freq (Hz), thigh_amp, calf_amp}. For a "
    "MANIPULATION task the controller is grasp-skill params: {grip, phase_steps, approach_dz}. Submit within a few "
    "steps; you have no verified search loop or training — reason from the body + simulation to a good controller.")


def _arm0_tool_specs() -> list:
    return [{"name": n, "description": TOOLS[n]["description"], "parameters": TOOLS[n]["parameters"]}
            for n in _ARM0_TOOLS if n in TOOLS]


def _extract_cpg(controller):
    """Merge the agent's submitted gait onto the default CPG so a partial spec (e.g. just calf_phase) is valid.
    Accepts either a flat {calf_phase,...} or a nested {cpg:{...}}. None -> the default gait (== Arm A)."""
    from virturoid.services.morph_policy import CPG_DEFAULT
    if not isinstance(controller, dict):
        return dict(CPG_DEFAULT)
    src = controller.get("cpg") if isinstance(controller.get("cpg"), dict) else controller
    merged = dict(CPG_DEFAULT)
    for k in ("freq", "thigh_amp", "calf_amp", "calf_phase", "residual_scale", "leg_flip"):
        if k in src:
            merged[k] = src[k]
    return merged


def run_arm_0(task_id: str, llm=None, *, max_steps: int = 6, seed: int | None = None, auto_llm: bool = True) -> dict:
    """Run the Claude+MCP baseline on one frozen task. ``llm`` drives the ReAct loop (injected for tests); if None
    and ``auto_llm``, resolve the env backend via ``make_routed_llm`` (None -> honest ``no_llm_backend``). Returns
    the SAME verdict shape as the other arms (verified by ``verify_submission`` at the frozen horizon) plus a
    ``budget`` ledger (tool_calls, llm_calls) and the ``transcript``."""
    from virturoid.services.virt_bench_arms import _task_body, _zero_policy_with_cpg
    task = get_task(task_id)
    gene = _task_body(task)
    if gene is None:
        return {"task": task_id, "arm": "A0", "verified_pass": False, "failure_mode": "unsupported_task",
                "metrics": {}, "method": "Claude+MCP baseline", "claimed_pass": False,
                "budget": {"tool_calls": 0, "llm_calls": 0, "gpu_iters": 0}}
    if llm is None and auto_llm:
        try:
            from virturoid.services.llm_client import make_routed_llm
            llm = make_routed_llm("designer")
        except Exception:  # noqa: BLE001
            llm = None
    if llm is None:
        return {"task": task_id, "arm": "A0", "verified_pass": False, "failure_mode": "no_llm_backend",
                "metrics": {}, "method": "Claude+MCP baseline (no backend configured)", "claimed_pass": False,
                "budget": {"tool_calls": 0, "llm_calls": 0, "gpu_iters": 0}}

    tools = _arm0_tool_specs()
    transcript: list = []
    controller = None
    claimed = True
    tool_calls = llm_calls = 0
    for _step in range(max_steps):
        user = (f"Task: {task['prompt']} (family={task['family']}, id={task_id}).\n"
                f"Robot body: a {task['family']} robot with {len(gene.actuated_joints())} actuated joints.\n"
                f"Available tools:\n{[t['name'] for t in tools]}\n"
                f"Transcript so far:\n{transcript}\n"
                "Output ONE JSON action (tool or submit).")
        try:
            out = llm.complete_json(_ARM0_SYSTEM, user, _ARM0_STEP_SCHEMA)
            llm_calls += 1
        except Exception:  # noqa: BLE001 - a model error ends the loop; whatever was chosen is submitted
            break
        if (out or {}).get("action") == "submit":
            controller = out.get("controller") or None
            claimed = bool(out.get("expect_pass", True))
            break
        name = (out or {}).get("tool")
        if name not in _ARM0_TOOLS:                          # the agent may only use the primitive surface
            transcript.append({"tool": name, "error": "not available to this agent (primitive MCP only)"})
            continue
        res = call_tool(name, (out or {}).get("args") or {})
        tool_calls += 1
        transcript.append({"tool": name, "result": res.get("result") if res.get("ok") else res.get("error")})

    if task["family"] == "locomotion":
        submit_pol = _zero_policy_with_cpg(gene, _extract_cpg(controller))
    else:
        submit_pol = controller if isinstance(controller, dict) and controller else None
    res = verify_submission(task_id, gene, submit_pol, seed=seed)
    res["arm"] = "A0"
    res["method"] = f"Claude+MCP baseline ({tool_calls} tool calls, {llm_calls} llm calls)"
    res["claimed_pass"] = bool(claimed)                      # the agent's own belief; overclaim = claim − verified
    res["budget"] = {"tool_calls": tool_calls, "llm_calls": llm_calls, "gpu_iters": 0}
    res["transcript"] = transcript
    return res
