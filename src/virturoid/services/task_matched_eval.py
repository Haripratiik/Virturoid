"""Unified task-matched evaluation: evaluate ANY composed robot on the task its morphology implies.

The capstone that ties the morphology system together — given a composed ``RobotGene``, dispatch to the
evaluator that matches the robot's class/end-effector and return one comparable result:
  manipulator + gripper  -> pick-place success
  manipulator + spray    -> surface coverage
  mobile_base            -> navigation (fraction of goals reached)
  quadruped              -> locomotion (distance walked, upright)
So "compose a robot" and "evaluate it on what it's actually for" is a single call (§11), instead of
scoring every robot on pick-place. Needs MuJoCo.
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene


def robot_kind(gene: RobotGene) -> str:
    """Classify a robot by STRUCTURE, not its (LLM-authored) class string — so a 'hexapod', 'spider',
    'frog', 'octopod', 'snake' etc. route by what they physically ARE: wheels→mobile, a free-floating
    jointed body→legged, a gripper/hand→manipulator, a spray nozzle→spray. Topology generalizes to any
    morphology the LLM designs; string-matching the four anticipated class names does not."""
    ee = gene.end_effector_type or "none"
    segs = gene.segments
    has_wheels = any(getattr(s, "shape", None) == "cylinder" and s.joint_type == "revolute" for s in segs)
    n_revolute = sum(1 for s in segs if s.joint_type == "revolute")
    if ee == "spray_nozzle":
        return "spray"
    if has_wheels:
        return "mobile"
    if ee in ("gripper", "hand"):
        return "manipulator"
    if gene.base_mount == "free" and n_revolute >= 2:    # free-floating, jointed, no wheels/gripper → legged
        return "legged"
    return "manipulator"


def evaluate_robot(gene: RobotGene, *, prompt: str = "", controller_params: dict | None = None) -> dict:
    """Evaluate ``gene`` on its morphology-implied task (dispatched by STRUCTURE — see ``robot_kind`` — so
    any LLM-designed body routes to the right controller). Returns ``{task, metric, value, detail}``.

    ``controller_params`` (e.g. what co-design produced) is used for the manipulator pick-place eval so a
    co-designed robot is scored with the SAME brain it was tuned with — otherwise the co-designed BODY is
    evaluated with default gains and mis-scores (§22 metrics should match)."""
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf

    kind = robot_kind(gene)
    ee = gene.end_effector_type

    if kind == "legged":
        from virturoid.services.locomotion_controller import run_locomotion_episode
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        r = run_locomotion_episode(mj)
        return {"task": "locomotion", "metric": "distance_m", "value": r["distance_m"], "detail": r}

    if kind == "mobile":
        from virturoid.services.navigation_controller import run_navigation_episode
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        goals = [(1.0, 0.0), (0.0, -0.9), (-0.7, 0.4), (0.7, 0.5)]
        results = [run_navigation_episode(mj, g, [], horizon=3000) for g in goals]
        reached = sum(1 for r in results if r["status"] == "reached")
        return {"task": "navigation", "metric": "goal_reach_rate", "value": round(reached / len(goals), 3),
                "detail": {"reached": reached, "goals": len(goals)}}

    if kind == "spray":
        from virturoid.services.spray_coverage import evaluate_spray_coverage
        r = evaluate_spray_coverage(gene, grid=4)
        return {"task": "spray_coverage", "metric": "coverage", "value": r["coverage"], "detail": r}

    # A gripper/hand arm on a PURE grasp task (grasp + lift, no transport to a bin/target) is scored on
    # the real friction grasp-and-lift (§13) — the honest capability the morphology implies, not the
    # idealized pin-grasp pick-place. Transport tasks (sort / place / deliver) still go to pick-place.
    p = (prompt or "").lower()
    grasp_words = ("grasp", "lift", "pick up", "pick-up", "hold ")
    transport_words = ("sort", "place", "bin", "deliver", "transport", "stack", " into ", "warehouse", "move")
    if kind == "manipulator" and ee in ("gripper", "hand") \
            and any(w in p for w in grasp_words) and not any(w in p for w in transport_words):
        from virturoid.services.grasp_eval import evaluate_grasp_lift
        r = evaluate_grasp_lift(gene)
        return {"task": "grasp_lift", "metric": "success_rate", "value": r["success_rate"], "detail": r}

    # default: a manipulator does pick-place (sort / place-to-target), scored on real physics. Give it a
    # fair phase length (the composed arm needs ~300 steps to complete reach→grasp→place; the 200-step
    # default reports a misleading 0). This is an as-built score — co-design/training lifts it further.
    from virturoid.services.task_runtime import select_task_spec, generate_task_scenes, evaluate_gene_on_task
    spec = select_task_spec(prompt or "sort blocks into bins")
    scenes = generate_task_scenes(gene, spec, count=4)
    params = controller_params or {"kp": 9.0, "kd": 1.4, "phase_steps": 300}
    r = evaluate_gene_on_task(gene, spec, scenes, params=params)
    return {"task": spec.task_type, "metric": "success_rate", "value": r["success_rate"], "detail": r}
