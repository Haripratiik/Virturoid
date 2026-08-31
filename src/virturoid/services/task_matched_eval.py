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


from virturoid.services.body_kind import body_kind, has_wheels as _has_wheels  # noqa: F401 - re-export


def robot_kind(gene: RobotGene) -> str:
    """Classify a robot by STRUCTURE, not its (LLM-authored) class string — so a 'hexapod', 'spider',
    'frog', 'octopod', 'snake' etc. route by what they physically ARE: wheels→mobile, a free-floating
    jointed body→legged, a gripper/hand→manipulator, a spray nozzle→spray. Topology generalizes to any
    morphology the LLM designs; string-matching the four anticipated class names does not.

    The rules themselves now live in ``body_kind`` — ONE derivation, because this function's private copy of
    them kept drifting from the other classifiers and picking the wrong rubric for real robots (it checked
    HANDS before LEGS, so an imported Talos came out "manipulator" and was scored on pick-place; and a drone
    with zero joints fell through every branch to the same answer). See that module for the bug history."""
    return body_kind(gene).kind


def robot_capabilities(gene: RobotGene) -> set[str]:
    """The SET of skill-morphologies a body can satisfy. ``robot_kind`` returns one PRIMARY kind for dispatch,
    but a COMPOSITE has more than one: a mobile_manipulator (wheels + a gripper arm) is BOTH 'mobile' and
    'manipulator', so the task verifier must let it run pick_place AND navigate — the deep-analysis carrybot
    was wrongly rejected from its own pick task because kind()='mobile' hid the arm it physically carries.

    The extra capabilities are ADDED to the primary kind, never derived independently of it: the old copy of
    the legged test here ('floating + 2 revolute + no cylinder wheels') handed the LEGGED capability to an
    imported Tiago — a wheeled base — so a walking task read as feasible for a robot that cannot walk."""
    bk = body_kind(gene)
    ee = gene.end_effector_type or "none"
    caps: set[str] = {bk.kind}
    if ee in ("gripper", "hand", "suction"):
        caps.add("manipulator")
    if ee == "spray_nozzle":
        caps.add("spray")
    # A body can WALK while its primary kind is its end effector's (a legged sprayer routes to coverage, but it
    # still locomotes). Measured legs settle it when body_kind compiled the body; otherwise the long-standing
    # structural presumption stands — minus the bodies now known to roll or fly, which is the Tiago fix.
    if bk.kind == "legged" or (bk.n_legs or 0) >= 2:
        caps.add("legged")
    elif (bk.n_legs is None and bk.kind not in ("mobile", "aerial", "aquatic")
          and gene.base_mount == "free" and sum(1 for s in gene.segments if s.joint_type == "revolute") >= 2):
        caps.add("legged")
    return caps


def evaluate_robot(gene: RobotGene, *, prompt: str = "", controller_params: dict | None = None,
                   scene_params: dict | None = None) -> dict:
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
        # a LIMBLESS serial spine (snake) is scored on LAND SERPENTINE undulation, not a leg gait it can't run
        try:
            from virturoid.services.aquatic import _is_serial_spine
            if _is_serial_spine(gene):
                from virturoid.services.morph_policy import serpentine_rollout
                sr = serpentine_rollout(gene, steps=2000)
                return {"task": "serpentine_locomotion", "metric": "distance_m",
                        "value": round(float(sr.get("planar_m", 0.0)), 3), "detail": sr}
        except Exception:  # noqa: BLE001 - serpentine is value-add; fall through to the leg eval on any error
            pass
        from virturoid.services.locomotion_controller import run_locomotion_episode
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        r = run_locomotion_episode(mj)
        # A task score is forward achievement, never unsigned planar travel.
        # This keeps the fallback reachable for a backward trot and prevents a
        # lurch in the wrong direction from outranking a genuine crawl.
        forward = float(r["forward_m"]); detail = r; scored_gait = "trot"
        upright = bool(r.get("upright"))
        # Score the body by the BEST of its AVAILABLE gaits, not one fixed gait: a wide-stance quad that ROLLS
        # under the diagonal trot may WALK under the statically-stable CRAWL (one foot up at a time, 3 planted).
        # Try the crawl when the trot underperforms OR ROLLED OVER (upright=False) — the tippy-quad failure. This
        # lets per-request generation/co-design DISCOVER the gait that works for THIS body — never a fixed gait.
        if forward < 0.5 or not upright:
            try:
                from virturoid.services.morph_policy import crawl_gait_rollout
                cr = crawl_gait_rollout(gene, steps=1200)
                cr_up = bool(cr.get("survived")) and float(cr.get("height_ratio", 0.0)) > 0.5
                # an UPRIGHT crawl beats a rolled trot regardless of raw distance; else it must travel further
                if cr_up and float(cr.get("forward", 0.0)) > (forward if upright else -1e9):
                    forward, detail, scored_gait, upright = round(float(cr["forward"]), 3), cr, "crawl", True
            except Exception:  # noqa: BLE001 - the crawl is an OPTIONAL alternative gait; never break the eval
                pass
        # A body that ROLLS OVER / FALLS but drifts forward is NOT walking (F8 gameable-distance gate): score it on
        # staying upright, so a 0.87 m roll-over reads ~0 and the walkable-stance fallback correctly fires for it.
        return {"task": "locomotion", "metric": "forward_m", "value": (max(0.0, forward) if upright else 0.0),
                "detail": {**detail, "scored_gait": scored_gait, "upright": upright}}

    if kind == "mobile":
        from virturoid.services.navigation_controller import run_navigation_episode
        mj = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene))
        goals = [(1.0, 0.0), (0.0, -0.9), (-0.7, 0.4), (0.7, 0.5)]
        results = [run_navigation_episode(mj, g, [], horizon=3000) for g in goals]
        reached = sum(1 for r in results if r["status"] == "reached")
        return {"task": "navigation", "metric": "goal_reach_rate", "value": round(reached / len(goals), 3),
                "detail": {"reached": reached, "goals": len(goals)}}

    if kind == "aerial":
        from virturoid.services.ai_native_tools import _honest_fly
        goals = [(0.0, 0.0, 1.2), (1.5, 0.8, 1.4), (-1.2, 0.6, 1.0)]      # hover + two off-axis waypoints
        results = [_honest_fly(gene, target=g, steps=2000) for g in goals]
        reached = sum(1 for r in results if r.get("reached_target"))
        return {"task": "flight", "metric": "waypoint_reach_rate", "value": round(reached / len(goals), 3),
                "detail": {"reached": reached, "goals": len(goals)}}

    if kind == "aquatic":
        from virturoid.services.ai_native_tools import _honest_swim
        r = _honest_swim(gene, steps=2000)
        return {"task": "swim", "metric": "swim_m", "value": float(r.get("swim_m", 0.0)), "detail": r}

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
        # WS9: controller_params may carry a learned residual_params_path; scene_params sets the difficulty
        # (M5_arm_grasp_hard lowers friction so the scripted grasp is brittle and the residual has room to win).
        r = evaluate_grasp_lift(gene, controller_params=controller_params, **(scene_params or {}))
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
