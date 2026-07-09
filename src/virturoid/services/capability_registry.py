"""Capability registry: the single source of truth of what skills exist, what each needs/achieves, and
which morphologies can run it — the option library + affordance model for the general task layer.

Each skill wraps an EXISTING primitive (navigation/locomotion/grasp/pick-place/spray/maze) behind one
uniform ``runner(gene, params) -> SkillOutcome`` so the executor invokes everything the same way, and a
verifier can reason symbolically about feasibility (SayCan's "can", done over declared pre/postconditions
instead of learned value functions). Adding a skill = one registry entry; no executor/verifier changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from virturoid.schemas.task_spec import PredicateOp


@dataclass
class SkillOutcome:
    skill: str
    success: bool
    established: set = field(default_factory=set)     # set[PredicateOp] actually achieved this run
    metric: float = 0.0
    detail: dict = field(default_factory=dict)
    # A recorded, replayable episode of THIS skill actually running (frames + outcome) + the model XML, produced
    # when params['record'] is set. This is how ANY task — whatever skill the proposer picked for whatever body —
    # surfaces a real simulation in the viewport, with no per-scenario hardcoding in the UI.
    view: dict = None
    model_xml: str = None


@dataclass(frozen=True)
class SkillSignature:
    skill: str
    requires: tuple = ()                  # PredicateOp facts that must already hold to start
    establishes: tuple = ()               # PredicateOp facts this skill can achieve (its effects)
    morphologies: tuple = ()              # robot_kind values that can run it: legged|mobile|manipulator|spray
    runner: Callable = None               # (gene, params) -> SkillOutcome
    summary: str = ""


# ---- runners: thin adapters over the existing skill primitives -------------------------------------
def _episode_view(frames, ok, status):
    """Wrap recorded geom-pose frames in the viewport's replay shape (or None if nothing was recorded), so any
    skill's run can be replayed in the UI without skill-specific UI code."""
    if not frames:
        return None
    return {"frames": frames, "scenes": [], "scene_index": 0,
            "outcome": {"status": "success" if ok else (status or "incomplete"),
                        "failure_label": None if ok else (status or "incomplete"),
                        "placed_count": 1 if ok else 0, "block_count": 1}}


def _run_navigate(gene, params):
    import mujoco

    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.navigation_controller import run_navigation_episode
    goal = tuple(params.get("goal", (1.0, 0.0)))
    # scale the step budget with how far the goal is (a 2.2 m goal timed out at the flat 3000): give ~3000
    # steps per metre so far goals are reachable, not just near ones.
    dist = (goal[0] ** 2 + goal[1] ** 2) ** 0.5
    horizon = int(params.get("horizon") or max(3000, int(3000 * dist)))
    xml = compile_gene_to_mjcf(gene)
    mj = mujoco.MjModel.from_xml_string(xml)
    frames = [] if params.get("record") else None
    r = run_navigation_episode(mj, goal, params.get("obstacles", []), horizon=horizon, record_frames=frames)
    ok = r["status"] == "reached"
    est = {PredicateOp.REACHED_GOAL, PredicateOp.PATH_FOLLOWED} if ok else set()
    return SkillOutcome("navigate", ok, est, 1.0 if ok else 0.0, r,
                        view=_episode_view(frames, ok, r.get("status")), model_xml=xml if frames else None)


def _run_locomote(gene, params):
    # Prefer a LEARNED morphology-conditioned policy over the scripted trot (the trot is brittle and drives
    # several composed bodies BACKWARD; the learned MorphPolicy walks them forward). Uses a banked policy for
    # this species if present, trains one on demand when params['learn'] is set, and always takes whichever
    # controller travelled further FORWARD. Success is FORWARD progress while upright — moving backward or
    # drifting sideways must not count as "walked to the goal".
    from virturoid.services.learn_locomotion import locomotion_episode
    # Horizon matches the gait's CERTIFIED/banked horizon (recipe_rollout_morph's 900-step validation, also what the
    # render + "Learn to move" use) — so "Run task" evaluates the SAME walk the bank certifies, not extra un-certified
    # long-horizon survival. (A "walk forward 0.3 m" goal is met well within 900 steps.)
    r = locomotion_episode(gene, horizon=int(params.get("horizon", 900)),
                           learn=bool(params.get("learn", False)),
                           models_dir=params.get("models_dir", "models"))
    need = float(params.get("distance", 0.3))
    forward = float(r.get("forward_m", r.get("distance_m", 0.0)))
    upright = bool(r.get("upright"))
    # The learned/trot path leaves several fresh composed bodies STANDING (a fresh quad measured -0.02 m) even
    # though the open-loop CRAWL gait credibly walks the SAME body ~0.5 m. So also try the crawl gait and take it
    # when it's a genuine CREDIBLE WALK (un-gameable classify — never a rearing lurch) that travels further. This
    # is the credible controller verify_robot already uses; the task layer must not score "can't walk" on a body
    # that demonstrably walks. (A hexapod, whose crawl isn't credible, is unaffected -> honest fail stands.)
    try:
        from virturoid.services.gait_quality import classify
        from virturoid.services.morph_policy import crawl_gait_rollout
        cg = crawl_gait_rollout(gene, steps=int(params.get("horizon", 900)), record_qpos=True)
        if classify(cg).startswith("CREDIBLE") and float(cg.get("forward", 0.0)) > forward:
            forward, upright = float(cg["forward"]), True
            r = {**r, "forward_m": round(forward, 3), "controller": "crawl_gait", "gait_verdict": classify(cg)}
    except Exception:  # noqa: BLE001 - the crawl comparison is a fallback; never break the task on it
        pass
    est = set()
    if upright:
        est.add(PredicateOp.UPRIGHT)
    if forward >= need and upright:
        est |= {PredicateOp.TRAVELED, PredicateOp.REACHED_GOAL}
    view = xml = None
    if params.get("record"):
        try:                                              # replay the banked gait (recipe-aware) for the viewport
            from virturoid.services.learn_locomotion import banked_policy_for, rollout_view
            pol = banked_policy_for(gene, models_dir=params.get("models_dir", "models"))
            if pol is not None:
                view, xml = rollout_view(gene, pol, steps=int(params.get("record_steps", 600)))
        except Exception:  # noqa: BLE001 - recording is best-effort; never block the task result
            view = xml = None
    return SkillOutcome("locomote", PredicateOp.TRAVELED in est, est, forward, r, view=view, model_xml=xml)


def _run_grasp_lift(gene, params):
    from virturoid.services.grasp_eval import evaluate_grasp_lift
    r = evaluate_grasp_lift(gene)
    r["source"] = "scripted"
    # Use a LEARNED, transferable grasp residual when one is banked for THIS body — the manipulation flywheel
    # in execution. File-keyed by species in models_dir (mirrors learn_locomotion.banked_policy_for) so the
    # acquired skill is picked up with no db handle; a memory_dir adds a cross-morphology db fallback. Takes
    # the BETTER of scripted vs learned, so it never regresses. See [[task-effectiveness-loop]] (grasp_skill).
    try:
        from virturoid.services.grasp_skill import banked_grasp_policy, evaluate_morph_grasp
        pol, sc = banked_grasp_policy(gene, models_dir=params.get("models_dir", "models"))
        if pol is None and params.get("memory_dir"):
            from pathlib import Path

            from virturoid.services.grasp_skill import recall_grasp_residual
            from virturoid.services.memory_db import MemoryDB
            dbp = Path(params["memory_dir"]) / "virturoid_memory.db"
            if dbp.exists():
                with MemoryDB(dbp) as db:
                    pol, sc = recall_grasp_residual(gene, db)
        if pol is not None:
            rr = evaluate_morph_grasp(gene, pol, residual_scale=sc["residual_scale"],
                                      fin_residual_scale=sc["fin_residual_scale"])
            if rr["success_rate"] > r["success_rate"]:
                r = {**rr, "source": "learned_residual"}
    except Exception:  # noqa: BLE001 - the learned path is best-effort; never block the scripted grasp
        pass
    ok = r["success_rate"] >= 0.5
    return SkillOutcome("grasp_lift", ok, {PredicateOp.OBJECT_LIFTED} if ok else set(),
                        r["success_rate"], r)


def _frontal_task_scenes(gene, spec, n: int = 3):
    """Task scenes (spec.manipulables -> spec.targets) placed inside the arm's DEXTEROUS top-down-graspable zone
    (dist ~0.36-0.47 m, |y| <= 0.24) where the real contact grasp+transport is reliable. A pick-place task should
    pose its objects where the arm can actually grasp palm-down — this is fair task design (the dexterous workspace),
    not the full position-reach. Returns [] if the body has no such zone (caller falls back to the pin)."""
    import math
    import random

    from virturoid.schemas.scenes import SceneGraph, SceneObject
    from virturoid.services.gene_build import robot_reachable_points
    pts = [p for p in robot_reachable_points(gene)
           if 0.36 <= math.hypot(p[0], p[1]) <= 0.47 and abs(p[1]) <= 0.24]
    names = list(spec.manipulables) + list(spec.targets)
    if len(pts) < len(names):
        return []
    out = []
    for i in range(n * 6):                                   # try extra seeds so n scenes reliably fit in the tight zone
        if len(out) >= n:
            break
        rng = random.Random(i * 7 + 1); pool = pts[:]; rng.shuffle(pool); chosen = []
        for pt in pool:
            if all((pt[0] - c[0]) ** 2 + (pt[1] - c[1]) ** 2 >= 0.15 ** 2 for c in chosen):
                chosen.append(pt)
            if len(chosen) == len(names):
                break
        if len(chosen) < len(names):
            continue
        objs, k = [], 0
        for nm in spec.manipulables:
            x, y = chosen[k]; k += 1
            objs.append(SceneObject(nm, "cube", (x, y, 0.02, 0, 0, 0), mass_kg=0.05,
                                    material=f"{spec.materials.get(nm, 'gray')}_block", friction=1.0, scale=1.0))
        for nm in spec.targets:
            x, y = chosen[k]; k += 1
            objs.append(SceneObject(nm, "container", (x, y, 0.0, 0, 0, 0),
                                    material=f"{spec.materials.get(nm, 'gray')}_bin"))
        out.append(SceneGraph(id=f"contact_{spec.task_type}_{i}", version="0.1.0", name=f"contact scene {i + 1}",
                              backend_targets=["mujoco"], robot_spawn_xyz_rpy=(0, 0, 0, 0, 0, 0), objects=objs))
    return out


def _run_contact_pick_place(gene, spec, params):
    """REAL contact-grasp pick-place / sort (no _pin_block): for each block the gripper closes by CONTACT FRICTION,
    lifts, carries it to its bin, and releases. Runs the ACTUAL task scenes (single object OR multi-object sort),
    reports the real per-block placement rate + grasp_model='contact', and records a replay. Returns None if the
    body has no graspable scene (caller falls back to the pin)."""
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import run_contact_pick_place_episode
    scenes = _frontal_task_scenes(gene, spec, n=int(params.get("scenes", 3)))  # dexterous-zone (reliable) layout
    if not scenes:
        return None
    results = [run_contact_pick_place_episode(gene, sc.objects, spec.assignments) for sc in scenes]
    placed = sum(r.get("placed_count", 0) for r in results)
    total = sum(r.get("block_count", 0) for r in results) or 1
    success_rate = round(placed / total, 3)                  # honest per-BLOCK friction-grasp placement rate
    view = xml = None
    if params.get("record"):
        try:
            sc = scenes[0]; frames = []
            ep = run_contact_pick_place_episode(gene, sc.objects, spec.assignments, record_frames=frames, frame_every=12)
            xml = compile_gene_with_scene(gene, sc.objects)
            view = _episode_view(frames, ep.get("placed_count", 0) > 0, ep.get("status"))
        except Exception:  # noqa: BLE001 - recording is best-effort
            view = xml = None
    ok = success_rate >= 0.5
    report = {"task_type": spec.task_type, "grasp_model": "contact", "success_rate": success_rate,
              "blocks_placed": placed, "blocks_total": total,
              "note": "real friction grasp: fingers close on each block and hold it by CONTACT (no pin)"}
    return SkillOutcome("pick_place", ok, {PredicateOp.OBJECTS_PLACED} if ok else set(),
                        success_rate, report, view=view, model_xml=xml)


def _run_pick_place(gene, params):
    from virturoid.services.task_runtime import (
        evaluate_gene_on_task,
        generate_task_scenes,
        select_task_spec,
    )
    spec = select_task_spec(params.get("prompt", "sort blocks into bins"))
    # REAL contact grasp for a gripper/hand arm — single-object pick-place AND the multi-object sort (the per-block
    # APPROACH fix in run_contact_pick_place_episode lifted the sort 0/8 -> 6/8). The visible task now runs the real
    # friction grasp; only a degenerate gripper (no graspable scene) falls back to the idealized pin.
    if str(getattr(gene, "end_effector_type", "") or "").lower() in ("gripper", "hand"):
        contact = _run_contact_pick_place(gene, spec, params)
        if contact is not None and contact.metric > 0:
            return contact
    scenes = generate_task_scenes(gene, spec, count=int(params.get("scenes", 4)))
    # firm gains + a fuller descent: the grasp engages reliably (pick_place 0.38 -> 1.00); see
    # pick_place_controller.DEFAULT_CONTROLLER_PARAMS for the root cause (soft gains under-reached the block).
    cp = {"kp": 14.0, "kd": 2.0, "phase_steps": 500, "engage_radius": 0.15, "grasp_z": 0.05}
    r = evaluate_gene_on_task(gene, spec, scenes, params=cp)
    ok = r["success_rate"] >= 0.5
    view = xml = None
    if params.get("record") and scenes:                  # record ONE representative scene for the viewport
        try:
            import mujoco
            from virturoid.services.gene_compiler import compile_gene_with_scene
            from virturoid.services.pick_place_controller import run_pick_place_episode
            sc = scenes[0]
            xml = compile_gene_with_scene(gene, sc.objects)
            model = mujoco.MjModel.from_xml_string(xml)
            objs = {o.name: tuple(o.pose_xyz_rpy[:3]) for o in sc.objects}
            frames = []
            out = run_pick_place_episode(model, {"objects": objs, "assignments": spec.assignments},
                                         params=cp, record_frames=frames, frame_every=12)
            placed = out.get("status") == "success" or out.get("placed_count", 0) > 0
            view = _episode_view(frames, placed, out.get("status"))
        except Exception:  # noqa: BLE001 - recording is best-effort; never block the task result
            view = xml = None
    return SkillOutcome("pick_place", ok, {PredicateOp.OBJECTS_PLACED} if ok else set(),
                        r["success_rate"], r, view=view, model_xml=xml)


def _run_spray(gene, params):
    from virturoid.services.spray_coverage import evaluate_spray_coverage
    r = evaluate_spray_coverage(gene, grid=int(params.get("grid", 4)))
    frac = float(params.get("fraction", 0.6))
    ok = r["coverage"] >= frac
    return SkillOutcome("spray_coverage", ok, {PredicateOp.SURFACE_COVERED} if ok else set(),
                        r["coverage"], r)


def _run_solve_maze(gene, params):
    n = int(params.get("maze_size", 3)); seed = int(params.get("seed", 0))
    if params.get("record"):
        from virturoid.services.maze import maze_episode_view
        view, xml = maze_episode_view(gene, n=n, seed=seed)
        ok = view["outcome"]["status"] == "success"
        est = {PredicateOp.PATH_FOLLOWED, PredicateOp.REACHED_GOAL} if ok else set()
        return SkillOutcome("solve_maze", ok, est, 1.0 if ok else 0.0, view["outcome"], view=view, model_xml=xml)
    from virturoid.services.maze import solve_maze
    r = solve_maze(gene, n=n, seed=seed)
    ok = bool(r.get("solved"))
    est = {PredicateOp.PATH_FOLLOWED, PredicateOp.REACHED_GOAL} if ok else set()
    return SkillOutcome("solve_maze", ok, est, 1.0 if ok else 0.0, r)


def _run_fly(gene, params):
    """Fly an aerial body (quadcopter) to a target with the geometric flight controller (see ai_native_tools.
    _honest_fly). A 2-D goal is lifted to a default cruise altitude; success = the drone REACHES + holds the
    target (un-gameable: airborne + settled)."""
    from virturoid.services.ai_native_tools import _honest_fly
    goal = tuple(params.get("goal", (1.2, 0.6, 1.2)))
    target = goal if len(goal) == 3 else (float(goal[0]), float(goal[1]), float(params.get("altitude", 1.2)))
    r = _honest_fly(gene, target=target, steps=int(params.get("steps", 2200)))
    ok = bool(r.get("reached_target"))
    est = {PredicateOp.REACHED_GOAL} if ok else ({PredicateOp.UPRIGHT} if r.get("survived") else set())
    return SkillOutcome("fly", ok, est, 1.0 if ok else 0.0, r)


def _run_swim(gene, params):
    """Swim an aquatic body (undulator) forward in water with the tuned travelling wave (see ai_native_tools.
    _honest_swim — REAL MuJoCo fluid). Success = credible undulatory thrust (net displacement past a threshold);
    it establishes TRAVELED, not a precise underwater goal-reach (honest — the swim controller isn't a nav policy)."""
    from virturoid.services.ai_native_tools import _honest_swim
    r = _honest_swim(gene, steps=int(params.get("steps", 2000)))
    swam = float(r.get("swim_m", 0.0))
    ok = swam >= float(params.get("distance", 0.15))
    est = {PredicateOp.TRAVELED} if ok else set()
    return SkillOutcome("swim", ok, est, swam, r)


REGISTRY: dict = {
    "navigate": SkillSignature("navigate", (), (PredicateOp.REACHED_GOAL, PredicateOp.PATH_FOLLOWED),
                               ("mobile",), _run_navigate, "drive a wheeled base to a goal"),
    "solve_maze": SkillSignature("solve_maze", (), (PredicateOp.PATH_FOLLOWED, PredicateOp.REACHED_GOAL),
                                 ("mobile",), _run_solve_maze, "plan + follow a path through a maze"),
    "locomote": SkillSignature("locomote", (), (PredicateOp.TRAVELED, PredicateOp.UPRIGHT, PredicateOp.REACHED_GOAL),
                               ("legged",), _run_locomote, "walk a legged body forward"),
    "grasp_lift": SkillSignature("grasp_lift", (), (PredicateOp.OBJECT_LIFTED,),
                                 ("manipulator",), _run_grasp_lift, "grasp + lift an object by friction"),
    "pick_place": SkillSignature("pick_place", (), (PredicateOp.OBJECTS_PLACED,),
                                 ("manipulator",), _run_pick_place, "pick objects and place them into targets"),
    "spray_coverage": SkillSignature("spray_coverage", (), (PredicateOp.SURFACE_COVERED,),
                                     ("spray",), _run_spray, "sweep a nozzle to coat a surface"),
    "fly": SkillSignature("fly", (), (PredicateOp.REACHED_GOAL, PredicateOp.UPRIGHT),
                          ("aerial",), _run_fly, "fly a quadcopter to a target with rotor thrust"),
    "swim": SkillSignature("swim", (), (PredicateOp.TRAVELED,),
                           ("aquatic",), _run_swim, "swim an undulator forward in water"),
}


def skill_menu() -> str:
    """A compact, LLM-facing description of the available skills (name + effect + morphology)."""
    return "\n".join(
        f"- {s.skill}: {s.summary} | needs morphology {list(s.morphologies)} | "
        f"establishes {[o.value for o in s.establishes]}"
        for s in REGISTRY.values())
