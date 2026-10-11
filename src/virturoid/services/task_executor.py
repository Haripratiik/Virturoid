"""Deterministic task executor: run a (verified) TaskSpec by dispatching each step to its registered
skill primitive, collect the facts the real physics outcomes establish, and score the goal predicates.

No LLM at runtime — a pure interpreter over the skill sequence. Verification runs first (honest: an
infeasible task is reported, never silently mis-run); score is the fraction of goal predicates the run
actually established, so partial progress is reported truthfully rather than as a hard pass/fail.

v1 runs each step as its own episode (the existing skills are single-episode). Continuous-sim state
hand-off between steps (e.g. navigate-then-grasp in one rollout) is the deferred v2 upgrade.
"""

from __future__ import annotations

from virturoid.services.capability_registry import REGISTRY

#: The grasp model a skill reports when it did NOT simulate the grasp: the scripted pick-place controller
#: welds the block to the end effector the moment they touch (``task_runtime.evaluate_gene_on_task``).
IDEALIZED_GRASP = "idealized_pin"

#: Bounds on what a step's ``detail`` may carry out of the executor. A skill's own report holds whole episodes
#: (per-scene metrics, grasp attempts with their frames, rollout traces); the step keeps its scalars and short
#: strings and a count of everything else, so a tool response stays a few hundred bytes per step.
_DETAIL_MAX_KEYS = 16
_DETAIL_MAX_STR = 200


def _compact_detail(detail) -> dict:
    """The scalars and short strings of a skill's ``detail`` report, never its frames or arrays.

    ``run_task`` used to drop ``detail`` altogether, and with it every number a skill measured about itself:
    the placement rate, the blocks placed out of the blocks tried, the grasp model, the reason an attempt
    failed. A list of per-episode or per-attempt records is kept as a count plus a tally of the outcome labels
    the records carry (``failure_label`` / ``reason`` / ``status``), because "3 of 4 missed the grasp" is what an
    agent needs from them and the records themselves are not. Non-finite numbers become ``None`` so the result
    stays strict JSON."""
    import math
    import numbers
    from collections import Counter

    out: dict = {}
    if not isinstance(detail, dict):
        return out
    for key, val in detail.items():
        if len(out) >= _DETAIL_MAX_KEYS:
            break
        key = str(key)
        if getattr(val, "shape", None) == () and hasattr(val, "item"):     # a numpy scalar
            val = val.item()
        if val is None or isinstance(val, bool):
            out[key] = val
        elif isinstance(val, numbers.Real):
            num = float(val)
            out[key] = (None if not math.isfinite(num) else
                        int(val) if isinstance(val, numbers.Integral) else round(num, 4))
        elif isinstance(val, str):
            out[key] = val if len(val) <= _DETAIL_MAX_STR else val[:_DETAIL_MAX_STR - 3] + "..."
        elif isinstance(val, (list, tuple)) and val and all(isinstance(v, dict) for v in val):
            out[f"{key}_n"] = len(val)
            tally = Counter(str(v.get("failure_label") or v.get("reason") or v.get("status") or "")
                            for v in val)
            tally.pop("", None)
            if tally:
                out[f"{key}_outcomes"] = ", ".join(f"{label}: {n}" for label, n in tally.most_common(4))
        # anything else (frames, arrays, nested reports) stays with the skill
    return out


def run_task(spec, gene, *, verify: bool = True, record: bool = False) -> dict:
    """Execute ``spec`` on ``gene``. Returns ``{feasible, success, score, goal_met, goal_total, steps,
    issues}``. If ``verify`` and the spec is infeasible, it is NOT run — the gaps are returned instead.
    ``record`` asks each skill to capture a replayable episode; the result then carries ``view`` + ``model_xml``
    so the UI can show the REAL run of whatever skill was dispatched (general: no per-task UI code).

    ``success``/``score``/``goal_met`` are the GOAL PREDICATES, scored under whatever physics each skill ran,
    and that includes a skill that idealized its grasp. Two fields keep that from passing unnoticed:
    ``idealized_grasp`` is true when any step pinned the object instead of grasping it, and
    ``goal_met_without_idealized_grasp`` counts only the goal predicates some NON-idealized step established.
    The agent-facing tools (``agent_design_tools.run_task``/``submit_task``) score success on the latter; each
    step carries its own ``metric``, a compact ``detail`` (see :func:`_compact_detail`) and its ``grasp_model``."""
    if verify:
        from virturoid.services.task_verifier import verify_task
        v = verify_task(spec, gene)
        if not v["ok"]:
            return {"feasible": False, "success": False, "score": 0.0, "goal_met": 0,
                    "goal_total": len(spec.goal), "steps": [], "issues": v["issues"], "kind": v["kind"],
                    "view": None, "model_xml": None}

    established = set()
    established_for_real = set()                        # facts a step established WITHOUT pinning the object
    step_results = []
    view = model_xml = grasp_model = grasp_success_rate = None
    for step in spec.steps:
        sig = REGISTRY.get(step.skill)
        if sig is None:                                  # guarded by the verifier, but stay safe
            step_results.append({"step_id": step.step_id, "skill": step.skill, "success": False,
                                 "error": "unknown skill"})
            continue
        outcome = sig.runner(gene, {**step.params, "record": record})
        established |= set(outcome.established)
        if getattr(outcome, "view", None) is not None:   # keep the recorded episode of the visible task step
            view, model_xml = outcome.view, outcome.model_xml
        gm = (getattr(outcome, "detail", None) or {}).get("grasp_model")
        if gm != IDEALIZED_GRASP:
            established_for_real |= set(outcome.established)
        # surface the grasp disclosure (contact vs idealized pin). An idealized step's disclosure is never
        # overwritten by a later step's: one pinned grasp anywhere in the plan is what the run has to admit to.
        if gm and grasp_model != IDEALIZED_GRASP:
            grasp_model = gm
            grasp_success_rate = round(float(outcome.metric), 3)   # the REAL placement rate behind the disclosure,
            #                                                        so the UI never lets a lenient goal-predicate
            #                                                        score (1.0 at >=0.5 placed) overstate the truth
        row = {"step_id": step.step_id, "skill": step.skill, "success": bool(outcome.success),
               "metric": round(float(outcome.metric), 3), "detail": _compact_detail(getattr(outcome, "detail", None))}
        if gm:
            row["grasp_model"] = gm
        step_results.append(row)

    goal_met = sum(1 for g in spec.goal if g.op in established)
    total = max(1, len(spec.goal))
    return {"feasible": True, "success": goal_met == len(spec.goal),
            "score": round(goal_met / total, 3), "goal_met": goal_met, "goal_total": len(spec.goal),
            "steps": step_results, "issues": [], "view": view, "model_xml": model_xml, "grasp_model": grasp_model,
            "grasp_success_rate": grasp_success_rate, "idealized_grasp": grasp_model == IDEALIZED_GRASP,
            "goal_met_without_idealized_grasp": sum(1 for g in spec.goal if g.op in established_for_real)}


def evaluate_task(prompt: str, gene, *, llm="auto", record: bool = False, memory_dir=None) -> dict:
    """General entry point: PROPOSE a TaskSpec from a free prompt, VERIFY it against the robot, and RUN
    it — the morphology-agnostic replacement for per-class dispatch. Returns the executor result plus the
    planned skills + provenance (so the UI can show what it tried and whether the plan was feasible).
    ``record=True`` captures a replayable episode of the run for the viewport. ``memory_dir`` (Phase 2 RAG):
    when set, the planner gets a PRIOR-KNOWLEDGE block for this body so it proposes tasks the morphology has
    accomplished before."""
    if gene is None:                                      # defensive: never crash the caller on a missing robot
        return {"feasible": False, "success": False, "score": 0.0, "goal_met": 0, "goal_total": 0,
                "steps": [], "issues": ["no robot to run the task on"], "view": None, "model_xml": None,
                "task": "none", "task_source": "none", "steps_planned": []}
    from virturoid.services.task_proposer import propose_task
    prior = ""
    if memory_dir is not None:
        try:
            from virturoid.services.knowledge_context import assemble_prior_knowledge
            prior = assemble_prior_knowledge(memory_dir, gene=gene)
        except Exception:  # noqa: BLE001 - RAG is best-effort; never break task execution
            prior = ""
    spec = propose_task(prompt, gene, llm=llm, prior_knowledge=prior)
    result = run_task(spec, gene, record=record)
    result.update({"task": spec.name, "task_source": spec.source,
                   "steps_planned": [s.skill for s in spec.steps]})
    return result


# Skills whose effectiveness comes from a LEARNED policy (not a fixed script): if one of these fails and no
# policy is banked yet, complete_task ACQUIRES one on demand (recall->adapt->bank) and retries. locomote -> a
# MorphPolicy gait; grasp_lift -> a per-token MorphPolicy grasp residual (both transfer + compound).
_LEARNABLE_SKILLS = {"locomote", "grasp_lift"}
_ACQUIRE_SKILL_FOR = {"locomote": "walk", "grasp_lift": "grasp"}   # task skill -> skill_library skill name


def complete_task(prompt: str, gene, *, llm="auto", learn: bool = True, models_dir: str = "models",
                  memory_dir: str = "build/memory", generations: int = 40, pop: int = 26, steps: int = 320,
                  progress=None) -> dict:
    """LEARN-ON-DEMAND task completion through the SKILL FLYWHEEL: propose → verify → run a task, and if it
    fails because a LEARNABLE control skill (locomotion) underperformed, ACQUIRE the skill — the agent RECALLS
    transferable knowledge banked from prior robots/users (cross-morphology) and ADAPTS it to this body, or
    cold-trains if nothing is banked yet — then banks it back so the next robot/user trains faster. So the
    robot doesn't merely *attempt* the task, it *learns to do it*, and that learning COMPOUNDS across builds.
    Returns the executor result plus ``{learned, attempts, acquisition}``. ``learn=False`` = one eval pass.
    See [[virturoid-flywheel-vision]] (services/skill_library.acquire_skill)."""
    res = evaluate_task(prompt, gene, llm=llm)
    if res.get("success") or not learn:
        return {**res, "learned": False, "attempts": 1}

    failed_learnable = {s["skill"] for s in res.get("steps", [])
                        if s.get("skill") in _LEARNABLE_SKILLS and not s.get("success")}
    if not failed_learnable:
        return {**res, "learned": False, "attempts": 1}

    from pathlib import Path

    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.skill_library import acquire_skill
    skill_name = next((_ACQUIRE_SKILL_FOR[s] for s in sorted(failed_learnable) if s in _ACQUIRE_SKILL_FOR), "walk")
    if progress:
        progress(f"acquiring {sorted(failed_learnable)} ('{skill_name}') via the skill flywheel (recall banked "
                 f"knowledge → adapt to this body → bank back)…")
    db_path = Path(memory_dir) / "virturoid_memory.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition = None
    try:
        with MemoryDB(db_path) as db:
            acquisition = acquire_skill(gene, skill_name, db=db, models_dir=models_dir, target=0.45,
                                        cold=(generations, pop, steps, 1),
                                        warm=(max(8, generations // 3), pop, steps, 1), progress=progress)
    except Exception:  # noqa: BLE001 - the flywheel is best-effort; never block task completion
        from virturoid.services.learn_locomotion import banked_policy_for, learn_locomotion
        if banked_policy_for(gene, models_dir=models_dir) is None:
            learn_locomotion(gene, generations=generations, pop=pop, steps=steps, models_dir=models_dir)
    res2 = evaluate_task(prompt, gene, llm=llm)
    return {**res2, "learned": True, "attempts": 2, "acquisition": acquisition}
