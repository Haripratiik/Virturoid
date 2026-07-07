"""Training ladder planner — dossier core algorithm (step 1-2) + R2/R6/R8 as one deterministic compile step.

Given a task (family or free text) and a robot/task reference, this produces a :class:`TrainingPlan` that names
the skill family, the three-phase trainer ladder (cheap reuse -> distill/adapt -> expensive RL), the teacher
sources to try *before* RL (dossier R6: demonstrations are the default manipulation substrate), the backend
budget, deploy-sim checkpoint selection (R8), and evaluator-certified banking rules (R4/R6). It is deterministic
and backend-free: it decides the plan; it does not run any trainer (AGENTS.md layering — no simulator calls here).
"""

from __future__ import annotations

import re

from virturoid.schemas.training_plan import TRAINER_RUNGS, TaskFamily, TrainingPlan

MANIPULATION_FAMILIES = frozenset({
    TaskFamily.GRASP, TaskFamily.REACH, TaskFamily.PUSH,
    TaskFamily.PICK_PLACE, TaskFamily.SORT, TaskFamily.TOOL_USE,
})

_MOBILITY_PATTERN = re.compile(
    r"\bwalk|gait|locomot|trot|\bhop\b|crawl|climb|\brun\b|navigat|obstacle|go to|drive to|\bpath\b|avoid",
    re.IGNORECASE,
)
_NAV_PATTERN = re.compile(r"navigat|obstacle|go to|drive to|\bpath\b|avoid", re.IGNORECASE)
_LOCO_PATTERN = re.compile(r"\bwalk|gait|locomot|trot|\bhop\b|crawl|climb|\brun\b", re.IGNORECASE)
_INSPECT_PATTERN = re.compile(r"inspect|survey|monitor|\bscan\b", re.IGNORECASE)

# Manipulation keywords, priority-ordered so compound phrases win over their sub-words.
_MANIP_KEYWORDS: tuple[tuple[str, TaskFamily], ...] = (
    (r"\bsort\b|sorting", TaskFamily.SORT),
    (r"\btool\b|screw|drill|wrench|insert", TaskFamily.TOOL_USE),
    (r"pick[\s\-]*and[\s\-]*place|pick[\s\-]*place|\bplace\b", TaskFamily.PICK_PLACE),
    (r"\bpush\b|shove|nudge", TaskFamily.PUSH),
    (r"grasp|grab|pick up|\bpick\b|\bgrip\b|\blift\b", TaskFamily.GRASP),
    (r"\breach\b", TaskFamily.REACH),
)

_TEACHERS: dict[TaskFamily, list[str]] = {
    TaskFamily.LOCOMOTION: ["recipe_controller", "banked_locomotion_policy", "transfer_seed"],
    TaskFamily.NAVIGATION: ["planner_path", "banked_navigation_policy", "perception_nav_demo"],
    TaskFamily.GRASP: ["scripted_topdown_grasp", "ik_planner", "banked_grasp_residual"],
    TaskFamily.REACH: ["ik_planner", "reach_cem", "banked_reach_policy"],
    TaskFamily.PUSH: ["object_relative_push_primitive", "banked_push_policy"],
    TaskFamily.PICK_PLACE: ["demonstration_amplifier", "scripted_topdown_grasp", "ik_planner"],
    TaskFamily.SORT: ["demonstration_amplifier", "scripted_topdown_grasp", "vla_or_diffusion_teacher"],
    TaskFamily.TOOL_USE: ["demonstration_amplifier", "ik_planner", "vla_or_diffusion_teacher"],
    TaskFamily.INSPECT: ["planner_path", "perception_nav_demo"],
    TaskFamily.COMPOSITE: ["skill_phase_decomposition", "banked_skills", "demonstration_amplifier"],
    TaskFamily.UNKNOWN: ["scripted_baseline", "banked_skills"],
}

_SKILL_SEQUENCES: dict[TaskFamily, list[str]] = {
    TaskFamily.LOCOMOTION: ["stand", "track_velocity", "recover"],
    TaskFamily.NAVIGATION: ["perceive", "plan_path", "avoid", "reach_goal"],
    TaskFamily.GRASP: ["locate", "approach", "align", "grasp", "lift"],
    TaskFamily.REACH: ["locate", "approach", "hold"],
    TaskFamily.PUSH: ["locate", "approach", "push", "settle"],
    TaskFamily.PICK_PLACE: ["locate", "approach", "grasp", "transport", "release"],
    TaskFamily.SORT: ["locate", "classify", "grasp", "transport", "release"],
    TaskFamily.TOOL_USE: ["locate", "acquire_tool", "align", "actuate", "release"],
    TaskFamily.INSPECT: ["perceive", "traverse", "capture"],
    TaskFamily.COMPOSITE: ["decompose", "sequence_skills"],
    TaskFamily.UNKNOWN: ["attempt", "evaluate"],
}


def classify_task_family(text_or_family: str | TaskFamily) -> TaskFamily:
    """Deterministically map a task family or free-text task to a :class:`TaskFamily`.

    A task that combines mobility (walk/navigate) with manipulation (grasp/place/...) is a COMPOSITE
    (mobile manipulation), matching the dossier's composite family.
    """
    if isinstance(text_or_family, TaskFamily):
        return text_or_family
    text = text_or_family or ""
    manip: TaskFamily | None = None
    for pattern, family in _MANIP_KEYWORDS:
        if re.search(pattern, text, re.IGNORECASE):
            manip = family
            break
    mobility = bool(_MOBILITY_PATTERN.search(text))
    if mobility and manip is not None:
        return TaskFamily.COMPOSITE
    if manip is not None:
        return manip
    if _NAV_PATTERN.search(text):
        return TaskFamily.NAVIGATION
    if _LOCO_PATTERN.search(text):
        return TaskFamily.LOCOMOTION
    if _INSPECT_PATTERN.search(text):
        return TaskFamily.INSPECT
    return TaskFamily.UNKNOWN


def is_manipulation(family: TaskFamily) -> bool:
    return family in MANIPULATION_FAMILIES


def plan_training(
    task: str | TaskFamily,
    *,
    robot_genome_id: str,
    task_graph_id: str,
    scene_set_refs: list[str] | None = None,
    observation_contract_ref: str | None = None,
    objective_ref: str | None = None,
    gpu_available: bool = True,
    deployable: bool = True,
) -> TrainingPlan:
    """Compile a task into a deterministic three-phase :class:`TrainingPlan` (does not run training)."""
    family = classify_task_family(task)
    ladder = list(TRAINER_RUNGS)
    if not gpu_available:
        ladder = ladder + ["cpu_fallback"]

    backend_budget = {
        "reuse_evaluate": "cpu_eval",
        "distill_adapt": "cpu_bc" if not gpu_available else "gpu_bc",
        "optimize_rl": "gpu_mjx_ppo" if gpu_available else "cpu_es",
    }
    banking_rules = {
        "require_heldout_eval": True,        # R6: never bank a skill without held-out success
        "require_process_gate": True,        # R4: process-gated, not reward-hacked
        "require_actuator_cert_if_deployable": bool(deployable),
    }
    dr_profile = (
        {"mass": True, "friction": True, "damping": True, "sensor_latency": True, "actuator_clamp": True}
        if deployable else {}
    )
    return TrainingPlan(
        id=f"tp_{family.value}_{robot_genome_id}",
        task_family=family.value,
        robot_genome_id=robot_genome_id,
        task_graph_id=task_graph_id,
        scene_set_refs=list(scene_set_refs or []),
        observation_contract_ref=observation_contract_ref,
        skill_sequence=list(_SKILL_SEQUENCES.get(family, [])),
        trainer_ladder=ladder,
        teacher_sources=list(_TEACHERS.get(family, _TEACHERS[TaskFamily.UNKNOWN])),
        objective_ref=objective_ref,
        backend_budget=backend_budget,
        domain_randomization_profile=dr_profile,
        checkpoint_selection={"by": "deploy_sim", "keep_intermediate": True},  # R8
        banking_rules=banking_rules,
    )


def ladder_report(plan: TrainingPlan) -> dict:
    """Print-ready summary of the chosen ladder (dossier Phase 0: 'print the chosen ladder for a task')."""
    family = plan.task_family
    manip = family in {f.value for f in MANIPULATION_FAMILIES}
    return {
        "training_plan_id": plan.id,
        "task_family": family,
        "trainer_ladder": list(plan.trainer_ladder),
        "teacher_sources": list(plan.teacher_sources),
        "skill_sequence": list(plan.skill_sequence),
        "backend_budget": dict(plan.backend_budget),
        "checkpoint_selection": dict(plan.checkpoint_selection),
        "banking_rules": dict(plan.banking_rules),
        "notes": [
            "Try reuse_evaluate first; only escalate to optimize_rl for the remaining residual.",
            ("Manipulation: generate teacher demonstrations before RL (demonstration amplifier)."
             if manip else
             "Screen banked policies zero-shot (transfer seed) before warm-starting."),
            "Bank a skill only after held-out evaluation (evaluator-certified).",
        ],
    }


def format_ladder_report(report: dict) -> str:
    lines = [
        f"Training plan: {report['training_plan_id']}  (family={report['task_family']})",
        f"  ladder:   {' -> '.join(report['trainer_ladder'])}",
        f"  teachers: {', '.join(report['teacher_sources'])}",
        f"  skills:   {' -> '.join(report['skill_sequence'])}",
        f"  backend:  {report['backend_budget']}",
        f"  banking:  {report['banking_rules']}",
    ]
    for note in report["notes"]:
        lines.append(f"  - {note}")
    return "\n".join(lines)
