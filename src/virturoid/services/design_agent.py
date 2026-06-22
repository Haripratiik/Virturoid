"""Robot Designer agent (plan §8): an LLM proposes the robot body; physics disposes.

The agent asks an LLM for a structured arm design (link lengths, actuator torque),
then validates it against a schema and a physics-grounded reach feasibility gate.
If the proposal is infeasible it feeds the failure back to the model and retries
(self-repair). With no LLM backend configured it returns None and the caller uses
the deterministic formula — so the product runs offline with zero API keys.

This is the "AI deciding the design" layer the plan describes; the existing
co-design optimizer then refines the LLM's starting point against real rollouts.
"""

from __future__ import annotations

import math

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "forearm_mm": {"type": "number"},
        "wrist_mm": {"type": "number"},
        "actuator_torque_nm": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["forearm_mm", "wrist_mm", "actuator_torque_nm"],
    "additionalProperties": False,
}

_BOUNDS = {"forearm_mm": (200.0, 500.0), "wrist_mm": (80.0, 260.0), "actuator_torque_nm": (5.0, 30.0)}

_SYSTEM = (
    "You are Virturoid's robot designer. You design a 3-DOF tabletop manipulator "
    "(base yaw + two pitch joints + gripper) for a block-sorting task. You output ONLY "
    "a JSON object with forearm_mm, wrist_mm, actuator_torque_nm, and a short rationale. "
    "The arm's reach envelope is (forearm_mm + wrist_mm)/1000 + 0.07 metres, measured from "
    "a shoulder ~0.165 m above the table; it must cover the farthest task target with margin. "
    "Prefer the shortest body that comfortably reaches the workspace (over-long arms overshoot "
    "near targets). Actuator torque must hold the arm against gravity (8-22 Nm is typical). "
    "A physics validator will check your proposal and may ask you to correct it."
)


def propose_arm_design(prompt: str, requirements, task, llm, max_repairs: int = 2) -> dict | None:
    """Ask the LLM for a feasible arm design; validate + self-repair. None if no backend."""
    if llm is None:
        return None

    required_reach = _required_reach_for_task(task)
    user = (
        f"Task: {prompt}\n"
        f"Reach requirement: {requirements.reach_m or 0.6} m. Payload: {requirements.payload_kg or 0.1} kg.\n"
        f"Farthest task target distance from the shoulder: {required_reach:.3f} m.\n"
        f"Return a JSON design whose reach envelope >= {required_reach + 0.03:.3f} m."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, DESIGN_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - any backend failure -> caller falls back
            return {"design": None, "backend": getattr(llm, "name", "?"), "attempts": attempts,
                    "feasible": False, "error": str(exc)}
        design, issues = _validate(raw, required_reach)
        if not issues:
            return {
                "design": design,
                "backend": getattr(llm, "name", "?"),
                "attempts": attempts,
                "feasible": True,
                "rationale": raw.get("rationale", ""),
            }
        last_issues = issues
        user = (
            f"{user}\n\nYour previous proposal {{'forearm_mm': {raw.get('forearm_mm')}, "
            f"'wrist_mm': {raw.get('wrist_mm')}, 'actuator_torque_nm': {raw.get('actuator_torque_nm')}}} "
            f"was rejected by the physics validator: {'; '.join(issues)}. Return a corrected JSON design."
        )

    return {"design": None, "backend": getattr(llm, "name", "?"), "attempts": attempts,
            "feasible": False, "issues": last_issues}


def _validate(raw: dict, required_reach: float) -> tuple[dict, list[str]]:
    issues: list[str] = []
    design = {}
    for key, (lo, hi) in _BOUNDS.items():
        try:
            value = float(raw[key])
        except (KeyError, TypeError, ValueError):
            issues.append(f"{key} missing or not a number")
            continue
        if not (lo <= value <= hi):
            issues.append(f"{key}={value} out of range [{lo}, {hi}]")
        design[key] = value
    if {"forearm_mm", "wrist_mm"} <= set(design):
        envelope = (design["forearm_mm"] + design["wrist_mm"]) / 1000.0 + 0.07
        if envelope < required_reach + 0.02:
            issues.append(f"reach envelope {envelope:.3f} m < required {required_reach + 0.02:.3f} m")
    return design, issues


def _required_reach_for_task(task) -> float:
    """Farthest reachable-workspace block distance from the shoulder (geometric, no physics)."""
    try:
        from virturoid.services.scene_generator import generate_scene_set
    except Exception:  # noqa: BLE001
        return 0.46
    shoulder_z = 0.165
    farthest = 0.0
    for purpose, count in (("variation", 8), ("holdout", 5)):
        try:
            scenes = generate_scene_set(task, count=count, purpose=purpose).scenes
        except Exception:  # noqa: BLE001
            continue
        for scene in scenes:
            for obj in scene.objects:
                if obj.object_type in {"cube", "box"}:
                    x, y, z = obj.pose_xyz_rpy[0], obj.pose_xyz_rpy[1], obj.pose_xyz_rpy[2]
                    farthest = max(farthest, math.hypot(math.hypot(x, y), z - shoulder_z))
    return round(farthest or 0.46, 4)
