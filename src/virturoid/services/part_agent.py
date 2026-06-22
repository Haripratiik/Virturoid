"""Part Recommender agent (plan §8, Phase C): LLM selects parts; specs dispose.

A high-volume specialist agent (routed to the local Nemotron-class model). It picks
a component for each robot role from the catalog, and every choice is validated by
hard compatibility checks — actuator torque vs. the task's required torque, sensor
type vs. the requirement, voltage-rail overlap, and category match. Infeasible picks
are fed back to the model (self-repair). With no LLM backend it returns None and the
caller falls back to the deterministic ``part_resolver`` — offline-safe.

A production build swaps ``extended_component_library()`` for a live supplier catalog
behind the same ``Component`` schema; the validation gate is unchanged.
"""

from __future__ import annotations

from virturoid.schemas.components import Component, ComponentCategory
from virturoid.services.part_resolver import estimate_required_actuator_torque_nm

_ROLE_CATEGORY = {
    "joint_actuator": ComponentCategory.ACTUATOR,
    "wrist_sensor": ComponentCategory.SENSOR,
    "gripper": ComponentCategory.GRIPPER,
    "main_compute": ComponentCategory.COMPUTE,
}

PART_SCHEMA = {
    "type": "object",
    "properties": {
        "joint_actuator": {"type": "string"},
        "wrist_sensor": {"type": "string"},
        "gripper": {"type": "string"},
        "main_compute": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["joint_actuator", "wrist_sensor", "gripper", "main_compute"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's part recommender. Choose one catalog component for each robot role "
    "(joint_actuator, wrist_sensor, gripper, main_compute) by its exact component id. Output ONLY "
    "JSON with those four ids and a short rationale. Rules: the joint_actuator's stall torque must "
    "exceed the required shoulder torque; the wrist_sensor must match the requested sensor type; "
    "prefer the lightest/cheapest part that satisfies the requirement; pick parts whose voltage rails "
    "are compatible. A validator will check your choices and may ask you to correct them."
)


def recommend_parts(prompt: str, requirements, components: list[Component], llm, max_repairs: int = 2) -> dict | None:
    """Recommend a component per role; validate + self-repair. None if no backend."""
    if llm is None:
        return None

    by_id = {c.id: c for c in components}
    required_torque = estimate_required_actuator_torque_nm(requirements)
    sensor_req = " ".join(requirements.sensor_requirements).lower() if requirements.sensor_requirements else ""
    user = (
        f"Task: {prompt}\n"
        f"Required shoulder torque: {required_torque:.2f} Nm. Requested sensor: {sensor_req or 'any camera'}.\n"
        f"Catalog:\n{_catalog_listing(components)}\n"
        "Return JSON: {joint_actuator, wrist_sensor, gripper, main_compute, rationale}."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, PART_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"selection": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "feasible": False, "error": str(exc)}
        selection, issues = _validate(raw, by_id, required_torque, sensor_req)
        if not issues:
            return {"selection": selection, "backend": getattr(llm, "name", "?"), "attempts": attempts,
                    "feasible": True, "rationale": raw.get("rationale", "")}
        last_issues = issues
        user = f"{user}\n\nYour previous choices {selection or raw} were rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"selection": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "feasible": False, "issues": last_issues}


def _validate(raw: dict, by_id: dict, required_torque: float, sensor_req: str) -> tuple[dict, list[str]]:
    issues: list[str] = []
    selection: dict = {}
    for role, category in _ROLE_CATEGORY.items():
        cid = raw.get(role)
        comp = by_id.get(cid) if cid else None
        if comp is None:
            issues.append(f"{role}: unknown component id {cid!r}")
            continue
        if comp.category != category:
            issues.append(f"{role}: {cid} is a {comp.category.value}, expected {category.value}")
            continue
        selection[role] = cid
        if role == "joint_actuator" and comp.actuator and comp.actuator.stall_torque_nm is not None:
            if comp.actuator.stall_torque_nm < required_torque:
                issues.append(f"joint_actuator {cid} stall {comp.actuator.stall_torque_nm} Nm < required {required_torque:.2f} Nm")
        if role == "wrist_sensor" and comp.sensor and sensor_req:
            want = "lidar" if "lidar" in sensor_req else ("rgbd_camera" if ("rgbd" in sensor_req or "depth" in sensor_req) else None)
            if want and comp.sensor.sensor_type != want:
                issues.append(f"wrist_sensor {cid} is {comp.sensor.sensor_type}, requested {want}")
    # Voltage-rail sanity: actuator and compute must share an overlapping range.
    act = by_id.get(selection.get("joint_actuator"))
    comp_c = by_id.get(selection.get("main_compute"))
    if act and comp_c and not _ranges_overlap(act.voltage_range, comp_c.voltage_range):
        issues.append(f"actuator {act.voltage_range}V and compute {comp_c.voltage_range}V rails do not overlap")
    return selection, issues


def _ranges_overlap(a, b) -> bool:
    if not a or not b:
        return True
    return a[0] <= b[1] and b[0] <= a[1]


def _catalog_listing(components: list[Component]) -> str:
    lines = []
    for c in components:
        spec = ""
        if c.actuator:
            spec = f"stall={c.actuator.stall_torque_nm}Nm rated={c.actuator.rated_torque_nm}Nm"
        elif c.sensor:
            spec = f"type={c.sensor.sensor_type} range={c.sensor.range_m}"
        lines.append(f"- {c.id} | {c.category.value} | {c.normalized_name} | {spec} | {c.voltage_range}V | {c.mass_kg}kg")
    return "\n".join(lines)
