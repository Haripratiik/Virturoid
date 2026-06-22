"""Scene Agent (plan §8, Phase C): Task Graph → scene-family randomization params.

A high-volume specialist agent (routed to the local Nemotron-class model) that
proposes the domain-randomization envelope for a scene family — how many scenes,
how far to jitter object positions, mass/friction ranges, materials — rather than
raw poses (that stays the deterministic generator's job). Every proposal is
validated against hard scene-feasibility gates: objects must stay inside the arm's
reachable tabletop annulus, masses must be graspable and positive, and a minimum
object separation prevents interpenetration. Infeasible envelopes are fed back to
the model (self-repair).

With no LLM backend it returns None and the caller uses the deterministic
``scene_generator`` envelope — offline-safe.
"""

from __future__ import annotations

from virturoid.services.scene_generator import _REACH_MAX_M, _REACH_MIN_M

# Physics-grounded bounds the proposal must respect.
_MAX_SCENES = 50
_MAX_GRASP_MASS_KG = 0.5      # the reference gripper/arm payload ceiling
_MIN_OBJECT_SEP_M = 0.04      # ~one cube width; below this objects interpenetrate

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_count": {"type": "integer"},
        "base_radius_m": {"type": "number"},
        "position_spread_m": {"type": "number"},
        "mass_range_kg": {"type": "array", "items": {"type": "number"}},
        "friction_range": {"type": "array", "items": {"type": "number"}},
        "min_separation_m": {"type": "number"},
        "materials": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["scene_count", "base_radius_m", "position_spread_m", "mass_range_kg", "min_separation_m"],
    "additionalProperties": False,
}

_SYSTEM = (
    "You are Virturoid's scene designer. Given a task, propose the domain-randomization "
    "envelope for a family of pick-and-place scenes (NOT raw object poses). Output ONLY JSON: "
    "scene_count, base_radius_m (nominal distance of objects from the arm base), "
    "position_spread_m (jitter radius), mass_range_kg [min,max], friction_range [min,max], "
    "min_separation_m, materials, rationale. Hard constraints, enforced by a validator: every "
    f"object must stay reachable, i.e. base_radius_m +/- position_spread_m within "
    f"[{_REACH_MIN_M}, {_REACH_MAX_M}] m; masses positive and <= {_MAX_GRASP_MASS_KG} kg; "
    f"min_separation_m >= {_MIN_OBJECT_SEP_M} m to avoid interpenetration."
)


def propose_scene_envelope(prompt: str, requirements, llm, max_repairs: int = 2) -> dict | None:
    """Propose a randomization envelope; validate + self-repair. None if no backend."""
    if llm is None:
        return None

    user = (
        f"Task: {prompt}\n"
        f"Reachable annulus: [{_REACH_MIN_M}, {_REACH_MAX_M}] m from the base. "
        f"Grasp payload ceiling: {_MAX_GRASP_MASS_KG} kg.\n"
        "Return JSON: {scene_count, base_radius_m, position_spread_m, mass_range_kg, "
        "friction_range, min_separation_m, materials, rationale}."
    )

    attempts = 0
    last_issues: list[str] = []
    for _ in range(max_repairs + 1):
        attempts += 1
        try:
            raw = llm.complete_json(_SYSTEM, user, SCENE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return {"envelope": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "feasible": False, "error": str(exc)}
        issues = _validate(raw)
        if not issues:
            return {"envelope": _normalize(raw), "backend": getattr(llm, "name", "?"), "attempts": attempts,
                    "feasible": True, "rationale": raw.get("rationale", "")}
        last_issues = issues
        user = f"{user}\n\nYour previous envelope {raw} was rejected: {'; '.join(issues)}. Return corrected JSON."

    return {"envelope": None, "backend": getattr(llm, "name", "?"), "attempts": attempts, "feasible": False, "issues": last_issues}


def _validate(raw: dict) -> list[str]:
    issues: list[str] = []
    n = raw.get("scene_count")
    if not isinstance(n, int) or not (1 <= n <= _MAX_SCENES):
        issues.append(f"scene_count must be an integer in [1, {_MAX_SCENES}]")
    base = raw.get("base_radius_m")
    spread = raw.get("position_spread_m")
    if not isinstance(base, (int, float)) or not isinstance(spread, (int, float)) or spread < 0:
        issues.append("base_radius_m and position_spread_m must be non-negative numbers")
    else:
        if base - spread < _REACH_MIN_M:
            issues.append(f"base_radius_m - spread = {base - spread:.3f} < reach min {_REACH_MIN_M} (objects too close)")
        if base + spread > _REACH_MAX_M:
            issues.append(f"base_radius_m + spread = {base + spread:.3f} > reach max {_REACH_MAX_M} (objects out of reach)")
    mr = raw.get("mass_range_kg") or []
    if len(mr) != 2 or mr[0] <= 0 or mr[1] <= 0 or mr[0] > mr[1]:
        issues.append("mass_range_kg must be [min,max] with 0 < min <= max")
    elif mr[1] > _MAX_GRASP_MASS_KG:
        issues.append(f"mass_range_kg max {mr[1]} exceeds grasp ceiling {_MAX_GRASP_MASS_KG} kg")
    fr = raw.get("friction_range")
    if fr is not None and (len(fr) != 2 or fr[0] <= 0 or fr[0] > fr[1]):
        issues.append("friction_range must be [min,max] with 0 < min <= max")
    sep = raw.get("min_separation_m")
    if not isinstance(sep, (int, float)) or sep < _MIN_OBJECT_SEP_M:
        issues.append(f"min_separation_m must be >= {_MIN_OBJECT_SEP_M} m to avoid interpenetration")
    return issues


def _normalize(raw: dict) -> dict:
    return {
        "scene_count": int(raw["scene_count"]),
        "base_radius_m": float(raw["base_radius_m"]),
        "position_spread_m": float(raw["position_spread_m"]),
        "mass_range_kg": [float(raw["mass_range_kg"][0]), float(raw["mass_range_kg"][1])],
        "friction_range": [float(raw["friction_range"][0]), float(raw["friction_range"][1])] if raw.get("friction_range") else [0.8, 1.3],
        "min_separation_m": float(raw["min_separation_m"]),
        "materials": list(raw.get("materials", []) or ["plastic", "wood", "metal"]),
    }
