"""AI-native EDIT OPERATORS (docs/ai_native_plan.md P1) — typed, LOCALIZED, gated mutations over a robot gene.

The proven pattern (CAD-Assistant arXiv:2412.13810 / LLM4CAD-Editor / Zoo edit-endpoint): a small semantic
request ("make it taller", "give it longer legs", "make it carbon-fiber") becomes a TYPED operator over the
existing design — never a regeneration. Each op: deep-copy -> mutate only the matched segments -> re-derive
masses/BOM (``ground_gene``) -> run the validity gate -> return a DIFF of exactly what changed. Everything
else is preserved byte-for-byte, so the edit is reviewable and one-undo-able (session_state owns the ring).

MEASURED (docs/ai_native_plan.md §0 probe): scale legs length x1.2 -> standing height 0.329->0.391 (+19%),
segment identity preserved, gene valid, appendage discovery unchanged. Ops return ``(new_gene, diff)`` or
raise ``EditError`` with a TEACHING message (SWE-agent ACI: the error tells the agent how to fix it).
"""
from __future__ import annotations


class EditError(Exception):
    """A localized edit could not be applied (bad group / factor / it broke a validity gate). Message teaches."""


def _num(value, name, *, cast=float):
    """Coerce a user-supplied edit arg to a number with a TEACHING error, never a raw ValueError. A tester who
    passes {factor:'big'} should read 'factor must be a number', not a Python stack-trace string."""
    try:
        return cast(value)
    except (TypeError, ValueError):
        raise EditError(f"{name} must be a number, got {value!r}")


# group name -> substrings that identify its segments (robust across the product's builders; a bespoke LLM gene
# still names limbs recognizably). "all" matches everything.
_GROUP_WORDS = {
    "legs": ("leg", "thigh", "shank", "femur", "tibia", "shin", "calf", "coxa"),
    "arms": ("arm", "forearm", "upper_arm", "wrist", "elbow", "shoulder"),
    "torso": ("torso", "body", "trunk", "chest", "abdomen", "spine", "base_link"),
    "head": ("head", "skull", "snout", "cranium"),
    "neck": ("neck",),
    "tail": ("tail",),
    "feet": ("foot", "feet", "toe"),
}
_DIMS = ("length", "girth", "both")


def _clone(gene):
    from virturoid.schemas.gene import RobotGene
    return RobotGene.from_dict(gene.to_dict())


def _dominant_material(gene) -> str:
    from collections import Counter
    mats = Counter((s.material or "") for s in gene.segments if s.material)
    return (mats.most_common(1)[0][0] if mats else "") or "aluminum"


def segments_for_group(gene, group: str) -> list:
    """The segments a group name refers to. 'all' -> every segment. Unknown -> EditError listing valid groups."""
    group = (group or "").lower().strip()
    if group == "all":
        return list(gene.segments)
    words = _GROUP_WORDS.get(group)
    if words is None:
        raise EditError(f"unknown group '{group}'; valid groups: {sorted(_GROUP_WORDS) + ['all']}")
    return [s for s in gene.segments if any(w in (s.name or "").lower() for w in words)]


def _standing_height(gene) -> float:
    try:
        from virturoid.services.gene_compiler import standing_spawn_z
        return round(float(standing_spawn_z(gene)), 4)
    except Exception:  # noqa: BLE001
        return 0.0


def _reground_and_gate(gene, *, material: str):
    """Re-derive masses/BOM for the mutated geometry, then GATE: the gene must still validate. Teaches on fail."""
    from virturoid.services.grounded_physics import ground_gene
    try:
        ground_gene(gene, material=material, fill=0.25)
    except Exception:  # noqa: BLE001 - grounding is value-add; a mutated gene can still be scored/rendered
        pass
    issues = gene.validate()
    if issues:
        raise EditError(f"edit would make the robot invalid ({'; '.join(issues[:2])}); "
                        "try a smaller factor or a different group")


def scale_group(gene, *, group: str = "legs", dims: str = "length", factor: float = 1.2):
    """LENGTHEN / THICKEN a group of segments by ``factor`` (dims: length | girth | both). The workhorse:
    'make it taller' -> scale_group(legs, length, ~1.2). Only the matched segments change; mass/BOM re-derive."""
    if dims not in _DIMS:
        raise EditError(f"dims must be one of {_DIMS}, got '{dims}'")
    f = _num(factor, "factor")
    if not (0.2 <= f <= 5.0):
        raise EditError(f"factor {factor} out of the safe range [0.2, 5.0]")
    g = _clone(gene)
    targets = segments_for_group(g, group)
    if not targets:
        raise EditError(f"no '{group}' segments on this robot (it is a {g.robot_class}); "
                        f"available groups here: {[k for k in _GROUP_WORDS if segments_for_group(g, k)] + ['all']}")
    from virturoid.services.bom_builder import _scale_geo
    changed = []
    for s in targets:
        before = (round(s.length_m, 4), round(s.radius_m, 4))
        if dims in ("length", "both"):
            s.length_m = round(s.length_m * f, 5)
        if dims in ("girth", "both"):
            s.radius_m = round(s.radius_m * f, 5)
        if dims == "both" and s.geometry:
            _scale_geo(s.geometry, f)
        changed.append({"segment": s.name, "length_m": [before[0], round(s.length_m, 4)],
                        "radius_m": [before[1], round(s.radius_m, 4)]})
    h0 = _standing_height(gene)
    _reground_and_gate(g, material=_dominant_material(gene))
    h1 = _standing_height(g)
    diff = {"op": "scale_group", "group": group, "dims": dims, "factor": round(f, 3),
            "n_changed": len(changed), "changed": changed[:8], "n_segments_total": len(g.segments),
            "standing_height_m": [h0, h1]}
    return g, diff


def set_height(gene, *, target_m: float):
    """Make the robot stand at ~``target_m`` by scaling the LEGS (the height-bearing group). Solves the factor
    from the current standing height, then defers to scale_group so the same gate/diff apply."""
    tm = _num(target_m, "target_m")
    if not (0.05 <= tm <= 5.0):
        raise EditError(f"target height {target_m} m is implausible (expected 0.05-5.0 m)")
    cur = _standing_height(gene)
    if cur <= 1e-3:
        raise EditError("cannot measure the robot's current height to scale toward a target")
    factor = tm / cur
    legs = segments_for_group(gene, "legs")
    return scale_group(gene, group=("legs" if legs else "all"), dims="length", factor=factor)


def scale_robot(gene, *, factor: float = 1.2):
    """Uniformly scale the WHOLE robot (all segments, both dims) by ``factor`` — a bigger/smaller version."""
    return scale_group(gene, group="all", dims="both", factor=factor)


def set_material(gene, *, group: str = "all", material: str = "aluminum"):
    """Change the (render/BOM) material of a group of segments; re-derives mass at the new density."""
    known = ("steel", "aluminum", "carbon_fiber", "titanium", "abs_plastic", "shell", "metal", "skeleton", "frame")
    if material not in known:
        raise EditError(f"unknown material '{material}'; known: {known}")
    g = _clone(gene)
    targets = segments_for_group(g, group)
    if not targets:
        raise EditError(f"no '{group}' segments to re-material on this {g.robot_class}")
    for s in targets:
        s.material = material
    _reground_and_gate(g, material=material)
    return g, {"op": "set_material", "group": group, "material": material, "n_changed": len(targets)}


def set_leg_count(gene, *, n_pairs: int):
    """STRUCTURAL edit (confirm-gated by the caller): rebuild the body as an N-pair legged creature. This
    changes topology, so it is a bigger edit than a parameter tweak — flagged ``structural`` in the diff."""
    n = _num(n_pairs, "n_pairs", cast=int)
    if not (1 <= n <= 8):
        raise EditError(f"n_pairs {n_pairs} out of range [1, 8]")
    from virturoid.services.anatomy_compiler import _generic_legged_graph, build_from_anatomy
    old_legs = len({s.name.rsplit("_", 1)[0] for s in segments_for_group(gene, "legs")}) or 0
    g = build_from_anatomy(_generic_legged_graph(n_pairs=n))
    if g is None:
        raise EditError("could not build a body at that leg count")
    _reground_and_gate(g, material=_dominant_material(gene))
    return g, {"op": "set_leg_count", "n_pairs": n, "structural": True,
               "note": "topology changed (legs rebuilt); torso/appendage customization not carried over"}


def set_payload(gene, *, payload_kg: float = 2.0, girth_scale: bool = True):
    """CAPABILITY amend: make the robot CARRY / LIFT a target payload by upsizing its actuators (and, optionally,
    the load-path link girth) so its joints can hold the extra load. The actuated joints must support the robot's
    own weight PLUS the payload, so each joint's required torque is scaled by ``(total_mass + payload)/total_mass``;
    re-grounding then selects BIGGER real motors for that torque and the BOM/mass rise honestly (a stronger robot
    costs more + weighs more). This is what turns 'make it lift 10 kg' into a real amendment, not just a wish."""
    pk = _num(payload_kg, "payload_kg")
    if not (0.1 <= pk <= 50.0):
        raise EditError(f"payload {payload_kg} kg out of the safe range [0.1, 50.0]; "
                        "for heavier loads redesign with a larger actuator class")
    g = _clone(gene)
    actuated = [s for s in g.segments if s.joint_type in ("revolute", "prismatic")]
    if not actuated:
        raise EditError("this robot has no actuated joints, so it cannot be amended to carry a payload")
    total_mass = sum(float(s.mass_kg or 0.0) for s in g.segments) or 1.0
    load_factor = (total_mass + pk) / total_mass
    changed = []
    for s in actuated:                                          # scale the joint's torque REQUIREMENT (not the last
        req0 = s.torque_req_nm if s.torque_req_nm is not None else abs(s.actuator_torque_nm or 8.0)
        before = round(float(req0), 2)                          # motor's peak) so re-grounding upsizes the motor and
        s.torque_req_nm = round(before * load_factor, 2)        # stays idempotent under repeated grounding
        s.actuator_torque_nm = s.torque_req_nm                  # keep them consistent until ground re-selects the peak
        changed.append({"segment": s.name, "required_nm": [before, round(s.torque_req_nm, 2)]})
    if girth_scale and load_factor > 1.05:                      # thicken the load-path limbs (sub-linear) for rigidity
        girth_mult = min(1.4, float(load_factor) ** 0.3)
        for s in actuated:
            if s.radius_m and s.radius_m > 0:
                s.radius_m = round(max(0.005, s.radius_m * girth_mult), 5)
    required_by_name = {c["segment"]: c["required_nm"][1] for c in changed}
    mass0 = round(total_mass, 3)
    _reground_and_gate(g, material=_dominant_material(gene))    # upsizes actuators for the new torque; re-derives mass
    mass1 = round(sum(float(s.mass_kg or 0.0) for s in g.segments), 3)
    # HONEST saturation check: if a joint's chosen real motor can't actually meet the scaled requirement, the
    # payload exceeds what the actuator catalog can drive -- say so (a gearbox / bigger class is needed) rather
    # than pretending the maxed-out motor is enough.
    undersized = []
    for s in g.segments:
        req = required_by_name.get(s.name)
        if req is not None and float(s.actuator_torque_nm or 0.0) + 1e-6 < float(req):
            undersized.append({"segment": s.name, "required_nm": round(float(req), 1),
                               "best_motor_nm": round(float(s.actuator_torque_nm or 0.0), 1)})
    out = {"op": "set_payload", "payload_kg": pk, "load_factor": round(load_factor, 3),
           "n_joints_upsized": len(changed), "changed": changed[:8], "total_mass_kg": [mass0, mass1],
           "note": "joint torque raised for the payload -> re-grounding upsized the real motors (mass/cost rose)"}
    if undersized:
        out["undersized_joints"] = undersized[:8]
        out["warning"] = (f"{len(undersized)} joint(s) exceed the strongest catalog motor for this payload -- add a "
                          "gearbox / reduction stage or split the load; the design is over the actuator envelope here")
    return g, out


# op name -> callable(gene, **args). The typed operator library the intent-classifier maps requests onto.
OPERATORS = {
    "scale_group": scale_group,
    "set_height": set_height,
    "scale_robot": scale_robot,
    "set_material": set_material,
    "set_leg_count": set_leg_count,
    "set_payload": set_payload,
}
_STRUCTURAL = {"set_leg_count"}


def op_specs() -> list[dict]:
    """JSON-schema-ish specs of the edit operators (for the intent classifier + tool docs)."""
    return [
        {"op": "scale_group", "args": {"group": list(_GROUP_WORDS) + ["all"], "dims": list(_DIMS), "factor": "0.2-5.0"},
         "for": "lengthen/thicken part of the robot, e.g. taller = scale_group legs length 1.2"},
        {"op": "set_height", "args": {"target_m": "0.05-5.0"}, "for": "stand at a specific height"},
        {"op": "scale_robot", "args": {"factor": "0.2-5.0"}, "for": "make the whole robot bigger/smaller"},
        {"op": "set_material", "args": {"group": list(_GROUP_WORDS) + ["all"], "material": "steel|aluminum|carbon_fiber|titanium|..."},
         "for": "change what a part is made of"},
        {"op": "set_leg_count", "args": {"n_pairs": "1-8"}, "for": "STRUCTURAL: change how many legs (rebuilds)"},
        {"op": "set_payload", "args": {"payload_kg": "0.1-50.0", "girth_scale": "true|false"},
         "for": "make it CARRY/LIFT heavier: upsize actuators (+ load-path girth) for the payload; BOM/mass rise"},
    ]


def apply_op(gene, op: str, args: dict | None = None):
    """Apply one typed operator. Returns ``(new_gene, diff)`` or raises ``EditError``."""
    fn = OPERATORS.get(op)
    if fn is None:
        raise EditError(f"unknown edit op '{op}'; valid ops: {sorted(OPERATORS)}")
    new_gene, diff = fn(gene, **(args or {}))
    diff["structural"] = op in _STRUCTURAL or diff.get("structural", False)
    return new_gene, diff


def apply_ops(gene, ops: list[dict]):
    """Apply a sequence of ops, threading the gene. Returns ``(final_gene, [diffs])``; the FIRST failure raises
    (nothing is committed by this module — the caller commits the returned gene as one undo step)."""
    g = gene
    diffs = []
    for spec in (ops or []):
        g, d = apply_op(g, str(spec.get("op")), spec.get("args") or {})
        diffs.append(d)
    return g, diffs
