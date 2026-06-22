"""STAGE 2 — physically GROUND a procedurally-composed body instead of using guessed scalars.

The procedural composer (the generative B-path) sets every link to a flat 0.3 kg and every joint to a
constant torque. This module replaces those guesses with grounded numbers:

  • link structural mass  = primitive volume x material density x hollow fill-fraction (real links are
    shells, not solid), so a longer/fatter link is genuinely heavier; and
  • each actuated joint    is matched to a REAL actuator from a small catalog (real datasheet mass +
    stall torque) whose stall torque covers the joint's required torque with margin — the actuator's
    mass is added to its link and its real stall torque becomes the joint limit.

It returns the grounded gene + a real Bill of Materials (the actuators actually used). This is what makes
the dynamics — and the parts list — real rather than plausible-looking.
"""

from __future__ import annotations

import math

# material density (kg/m^3) — real values for common robot-structure materials
MATERIALS: dict[str, float] = {
    "aluminum": 2700.0, "steel": 7850.0, "abs": 1040.0, "carbon_fiber": 1600.0, "titanium": 4500.0,
}

# REAL, off-the-shelf actuators — the supplier-backed "part candidates" of report 7's real-part entity model.
# ONE source of truth: ``component_catalog.ACTUATORS`` (the same 12-part ladder the BOM specs from). We derive a
# dict view of it here, sorted ascending by torque, so _pick_actuator picks the smallest that fits and the
# RENDERED motor housing (component_geometry reads these dicts) cites the EXACT part the BOM lists. Each row
# carries the manufacturer datasheet a build can cite: part number, mass, stall torque, the physical ENVELOPE
# (bounding box x/y/z in metres), the housing SHAPE (servos are boxes; quasi-direct-drive motors are pancake
# cylinders), and a supplier reference. Genuine off-the-shelf COMPONENTS (a servo you can buy) — NOT a real
# robot's body shell; using them is grounding, not the forbidden design-copying ([[original-generation-mandate]]).
from virturoid.services.component_catalog import ACTUATORS as _CATALOG_ACTUATORS

ACTUATOR_CATALOG: list[dict] = [
    {"part": a.name, "mass_kg": a.mass_kg, "stall_nm": a.peak_torque_nm,
     "shape": a.shape, "envelope_m": a.envelope_m, "axis_dim": a.axis_dim,
     "supplier": a.vendor, "voltage_v": a.voltage_v}
    for a in sorted(_CATALOG_ACTUATORS, key=lambda a: a.peak_torque_nm)
]
# Backward-compatible (name, mass_kg, stall_nm) view — derived from the catalog so there is ONE source of truth.
ACTUATORS: list[tuple[str, float, float]] = [(a["part"], a["mass_kg"], a["stall_nm"]) for a in ACTUATOR_CATALOG]


def _link_volume_m3(shape: str, length_m: float, radius_m: float) -> float:
    """Solid volume of the link's primitive (capsule = cylinder + 2 hemispheres; box uses radius as
    half-width in x/y; cylinder = disc x length)."""
    r, L = max(1e-4, radius_m), max(1e-4, length_m)
    if shape == "box":
        return (2 * r) * (2 * r) * L
    if shape == "cylinder":
        return math.pi * r * r * L
    return math.pi * r * r * L + (4.0 / 3.0) * math.pi * r ** 3   # capsule


def _pick_actuator(required_nm: float) -> tuple[str, float, float]:
    for name, mass, stall in ACTUATORS:
        if stall >= required_nm:
            return name, mass, stall
    return ACTUATORS[-1]


def ground_gene(gene, *, material: str = "aluminum", fill: float = 0.3, margin: float = 1.3) -> dict:
    """Mutate ``gene`` in place: set each link's mass from material+geometry (+ its actuator's mass) and
    each actuated joint's torque to a real actuator's stall torque. Returns ``{material, bom, total_mass_kg,
    actuator_count}`` — ``bom`` is the real parts list."""
    density = MATERIALS.get(material, MATERIALS["aluminum"])
    bom: list[dict] = []
    total = 0.0
    for s in gene.segments:
        struct = _link_volume_m3(s.shape, s.length_m, s.radius_m) * density * fill
        act_mass = 0.0
        if s.joint_type in ("revolute", "prismatic"):
            required = abs(s.actuator_torque_nm or 8.0) * margin
            name, act_mass, stall = _pick_actuator(required)
            s.actuator_torque_nm = stall                       # real stall torque drives the joint
            bom.append({"role": s.name, "part": name, "mass_kg": act_mass, "stall_nm": stall,
                        "required_nm": round(required, 2)})
        s.mass_kg = round(max(0.02, struct + act_mass), 3)     # grounded mass = structure + actuator
        total += s.mass_kg
    return {"material": material, "bom": bom, "total_mass_kg": round(total, 3),
            "actuator_count": len(bom)}
