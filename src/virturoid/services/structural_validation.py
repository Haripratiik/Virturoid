"""Structural validation (roadmap_beyond_phase3 Part B, rung a): turn a *plausible* gene
into a *buildable* design by flagging links that would physically fail under load.

Per the roadmap this is the highest-ROI realism upgrade in our artifacts: we already
have parts-grounded morphology + BOM + kinematics, but nothing checks whether a proposed
link's cross-section and material can actually carry the loads the robot imposes. This
module adds that gate — "LLM proposes, physics disposes", same spirit as
``RobotGene.validate`` (kinematic feasibility) and ``scene_agent`` (scene feasibility).

WHAT THIS IS (and is NOT): a transparent *engineering estimate*, NOT finite-element
analysis. We model each link as a **cantilever beam** built into its parent joint and
loaded at its tip by everything downstream of it (distal links + payload). For each link
we take the worst-case static bending moment to be the LARGER of:

  * the gravity load of everything downstream acting at a conservative moment arm
    (the cumulative downstream length — i.e. the arm fully extended horizontally), and
  * the actuator torque driving that joint (a stalled motor against a hard stop loads the
    link it drives just as a tip force would).

From that moment ``M`` we compute peak bending stress ``sigma = M*c / I`` using the link's
own cross-section (second moment of area ``I`` and extreme-fibre distance ``c``), then a
safety factor ``SF = yield_strength / sigma``. A link is flagged infeasible if
``SF < safety_factor_min``.

HONEST LIMITATIONS — do not oversell this:
  * Static only: no dynamics, inertia, vibration, fatigue, buckling, or stress
    concentrations (joints/holes/fillets are where real parts actually fail).
  * Bending only: shear, torsion, and combined/axial loading are ignored.
  * Worst-case-fully-extended geometry and stalled-actuator torque are deliberately
    conservative; the real peak depends on pose and trajectory.
  * Idealised solid sections (a real link is hollow/ribbed); ``radius_m`` is reused as the
    box half-width / tube radius. Treat the result as a *screening* gate and a confidence
    signal for the design report, not a substitute for FEA or a physical test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Material yield strengths in Pa (order-of-magnitude engineering values for the alloys/
# plastics a buildable robot link is plausibly made from). Conservative-ish nominal yields.
MATERIALS_PA: dict[str, float] = {
    "aluminum": 270e6,      # 6061-T6 ~ 240-280 MPa
    "steel": 350e6,         # mild structural steel ~ 250-400 MPa
    "pla": 50e6,            # 3D-printed PLA ~ 40-60 MPa
    "abs": 40e6,            # 3D-printed ABS ~ 30-45 MPa
    "carbon_fiber": 600e6,  # CFRP laminate (tensile, very layup-dependent)
}

DEFAULT_MATERIAL = "aluminum"


@dataclass
class LinkResult:
    """Per-link structural verdict (also exported as a plain dict in the report)."""
    name: str
    stress_pa: float        # peak bending stress sigma = M*c/I
    safety_factor: float    # yield_strength / sigma (inf if effectively unloaded)
    feasible: bool          # safety_factor >= safety_factor_min
    moment_nm: float        # the worst-case bending moment used
    moment_source: str      # "actuator_torque" | "gravity_load" — which one dominated

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "stress_pa": self.stress_pa,
            "safety_factor": self.safety_factor,
            "feasible": self.feasible,
            "moment_nm": self.moment_nm,
            "moment_source": self.moment_source,
        }


def _section_properties(segment) -> tuple[float, float]:
    """Return ``(I, c)`` — second moment of area [m^4] and extreme-fibre distance [m].

    ``radius_m`` is the half-width (box) or radius (cylinder/capsule), per the gene schema.
    Box about a centroidal bending axis: I = w*h^3/12 with w = h = 2r, c = h/2 = r.
    Round section (cylinder/capsule): I = pi*r^4/4, c = r.
    """
    r = segment.radius_m
    if segment.shape == "box":
        w = h = 2.0 * r
        return (w * h ** 3) / 12.0, h / 2.0
    # cylinder | capsule -> solid circular section (capsule end-caps add negligible bending I)
    return (math.pi * r ** 4) / 4.0, r


def _downstream(gene, name: str) -> tuple[float, float]:
    """Cumulative ``(mass_kg, length_m)`` of every link strictly downstream of ``name``.

    Walks the kinematic subtree below ``name`` (not counting the link itself). The summed
    length is used as a conservative moment arm (arm fully extended); the summed mass is
    the load that link must hold. A leaf/end-effector has (0, 0) downstream — handled by
    the caller, which still loads it with the external payload.
    """
    mass = 0.0
    length = 0.0
    for child in gene.children_of(name):
        mass += child.mass_kg
        length += child.length_m
        cm, cl = _downstream(gene, child.name)
        mass += cm
        length += cl
    return mass, length


def validate_structure(
    gene,
    material: str = DEFAULT_MATERIAL,
    gravity: float = 9.81,
    payload_kg: float = 0.0,
    safety_factor_min: float = 2.0,
) -> dict:
    """Estimate whether each link of ``gene`` survives its worst-case static load.

    Cantilever-beam screening estimate (see module docstring for the model and its honest
    limitations). NOT FEA.

    Args:
        gene: a ``RobotGene`` (must be a valid kinematic tree).
        material: key into :data:`MATERIALS_PA` (unknown -> falls back to aluminum, noted
            in ``issues``).
        gravity: m/s^2.
        payload_kg: external payload carried at the end effector (adds to every link's
            downstream load).
        safety_factor_min: minimum acceptable yield/stress ratio; links below it fail.

    Returns:
        dict: ``{"ok", "material", "min_safety_factor", "links": [..], "issues": [..]}``
        where each link dict has ``name, stress_pa, safety_factor, feasible`` (plus
        ``moment_nm, moment_source``). ``ok`` is True iff every link meets
        ``safety_factor_min``.
    """
    issues: list[str] = []

    # Reuse the gene's own kinematic gate — a malformed tree can't be load-analysed.
    gene_issues = gene.validate()
    if gene_issues:
        return {
            "ok": False,
            "material": material,
            "min_safety_factor": 0.0,
            "links": [],
            "issues": [f"gene is not a valid kinematic tree: {i}" for i in gene_issues],
        }

    if material not in MATERIALS_PA:
        issues.append(
            f"unknown material {material!r}; falling back to {DEFAULT_MATERIAL!r} "
            f"(known: {sorted(MATERIALS_PA)})"
        )
        material = DEFAULT_MATERIAL
    yield_pa = MATERIALS_PA[material]

    ee = gene.end_effector()
    ee_name = ee.name if ee is not None else None

    results: list[LinkResult] = []
    for seg in gene.segments:
        down_mass, down_len = _downstream(gene, seg.name)

        # The external payload is carried at the end effector, so it loads the EE link and
        # every link on the chain above it. A link "carries" the payload iff the EE lies in
        # its downstream subtree (or it *is* the EE link itself).
        carries_payload = seg.name == ee_name or _is_ancestor_of(gene, seg.name, ee_name)
        load_mass = down_mass + (payload_kg if carries_payload else 0.0)

        # Moment arm: cumulative downstream length (fully-extended, conservative). A leaf
        # link still has its own half-length as a tiny arm for any payload it holds.
        arm = down_len if down_len > 0.0 else (seg.length_m / 2.0)
        m_gravity = load_mass * gravity * arm

        # Stalled actuator driving THIS joint loads the link just like a tip force.
        m_actuator = seg.actuator_torque_nm or 0.0

        if m_actuator >= m_gravity:
            moment, source = m_actuator, "actuator_torque"
        else:
            moment, source = m_gravity, "gravity_load"

        I, c = _section_properties(seg)
        if moment <= 0.0 or I <= 0.0:
            # Effectively unloaded (e.g. the root base with no torque and nothing pulling on
            # it sideways). Treat as safe with an unbounded safety factor.
            stress = 0.0
            sf = math.inf
        else:
            stress = moment * c / I
            sf = yield_pa / stress

        feasible = sf >= safety_factor_min
        results.append(LinkResult(seg.name, stress, sf, feasible, moment, source))

    finite_sfs = [r.safety_factor for r in results if math.isfinite(r.safety_factor)]
    min_sf = min(finite_sfs) if finite_sfs else math.inf

    for r in results:
        if not r.feasible:
            issues.append(
                f"link {r.name!r} is under-strength: safety factor {r.safety_factor:.2f} "
                f"< {safety_factor_min:.2f} (peak bending stress {r.stress_pa / 1e6:.1f} MPa "
                f"vs {material} yield {yield_pa / 1e6:.0f} MPa, driven by {r.moment_source}). "
                f"Suggestion: increase radius_m (stress falls ~1/r^3), shorten the downstream "
                f"reach, reduce the actuator torque, or use a stronger material."
            )

    return {
        "ok": all(r.feasible for r in results),
        "material": material,
        "min_safety_factor": float(min_sf),
        "links": [r.to_dict() for r in results],
        "issues": issues,
    }


def _is_ancestor_of(gene, name: str, descendant: str | None) -> bool:
    """True if ``name`` lies on the parent chain above ``descendant`` (exclusive)."""
    if descendant is None:
        return False
    node = gene.segment(descendant)
    while node is not None and node.parent is not None:
        if node.parent == name:
            return True
        node = gene.segment(node.parent)
    return False
