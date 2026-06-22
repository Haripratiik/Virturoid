"""Buildability report (plan: roadmap_beyond_phase3 §3/§4) — "plausible gene" -> "buildable design".

The research's credible product wedge is a **sim-validated, buildable design** with an honest
confidence score — not a promise that a novel body transfers to reality. This assembles that:

1. **Parts grounding** — every actuated joint's required torque must be coverable by a REAL
   actuator in the parts library (you can't ship a joint no buyable motor can drive). Each joint
   is matched to the smallest sufficient actuator → an actuator BOM. Flags impossible joints.
2. **Structural feasibility** — links survive their loads (delegates to ``structural_validation``).
3. **Transferability / confidence score** — a 0..1 heuristic combining parts grounding, structural
   safety margin, and (if available) measured sim success. It is explicitly a CONFIDENCE, not a
   transfer guarantee — for a novel morphology you cannot calibrate, the honest move is to score
   and flag risk (Koos/Mouret/Doncieux transferability), never to claim "works in reality."
"""

from __future__ import annotations

from virturoid.schemas.gene import RobotGene
from virturoid.services.structural_validation import validate_structure


def _actuators(components):
    return [c for c in components if getattr(c, "actuator", None) and c.actuator.stall_torque_nm]


def ground_actuators(gene: RobotGene, components) -> dict:
    """Match each actuated joint to the smallest real actuator that covers its torque."""
    acts = sorted(_actuators(components), key=lambda c: c.actuator.stall_torque_nm)
    max_stall = acts[-1].actuator.stall_torque_nm if acts else 0.0
    rows, issues = [], []
    for s in gene.actuated_joints():
        need = float(s.actuator_torque_nm or 0.0)
        fit = next((c for c in acts if c.actuator.stall_torque_nm >= need), None)
        if fit is None:
            issues.append(f"joint {s.name!r} needs {need:.1f} Nm but the strongest actuator is {max_stall:.1f} Nm")
            rows.append({"joint": s.name, "need_nm": round(need, 2), "actuator": None, "stall_nm": None})
        else:
            # continuous (rated) torque should ideally also cover the need; note if it's marginal.
            rated = fit.actuator.rated_torque_nm
            rows.append({"joint": s.name, "need_nm": round(need, 2), "actuator": fit.id,
                         "stall_nm": fit.actuator.stall_torque_nm, "rated_nm": rated,
                         "continuous_ok": (rated is None or rated >= need)})
    return {"grounded": not issues, "joint_actuators": rows, "issues": issues}


def transferability_score(parts_grounded: bool, structural: dict, sim_success: float | None) -> float:
    """0..1 confidence the design is buildable + likely to behave as simulated. Heuristic, honest.

    Combines: parts grounding (hard gate-ish), structural safety margin (capped), and measured sim
    success if available. NOT a guarantee of real-world transfer (novel bodies can't be calibrated).
    """
    parts = 1.0 if parts_grounded else 0.3
    sf = structural.get("min_safety_factor", 0.0)
    struct = max(0.0, min(1.0, sf / 3.0))          # SF>=3 -> full marks; <1 -> very low
    sim = 1.0 if sim_success is None else max(0.0, min(1.0, float(sim_success)))
    # Weighted: structure + parts are buildability; sim is behavioral confidence.
    return round(0.4 * parts + 0.35 * struct + 0.25 * sim, 3)


def assess_buildability(gene: RobotGene, components, *, material: str = "aluminum",
                        payload_kg: float = 0.0, sim_success: float | None = None) -> dict:
    """Full buildable-design assessment for a gene. Honest confidence, not a transfer guarantee."""
    parts = ground_actuators(gene, components)
    structural = validate_structure(gene, material=material, payload_kg=payload_kg)
    confidence = transferability_score(parts["grounded"], structural, sim_success)
    issues = list(parts["issues"]) + list(structural.get("issues", []))
    return {
        "buildable": bool(parts["grounded"] and structural["ok"]),
        "confidence": confidence,                    # 0..1, honest heuristic
        "parts": parts,                              # actuator BOM + grounding
        "structural": {k: structural[k] for k in ("ok", "material", "min_safety_factor")},
        "structural_links": structural.get("links", []),
        "sim_success": sim_success,
        "issues": issues,
        "disclaimer": "Confidence is a buildability + sim-behavior heuristic, NOT a guarantee of "
                      "real-world transfer; a novel morphology cannot be calibrated until built.",
    }
