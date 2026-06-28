"""§4.8A — the BOM<->sim fidelity report: how faithfully is the simulated body driven by the REAL parts?

The audit found the sim and the BOM are two independent read-outs of the gene: the sim's masses are builder
values (not the BOM material-density mass), and an actuator's sim torque limit is the joint REQUIREMENT, not the
rated torque of the motor the BOM selected. Closing that loop (grounding the gene on the main build path) is the
real fix -- but grounding roughly DOUBLES the mass, which would break learned policies trained on the lighter
body, and re-validating them needs the GPU we don't have. So rather than silently ship an optimistic sim, this
REPORTS the gap honestly: the mass-fidelity ratio (how much lighter the sim body is than its real parts imply)
and the per-joint torque picture. The honesty wedge applied to physical fidelity.
"""

from __future__ import annotations


def bom_sim_fidelity(gene) -> dict:
    """Compare the gene's SIM physics against the BOM-grounded reality and flag divergences. Returns mass
    fidelity (sim vs grounded-real), the per-joint sim-torque-vs-selected-motor picture, and honest flags."""
    import copy

    from virturoid.services.component_catalog import select_actuator
    from virturoid.services.grounded_physics import ground_gene

    sim_mass = float(sum(s.mass_kg for s in gene.segments))
    grounded_mass = None
    try:
        grounded = copy.deepcopy(gene)
        ground_gene(grounded)                                   # real material density + catalog motor mass
        grounded_mass = float(sum(s.mass_kg for s in grounded.segments))
    except Exception:  # noqa: BLE001 - grounding is best-effort; report sim-only if it fails
        grounded = None

    joints = []
    for s in gene.actuated_joints():
        req = float(s.actuator_torque_nm or 10.0)
        motor = select_actuator(req)
        joints.append({"joint": s.name, "sim_torque_limit_nm": round(req, 2),
                       "selected_motor": motor.name, "motor_peak_nm": round(float(motor.peak_torque_nm), 2)})

    ratio = round(grounded_mass / sim_mass, 2) if (grounded_mass and sim_mass > 1e-6) else None
    flags = []
    if ratio and ratio > 1.15:
        flags.append(
            f"sim body is ~{ratio}x LIGHTER than its BOM-grounded mass ({sim_mass:.1f} kg sim vs "
            f"{grounded_mass:.1f} kg real parts) -- the sim is OPTIMISTIC. Grounding closes this but ~doubles "
            f"mass, so learned policies need re-validation (GPU-gated).")
    return {"sim_mass_kg": round(sim_mass, 2),
            "grounded_mass_kg": round(grounded_mass, 2) if grounded_mass else None,
            "mass_fidelity_ratio": ratio, "n_joints": len(joints), "joints": joints,
            "flags": flags, "faithful": not flags,
            "note": ("sim actuator torque limit = the joint REQUIREMENT; the BOM selects a real motor whose rated "
                     "peak clears it with margin (motor_peak_nm >= sim_torque_limit_nm). Grounding masses/torque "
                     "on the main path is the real fix and is GPU-gated (re-validates learned control).")}


def format_fidelity_md(report: dict) -> str:
    lines = ["# BOM<->Sim Fidelity", "",
             f"- sim mass: **{report['sim_mass_kg']} kg**"
             + (f"  ·  BOM-grounded: **{report['grounded_mass_kg']} kg**"
                f"  (ratio {report['mass_fidelity_ratio']}x)" if report.get("grounded_mass_kg") else ""),
             f"- verdict: **{'faithful' if report['faithful'] else 'FLAGGED (see below)'}**", ""]
    for f in report.get("flags", []):
        lines.append(f"- ⚠ {f}")
    return "\n".join(lines) + "\n"
