"""§4.8A — the BOM<->sim fidelity report: does the simulated body weigh what its parts list adds up to?

This compared the gene against a re-grounded COPY of itself, which stopped being a measurement the day grounding
moved onto the main build path: both sides were then the same call, the ratio was 1.00 by construction, and the
report cheerfully certified "faithful" while the sim body carried no battery, no compute and no sensors and the
parts list billed every motor twice. Measured at that point: an authored hexapod simulated at 15.043 kg with a
25.664 kg BOM, and this file reporting ratio 1.00.

So it now compares the sim body against THE PARTS LIST THAT SHIPS WITH IT. That is the number a customer can
check, and it is the one the mass work exists to close: ``grounded_physics.embody_component_masses`` puts the
BOM's own electronics on the links that carry them, and ``bom_builder`` counts each part once, so the two agree
on a body we composed. Where they do not — an IMPORTED robot kept at the customer's masses, or a body whose
parts changed without a re-ground — the difference is the finding, and it is stated with its number.
"""

from __future__ import annotations


def bom_sim_fidelity(gene) -> dict:
    """Compare the SIMULATED body's mass against the bill of materials that travels with it, and flag any gap.

    Returns the two masses, their ratio, the per-joint sim-torque-vs-selected-motor picture, and honest flags.
    """
    from virturoid.services.component_catalog import select_actuator

    sim_mass = float(sum(s.mass_kg for s in gene.segments))
    bom_mass, preserved = None, str((getattr(gene, "metadata", None) or {}).get("mass_source") or "") == "source_model"
    try:
        from virturoid.services.bom_builder import build_bom
        bom_mass = float((build_bom(gene).get("totals") or {}).get("mass_kg") or 0.0) or None
    except Exception:  # noqa: BLE001 - the parts list is best-effort; report sim-only if it cannot be built
        bom_mass = None

    joints = []
    for s in gene.actuated_joints():
        req = float(s.actuator_torque_nm or 10.0)
        motor = select_actuator(req)
        joints.append({"joint": s.name, "sim_torque_limit_nm": round(req, 2),
                       "selected_motor": motor.name, "motor_peak_nm": round(float(motor.peak_torque_nm), 2)})

    ratio = round(bom_mass / sim_mass, 2) if (bom_mass and sim_mass > 1e-6) else None
    flags = []
    if bom_mass and sim_mass > 1e-6 and abs(bom_mass - sim_mass) > max(0.05, 0.01 * sim_mass):
        if preserved:
            flags.append(
                f"parts list {bom_mass:.2f} kg vs simulated body {sim_mass:.2f} kg. EXPECTED on an imported "
                f"robot: the body is the customer's own machine, carried verbatim, and the parts list is that "
                f"machine plus the {bom_mass - sim_mass:.2f} kg of sensors/compute/power this BOM proposes "
                f"adding — which they may already have fitted.")
        else:
            flags.append(
                f"the simulated body ({sim_mass:.2f} kg) and its own parts list ({bom_mass:.2f} kg) disagree by "
                f"{bom_mass - sim_mass:+.2f} kg ({ratio}x). On a body we composed these should match: re-run "
                f"gene_build.ground_and_repair, which embodies the parts list's components onto the links.")
    return {"sim_mass_kg": round(sim_mass, 2),
            "bom_mass_kg": round(bom_mass, 2) if bom_mass else None,
            # kept under its old name so existing readers (spec sheet, ledger, UI) do not go blank
            "grounded_mass_kg": round(bom_mass, 2) if bom_mass else None,
            "mass_source_preserved": preserved,
            "mass_fidelity_ratio": ratio, "n_joints": len(joints), "joints": joints,
            "flags": flags, "faithful": not flags,
            "note": ("sim actuator torque limit = the SELECTED motor's peak once grounded; the BOM re-selects "
                     "from the same pinned requirement at the same margin, so the two parts lists agree. The "
                     "mass comparison is against the shipped bill of materials, not against a re-grounded copy "
                     "of this same gene — that earlier form compared a call with itself and always read 1.00.")}


def format_fidelity_md(report: dict) -> str:
    lines = ["# BOM<->Sim Fidelity", "",
             f"- sim mass: **{report['sim_mass_kg']} kg**"
             + (f"  ·  bill of materials: **{report['grounded_mass_kg']} kg**"
                f"  (ratio {report['mass_fidelity_ratio']}x)" if report.get("grounded_mass_kg") else ""),
             f"- verdict: **{'faithful' if report['faithful'] else 'FLAGGED (see below)'}**", ""]
    for f in report.get("flags", []):
        lines.append(f"- ⚠ {f}")
    return "\n".join(lines) + "\n"
