"""Real off-the-shelf COMPONENT grounding (report 7's grounding layer) — render & cite the real actuator that
drives each joint, instead of a generic "motor can" guessed from the link radius.

Each actuated joint is matched to a real, buyable actuator from ``grounded_physics.ACTUATOR_CATALOG`` — which is
a dict view of the ONE source of truth, ``component_catalog.ACTUATORS`` (the same 12-part ladder the BOM specs
from) — by torque, and that part's true datasheet ENVELOPE is rendered as a visual-only housing at the joint.
Because render and BOM select from the SAME catalog with the SAME torque+margin, the motor you SEE on a joint
is the exact part the Bill of Materials lists (part number + supplier). This is the "use real parts"
differentiator: a real SERVO is an off-the-shelf component you buy and bolt on (legitimate grounding), NOT a
real robot's body shell (which would be design-copying).

THE BRIGHT LINE (enforced structurally + by test): this module may NOT import ``part_catalog`` (the real-robot
mesh kit-bash catalog). Components here are servos/fasteners/sensors — never a real robot's shell. Everything
emitted is ``mass=0 contype=0 conaffinity=0`` (visual-only), so dynamics/contacts stay byte-identical.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

from virturoid.services.grounded_physics import ACTUATOR_CATALOG


def select_actuator(required_nm: float) -> dict:
    """The smallest real actuator whose stall torque covers ``required_nm`` (else the largest in the catalog).
    Mirrors ``grounded_physics._pick_actuator`` but returns the FULL datasheet (envelope, part#, supplier)."""
    req = abs(float(required_nm or 0.0))
    for a in ACTUATOR_CATALOG:
        if a["stall_nm"] >= req:
            return a
    return ACTUATOR_CATALOG[-1]


def actuator_for_joint(seg, *, margin: float = 1.3) -> dict | None:
    """Datasheet of the actuator that drives ``seg``'s joint (None for welded/root/prismatic links). ``margin``
    matches ``grounded_physics.ground_gene`` so the rendered part is the one the BOM/physics would also pick."""
    if getattr(seg, "joint_type", None) != "revolute":
        return None
    required = abs(getattr(seg, "torque_req_nm", None) or seg.actuator_torque_nm or 8.0)
    # Use the exact same thermal-aware selector as grounding. A grounded segment stores the selected motor's
    # PEAK in ``actuator_torque_nm``; multiplying that peak by margin selected a second, larger motor for render/
    # CAD than the BOM actually named. ``torque_req_nm`` is the stable design load and is the correct input.
    from virturoid.services.component_catalog import select_actuator as select_catalog_actuator
    chosen = select_catalog_actuator(required, margin=margin, continuous_torque_nm=required * margin)
    return next((item for item in ACTUATOR_CATALOG if item["part"] == chosen.name),
                select_actuator(required * margin))


def actuator_housing_xml(seg, pad: str, *, margin: float = 1.3) -> str:
    """Visual-only MJCF geom for the real actuator at ``seg``'s joint, sized to the part's true datasheet
    envelope and oriented so its rotation axis aligns with the joint axis. ``mass=0 contype=0 conaffinity=0``
    so it never touches dynamics. Returns "" for non-actuated links."""
    ds = actuator_for_joint(seg, margin=margin)
    if ds is None:
        return ""
    ax = seg.joint_axis
    n = math.sqrt(ax[0] ** 2 + ax[1] ** 2 + ax[2] ** 2) or 1.0
    u = (ax[0] / n, ax[1] / n, ax[2] / n)
    env = ds["envelope_m"]
    ad = int(ds.get("axis_dim", 2))                       # which envelope dim is the rotation axis
    vis = ' material="mat_joint" mass="0" contype="0" conaffinity="0"'
    nm = escape(seg.name)
    if ds.get("shape") == "cylinder":                     # quasi-direct-drive pancake motor: a can on the axis
        r = max(env[0], env[1]) / 2.0
        hl = env[ad] / 2.0
        ex, ey, ez = u[0] * hl, u[1] * hl, u[2] * hl
        return (f'{pad}<geom name="{nm}_act" type="cylinder" '
                f'fromto="{-ex:.5f} {-ey:.5f} {-ez:.5f} {ex:.5f} {ey:.5f} {ez:.5f}" size="{r:.5f}"{vis}/>')
    # servo: a box whose output-shaft dimension (axis_dim) lies along the dominant world axis of the joint
    k = max(range(3), key=lambda i: abs(u[i]))
    others = [env[i] for i in range(3) if i != ad]
    half = [0.0, 0.0, 0.0]
    half[k] = env[ad] / 2.0
    rem = [i for i in range(3) if i != k]
    half[rem[0]] = others[0] / 2.0
    half[rem[1]] = others[1] / 2.0
    return (f'{pad}<geom name="{nm}_act" type="box" pos="0 0 0" '
            f'size="{half[0]:.5f} {half[1]:.5f} {half[2]:.5f}"{vis}/>')


def bill_of_materials(gene, *, margin: float = 1.3) -> list[dict]:
    """Read-only real-part BOM for a gene's actuated joints (does NOT mutate the gene, unlike
    ``grounded_physics.ground_gene``). One row per joint: which real actuator, its part#/mass/stall/supplier,
    and ``under_spec`` — True when the joint's required torque exceeds even the largest catalog actuator (an
    honest flag that no off-the-shelf part we stock covers this joint, rather than silently shipping it)."""
    bom: list[dict] = []
    for s in gene.segments:
        ds = actuator_for_joint(s, margin=margin)
        if ds is None:
            continue
        # ``actuator_torque_nm`` is the selected part's PEAK after grounding, not
        # the design load. Re-multiplying that peak by margin made every largest
        # catalog motor appear under-spec and caused safe edits to auto-revert.
        required = round(abs(getattr(s, "torque_req_nm", None) or s.actuator_torque_nm or 8.0) * margin, 2)
        bom.append({"joint": s.name, "part": ds["part"], "mass_kg": ds["mass_kg"],
                    "stall_nm": ds["stall_nm"], "supplier": ds["supplier"],
                    "required_nm": required, "under_spec": required > ds["stall_nm"] + 1e-6})
    return bom
