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


def _link_axis_perp(u: tuple[float, float, float]) -> tuple[float, float, float]:
    """The link's own +z direction, made perpendicular to the joint axis ``u`` — the second axis of the joint's
    frame. A link occupies [0, length] along +z, so this is "along the limb". Degenerates gracefully to +x when
    the joint spins about the limb's own axis (a roll joint), where "along the limb" has no in-plane meaning."""
    for ref in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
        d = u[0] * ref[0] + u[1] * ref[1] + u[2] * ref[2]
        v = (ref[0] - d * u[0], ref[1] - d * u[1], ref[2] - d * u[2])
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if n > 1e-6:
            return (v[0] / n, v[1] / n, v[2] / n)
    return (1.0, 0.0, 0.0)


def _joint_interface_xml(seg, pad: str, nm: str, u, r: float, hl: float) -> list[str]:
    """The MECHANICAL JOINT built around the motor can: a bearing race on each face of the can, a pivot pin on
    the rotation axis, and a two-plate CLEVIS bridging the parent's tip to this link's root.

    A datasheet motor can on its own is a drum threaded onto a stick, which is exactly how a composed leg read:
    a fat can, a thin rod, a fat can. Real hardware shows the INTERFACE — the bearing the link turns on, the pin
    it turns about, and the fork that carries load across the joint. Those three features are what make a machine
    read as ARTICULATED rather than as beads on a string, and they are the honest thing to draw: every one is
    derived from this joint's own axis and the selected actuator's envelope. Nothing here is copied from a real
    robot; it is generated from the same numbers the BOM cites.

    Geometry is expressed in the joint's own frame: ``u`` is the rotation axis and ``ey`` is the limb direction
    made perpendicular to it, so a clevis straddles the axis and reaches ALONG the limb regardless of how the
    joint is oriented in the body. Sizes are clamped against the link's own length so a short link is not
    swallowed by its own bracket.

    VISUAL ONLY, and deliberately named so that stays enforced: every geom here is ``mass=0 contype=0
    conaffinity=0`` and every name ends in ``_act``, which is the exact selector
    ``test_component_geometry.test_actuators_are_physics_byte_identical`` uses — so the existing guard that
    proves actuator housings cannot move a contact or a gram now covers these parts too, for free."""
    ey = _link_axis_perp(u)
    L = abs(float(getattr(seg, "length_m", 0.0) or 0.0))
    out: list[str] = []

    def _fromto(a: float, b: float) -> str:
        return (f'fromto="{u[0]*a:.5f} {u[1]*a:.5f} {u[2]*a:.5f} '
                f'{u[0]*b:.5f} {u[1]*b:.5f} {u[2]*b:.5f}"')

    # 1) BEARING RACES — a thin ring proud of each can face. This is the surface a real rotary joint turns on,
    #    and visually it terminates the can with a machined edge instead of a bare cylinder end.
    race_t, race_r = 0.10 * r, 1.10 * r
    for i, s in enumerate((1.0, -1.0)):
        out.append(f'{pad}<geom name="{nm}_race{i}_act" type="cylinder" '
                   f'{_fromto(s * hl, s * (hl + 2 * race_t))} size="{race_r:.5f}"'
                   f' material="mat_steel" mass="0" contype="0" conaffinity="0"/>')

    # 2) CLEVIS PLATES — two plates straddling the axis, reaching back over the parent's tip and forward onto
    #    this link. `xyaxes` pins the plate's own frame to the joint (local x along the axis, local y along the
    #    limb) so the fork is correctly oriented for any joint direction rather than guessing a world axis.
    plate_t = max(0.09 * r, 0.0015)
    plate_zh = min(1.35 * r, 0.46 * L) if L > 0 else 1.35 * r
    plate_wh = 0.70 * r
    d = hl + 2 * race_t + plate_t + 0.02 * r
    zc = 0.14 * r
    xy = (f'xyaxes="{u[0]:.5f} {u[1]:.5f} {u[2]:.5f} {ey[0]:.5f} {ey[1]:.5f} {ey[2]:.5f}"')
    for i, s in enumerate((1.0, -1.0)):
        px, py, pz = (u[0] * s * d + ey[0] * zc, u[1] * s * d + ey[1] * zc, u[2] * s * d + ey[2] * zc)
        out.append(f'{pad}<geom name="{nm}_yoke{i}_act" type="box" pos="{px:.5f} {py:.5f} {pz:.5f}" '
                   f'{xy} size="{plate_t:.5f} {plate_zh:.5f} {plate_wh:.5f}"'
                   f' material="mat_alu" mass="0" contype="0" conaffinity="0"/>')

    # 3) PIVOT PIN — through the clevis and proud on both sides. One small bright cylinder on the axis is the
    #    single most legible "this is a hinge" cue there is, and it ties the two plates and the can into one
    #    assembly instead of three unrelated lumps.
    pin_h = d + plate_t + 0.16 * r
    out.append(f'{pad}<geom name="{nm}_pin_act" type="cylinder" {_fromto(-pin_h, pin_h)} '
               f'size="{0.19 * r:.5f}" material="mat_metal" mass="0" contype="0" conaffinity="0"/>')
    return out


def actuator_housing_xml(seg, pad: str, *, margin: float = 1.3) -> str:
    """Visual-only MJCF for the real actuator at ``seg``'s joint AND the joint interface around it, sized to the
    part's true datasheet envelope and oriented so its rotation axis aligns with the joint axis. ``mass=0
    contype=0 conaffinity=0`` throughout, so it never touches dynamics. Returns "" for non-actuated links.

    The motor can is emitted exactly as before (same part, same envelope, same position — it is what the BOM
    cites and must stay truthful); ``_joint_interface_xml`` adds the bearing races, clevis and pivot pin that
    turn that can into a readable joint. See that function for why a bare can was the "beads on a string" cue."""
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
        motor = (f'{pad}<geom name="{nm}_act" type="cylinder" '
                 f'fromto="{-ex:.5f} {-ey:.5f} {-ez:.5f} {ex:.5f} {ey:.5f} {ez:.5f}" size="{r:.5f}"{vis}/>')
    else:
        # servo: a box whose output-shaft dimension (axis_dim) lies along the dominant world axis of the joint
        k = max(range(3), key=lambda i: abs(u[i]))
        others = [env[i] for i in range(3) if i != ad]
        half = [0.0, 0.0, 0.0]
        half[k] = env[ad] / 2.0
        rem = [i for i in range(3) if i != k]
        half[rem[0]] = others[0] / 2.0
        half[rem[1]] = others[1] / 2.0
        motor = (f'{pad}<geom name="{nm}_act" type="box" pos="0 0 0" '
                 f'size="{half[0]:.5f} {half[1]:.5f} {half[2]:.5f}"{vis}/>')
        r = max(others) / 2.0
        hl = env[ad] / 2.0
    return "\n".join([motor, *_joint_interface_xml(seg, pad, nm, u, r, hl)])


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
