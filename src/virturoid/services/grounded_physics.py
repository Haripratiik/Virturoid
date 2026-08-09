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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class PhysicalPrior:
    id: str
    mass_band_kg: tuple[float, float]
    source: str


# Everything a grounded body does NOT model: battery, compute, wiring harness, fasteners, covers. On commercial
# machines of these classes it is roughly a quarter of the finished mass (a ~15 kg mid-size quadruped carries
# ~1.5 kg of battery plus compute, harness and shell), and — the point — it is a FRACTION of the robot, not a
# constant that happens to make every robot weigh the same. See the clamp in ``ground_gene``.
#
# It is an ESTIMATE OF THE RESIDUE, not of the whole. Once :func:`embody_component_masses` has bolted the parts
# list's own battery/compute/sensors onto the body, those are named parts with datasheet masses and this fraction
# no longer has to guess at them — see ``_apply_balance_of_system``'s ``itemised_kg``, which is subtracted out so
# the same finished mass is reached with more of it itemised and less of it estimated.
_BALANCE_OF_SYSTEM_FRACTION = 0.25

#: BOM line categories that are NOT a bolt-on component: the motors grounding already folded into their links,
#: the raw structure the links already are, and the residual estimate this module itself applies.
_NON_COMPONENT_CATEGORIES = frozenset({"actuator", "material", "balance_of_system"})

#: Where a part physically rides. Everything not named here (battery, compute, IMU, radios) goes on the trunk,
#: which is where it goes on every machine of these classes.
_HEAD_CATEGORIES = frozenset({"camera", "lidar", "thermal", "depth", "microphone", "gps"})
_WHEEL_CATEGORIES = frozenset({"wheel", "drive_motor"})
_TIP_CATEGORIES = frozenset({"gripper", "force_torque"})

_QUADRUPED_PRIOR = PhysicalPrior("legged.quadruped", (12.0, 15.0), "commercial mid-size quadruped envelope")
_COBOT_PRIOR = PhysicalPrior("manipulator.cobot", (17.0, 21.0), "6/7-axis collaborative-arm envelope")
_AMR_PRIOR = PhysicalPrior("mobile.amr", (70.0, 150.0), "warehouse autonomous-mobile-robot envelope")


def _leg_branches(gene) -> list:
    root = gene.root()
    if root is None:
        return []
    out = []
    for child in gene.children_of(root.name):
        stack, revolute, fixed_leaf = [child], 0, False
        while stack:
            node = stack.pop()
            if node.joint_type == "revolute":
                revolute += 1
            kids = gene.children_of(node.name)
            fixed_leaf |= not kids and node.joint_type in (None, "fixed")
            stack.extend(kids)
        if revolute >= 2 and fixed_leaf:
            out.append(child)
    return out


def physical_prior_for(gene) -> PhysicalPrior | None:
    """Resolve an extensible evidence prior from structure; unknown morphologies remain unforced."""
    wheels = [s for s in gene.segments if s.shape == "cylinder" and s.joint_type == "revolute"]
    arm_joints = [s for s in gene.segments if s.joint_type == "revolute" and s not in wheels]
    if len(_leg_branches(gene)) == 4:
        return _QUADRUPED_PRIOR
    if len(wheels) >= 4 and len(_leg_branches(gene)) == 0:
        return _AMR_PRIOR
    if len(arm_joints) in (6, 7) and len(_leg_branches(gene)) == 0:
        return _COBOT_PRIOR
    return None


def _required_joint_loads(gene, structural_mass: dict[str, float], *, total_mass_hint: float | None = None) -> dict[str, float]:
    """Derive joint requirements from distal mass, lever arm and support role—not chain position.

    KNOWN AND MEASURED: ``total_mass_hint`` IS NOT THE MASS OF THE FINISHED ROBOT, so a leg's support torque is
    sized against a lighter machine than the one that will stand on it. The hint is the class band's midpoint
    when a prior applies and ``sum(structural_mass)`` otherwise — neither contains the motors, and since
    :func:`embody_component_masses` neither contains the battery, the compute or the sensors either. On this
    checkout, sized-against vs finished:

        two-legged robot   13.500 -> 50.089 kg   3.71x   (was 3.43x before components were embodied)
        six-legged robot    7.273 -> 17.894 kg   2.46x   (2.07x)
        authored horse     13.500 -> 18.849 kg   1.40x   (1.18x)
        authored cheetah   13.500 -> 17.872 kg   1.32x   (1.11x)
        authored dog       13.500 -> 12.791 kg   0.95x   (0.95x)

    This is a BOM FINDING — those actuators are undersized for the machine — and deliberately NOT closed here.
    Making it right means the requirement rises, a bigger motor is selected, the body gets heavier and the
    requirement rises again; the loop terminates (``torque_req_nm`` is pinned, so a re-ground never re-reads a
    motor's peak as a demand) but it moves every actuator, every mass and therefore every gait verdict that
    ships today. It needs its own change with its own walk gate. The wrong answer would be to shave the body
    back until the existing motors look adequate — see [[make-it-learn-dont-teach-it]]: a robot is not made
    buildable by making the spec sheet agree with itself.
    """
    by = {s.name: s for s in gene.segments}
    children: dict[str, list[str]] = {}
    for seg in gene.segments:
        children.setdefault(seg.parent, []).append(seg.name)

    def descendants(name: str) -> list[str]:
        out, stack = [], [(name, 0.0)]
        while stack:
            current, distance = stack.pop()
            out.append((current, distance))
            seg = by[current]
            stack.extend((child, distance + seg.length_m) for child in children.get(current, []))
        return out

    leg_roots = {s.name for s in _leg_branches(gene)}
    total_mass = float(total_mass_hint or sum(structural_mass.values()))
    wheel_count = max(1, sum(1 for s in gene.segments if s.shape == "cylinder" and s.joint_type == "revolute"))
    loads: dict[str, float] = {}
    for seg in gene.segments:
        if seg.joint_type not in ("revolute", "prismatic"):
            continue
        desc = descendants(seg.name)
        distal_hold = 9.81 * sum(structural_mass[name] * (distance + 0.5 * by[name].length_m)
                                 for name, distance in desc)
        if seg.shape == "cylinder":
            # A skid-steer drive must overcome lateral scrub while turning, not
            # only low straight-line rolling resistance. Sizing at rolling loss
            # alone moved a 110 kg AMR forward but left it unable to yaw at all.
            rolling = 2.0 * 0.03 * total_mass * 9.81 * max(seg.radius_m, 0.02) / wheel_count
            # Four fixed wheels must also scrub laterally during an in-place
            # skid turn.  The 0.25 effective scrub coefficient is deliberately
            # below dry-rubber Coulomb friction: tyre/contact compliance and
            # partial slip reduce the sustained load.  Keeping it explicit in
            # the load model lets a lighter or smaller-wheeled base select a
            # smaller drive naturally, while a warehouse-scale AMR is no
            # longer undersized by a straight-line-only calculation.
            skid_turn = 2.0 * 0.25 * total_mass * 9.81 * max(seg.radius_m, 0.02) / wheel_count
            required = max(rolling, skid_turn)
        elif seg.joint_type == "prismatic":
            required = 1.5 * 9.81 * sum(structural_mass[name] for name, _ in desc)
        else:
            branch = seg.name
            while by[branch].parent is not None and by[branch].parent != gene.root().name:
                branch = by[branch].parent
            if branch in leg_roots:
                reach = max((distance + by[name].length_m for name, distance in desc), default=seg.length_m)
                support = 9.81 * (total_mass / max(1, len(leg_roots))) * reach
                required = 2.2 * max(distal_hold, support)
            else:
                required = 1.5 * distal_hold
        loads[seg.name] = round(max(0.2, required), 3)
    return loads


def _link_volume_m3(shape: str, length_m: float, radius_m: float) -> float:
    """Solid volume of the link's primitive (capsule = cylinder + 2 hemispheres; box uses radius as
    half-width in x/y; cylinder = disc x length)."""
    r, L = max(1e-4, radius_m), max(1e-4, length_m)
    if shape == "box":
        return (2 * r) * (2 * r) * L
    if shape == "cylinder":
        return math.pi * r * r * L
    return math.pi * r * r * L + (4.0 / 3.0) * math.pi * r ** 3   # capsule


def _pick_actuator(required_nm: float):
    """Legacy (torque-only, stall-sized) view kept for callers that want the (name, mass, stall) tuple."""
    for name, mass, stall in ACTUATORS:
        if stall >= required_nm:
            return name, mass, stall
    return ACTUATORS[-1]


def physical_fingerprint(gene) -> dict:
    """The physical identity of a body: what has to match for two genes to be the SAME robot.

    Used to prove (or disprove) that the body a certificate was measured on is the body the package ships.
    Deliberately narrow -- mass, size and joint capacity, i.e. everything a verdict actually depends on --
    so cosmetic differences (materials, colours, visual meshes) never raise a false alarm.
    """
    return {
        "n_segments": len(gene.segments),
        "total_mass_kg": round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 6),
        "links": {str(s.name): (round(float(s.mass_kg or 0.0), 6), round(float(s.length_m or 0.0), 6),
                                round(float(s.radius_m or 0.0), 6),
                                round(float(s.actuator_torque_nm or 0.0), 6))
                  for s in gene.segments},
    }


def fingerprint_delta(a: dict, b: dict, *, mass_tol_kg: float = 0.01, dim_tol_m: float = 1e-4) -> dict:
    """Compare two :func:`physical_fingerprint` results. Returns ``{same, total_mass_kg, n_links_changed,
    changed}`` -- ``same`` is the whole point: it is what lets a certificate claim deploy==measure honestly
    instead of printing the claim unconditionally."""
    la, lb = a.get("links") or {}, b.get("links") or {}
    changed = []
    for name in sorted(set(la) | set(lb)):
        va, vb = la.get(name), lb.get(name)
        if va is None or vb is None:
            changed.append({"link": name, "was": va, "now": vb})
            continue
        if abs(va[0] - vb[0]) > mass_tol_kg or any(abs(va[i] - vb[i]) > dim_tol_m for i in (1, 2)):
            changed.append({"link": name, "was": {"mass_kg": va[0], "length_m": va[1], "radius_m": va[2]},
                            "now": {"mass_kg": vb[0], "length_m": vb[1], "radius_m": vb[2]}})
    dm = round(float(b.get("total_mass_kg", 0.0)) - float(a.get("total_mass_kg", 0.0)), 6)
    return {"same": not changed and abs(dm) <= mass_tol_kg,
            "total_mass_kg": [a.get("total_mass_kg"), b.get("total_mass_kg")], "delta_mass_kg": dm,
            "n_links_changed": len(changed), "changed": changed[:12]}


def source_declared_torques(gene) -> dict[str, float]:
    """``{segment: N.m}`` the CUSTOMER'S OWN MODEL declared, or ``{}`` for a body we generated.

    The torque twin of ``metadata['mass_source']``. ``robot_import`` stamps ``metadata['torque_source'] =
    'source_model'`` plus a per-joint record when it finds real limits in the file (actuator forcerange, a
    force-mode motor's ctrlrange, or a joint's actuatorfrcrange). Grounding reads it here so it sizes a BOM
    part AROUND the manufacturer's number instead of replacing it -- ``preserve_mass`` protected mass and
    there was no equivalent for torque, so a Menagerie Go2's 23.7/23.7/45.43 N.m shipped as 10.6/10.6/1.5
    with the ordering INVERTED (the knee is the strongest joint on the real machine and was the weakest by
    30x on ours), and a UR5e's 150/150/150/28/28/28 became 360/360/48/4.1/1.5/0.5.

    Deliberately reads the METADATA record rather than the segment's current ``actuator_torque_nm``: the
    record is the source of truth and survives a body that was already grounded wrong once.

    Empty for a segment whose ``torque_req_nm`` has been moved OFF the declared value. That is how a
    deliberate capability amend still works: ``edit_operators.set_payload`` scales the REQUIREMENT so that
    re-grounding picks a bigger motor, and a preservation rule with no release would have silently reverted
    the customer's own explicit "make it lift 10 kg". A declaration describes the joint as it was; once
    somebody re-specifies the joint, it no longer does.
    """
    meta = getattr(gene, "metadata", None)
    if not isinstance(meta, dict) or str(meta.get("torque_source") or "") != "source_model":
        return {}
    rec = meta.get("source_actuator_torque_nm")
    by = {s.name: s for s in gene.segments}
    out: dict[str, float] = {}
    items = (rec.items() if isinstance(rec, dict) and rec
             # A marker with no per-joint record (an older gene, or one rebuilt by an amend that dropped the
             # block): fall back to whatever positive limits the segments still carry, which on an imported
             # body are the imported ones.
             else [(s.name, s.actuator_torque_nm) for s in gene.segments])
    for name, value in items:
        seg = by.get(str(name))
        if seg is None or seg.joint_type not in ("revolute", "prismatic"):
            continue
        try:
            nm = abs(float(value))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(nm) or nm <= 0.0:
            continue
        req = getattr(seg, "torque_req_nm", None)
        if req is not None and abs(float(req) - nm) > max(1e-6, 1e-3 * nm):
            continue                                   # re-specified since import -> let grounding re-size
        out[str(name)] = nm
    return out


def ground_gene(gene, *, material: str = "aluminum", fill: float = 0.3, margin: float = 1.3,
                preserve_mass: bool = False, preserve_torque: bool | None = None,
                derive_mass_links=None) -> dict:
    """Mutate ``gene`` in place: set each link's mass from material+geometry (+ its actuator's mass) and each
    actuated joint's torque limit to a real actuator's PEAK. The actuator is sized so its CONTINUOUS (rated)
    torque covers the sustained requirement with ``margin`` -- real thermal practice; never size a joint at its
    stall, which cannot be held continuously (the failure the BOM certificate's G2 gate catches). Returns
    ``{material, bom, total_mass_kg, actuator_count}`` -- ``bom`` is the real parts list (now with rated torque and
    no-load speed so the executable-on-BOM certificate can grade the exact shipped part).

    ``preserve_mass`` sizes actuators and builds the BOM but leaves every ``mass_kg`` EXACTLY as it is. Use it
    when the body's masses are AUTHORITATIVE rather than derived -- an imported customer robot carries its
    manufacturer's real per-link masses, and re-deriving them from a primitive-volume estimate replaces the
    customer's robot with our guess at it (measured on a Menagerie Go2: 15.206 kg of real link masses became
    13.235 kg of carbon-fibre estimates, base 6.921 -> 6.107).

    ``derive_mass_links`` is the ESCAPE HATCH from ``preserve_mass``, per link, and it exists because
    preservation is a statement about PROVENANCE and provenance is per link, not per body. A customer's Go2 is
    authoritative; a robot arm somebody just bolted onto it is not, and there is no manufacturer number to
    preserve for a link that did not exist a second ago. Without this, an edit had exactly two options: keep
    every mass (so a new limb weighs whatever the operator guessed and never gets its actuator) or re-derive
    every mass (measured: ONE ``add_limb`` moved a preserved 15.206 kg Go2 to 32.683 kg while the arm itself
    accounted for 2.733 -- +14.7 kg of silent re-massing of the customer's own robot, base 6.921 -> 10.306 and
    every calf 0.241 -> 2.177). Named links are derived from geometry + actuator exactly as an unpreserved run
    would; every other link keeps the number it arrived with. The caller decides which is which -- see
    ``edit_operators._reground_and_gate``, where it is (links this op created) + (links this op re-specified).

    ``preserve_torque`` is the same protection for ACTUATOR CAPACITY, and it did not exist: mass was safe and
    torque was overwritten one line later, so an imported robot kept its manufacturer's masses and lost its
    manufacturer's motors. ``None`` (the default) means AUTO -- on by exactly the joints
    :func:`source_declared_torques` reports, which is empty for every body we generate, so no existing caller
    changes behaviour and every import path gets the protection without having to know to ask for it. Pass
    ``False`` to force a fresh catalog sizing (a deliberate re-spec), ``True`` to protect declared limits on a
    gene that carries them without the import marker.

    A preserved joint still gets a real BOM part: the declared limit becomes the joint's REQUIREMENT and the
    same sizing call runs on it, so the parts list stays real and stops claiming a 520 N.m motor drives an
    87 N.m Panda axis. Why this is urgent beyond accuracy: ``sysid.excitation`` bounds a plan written to be run
    ON PHYSICAL HARDWARE at a fraction of ``actuator_torque_nm``, and against the invented figures that put
    180 N.m of commanded torque on the Panda's 87 N.m J2.

    The material/fill actually used is RECORDED on the gene (``metadata['grounding']``) so a later re-ground
    reproduces this body instead of silently re-deriving it at a different density. That silent switch is
    exactly what made export ship a different robot than the one verified: ``ground_and_repair`` hardcoded
    carbon fibre (1600 kg/m3) while the held body had been grounded at aluminium (2700), so every link lost
    41% of its structural mass on the way out the door.
    """
    from virturoid.services.component_catalog import select_actuator
    density = MATERIALS.get(material, MATERIALS["aluminum"])
    # Links whose mass is OURS to derive even under ``preserve_mass`` (added or re-specified by an edit).
    _derive = {str(n) for n in (derive_mass_links or ())}

    def _mass_is_ours(seg) -> bool:
        return (not preserve_mass) or (seg.name in _derive)
    declared = source_declared_torques(gene) if preserve_torque is not False else {}
    if preserve_torque is True and not declared:
        declared = {s.name: abs(float(s.actuator_torque_nm))
                    for s in gene.segments
                    if s.joint_type in ("revolute", "prismatic") and s.actuator_torque_nm}
    torque_where = ((getattr(gene, "metadata", None) or {}).get("source_actuator_torque_where") or {})
    bom: list[dict] = []
    structural_mass = {s.name: _link_volume_m3(s.shape, s.length_m, s.radius_m) * density * fill
                       for s in gene.segments}
    prior = physical_prior_for(gene)
    target_mass = (sum(prior.mass_band_kg) / 2.0) if prior else None
    derived_loads = _required_joint_loads(gene, structural_mass, total_mass_hint=target_mass)
    total = 0.0
    embodied_actuator: dict[str, float] = {}
    for s in gene.segments:
        struct = structural_mass[s.name]
        act_mass = 0.0
        if s.joint_type in ("revolute", "prismatic"):
            # The joint's torque REQUIREMENT is fixed by the design -- pin it the first time so re-grounding is
            # IDEMPOTENT. Before this, `required` re-read `actuator_torque_nm`, which grounding overwrites with the
            # SELECTED motor's PEAK; so each re-ground treated a motor peak as the new requirement and sized an even
            # bigger motor -- a 120 Nm joint ratcheted to 520 Nm after two grounds, silently over-sizing every arm
            # (and its housing geom, which then collided in grasp scenes). Now the requirement is remembered.
            #
            # A DECLARED LIMIT IS THE REQUIREMENT. When the customer's own file states this joint's capacity
            # (``keep`` > 0), it is both the number the joint must keep AND the number the BOM part has to
            # cover: the manufacturer already did this sizing, on hardware. Pinning it as ``torque_req_nm``
            # also makes a later deliberate re-spec detectable -- see ``source_declared_torques``.
            keep = float(declared.get(s.name) or 0.0)
            if s.torque_req_nm is None:
                s.torque_req_nm = keep or derived_loads.get(s.name, abs(s.actuator_torque_nm or 8.0))
            required = s.torque_req_nm
            # The SELECTION call below is deliberately IDENTICAL for a preserved and a derived joint. The only
            # difference preservation makes is what ``required`` is (the customer's declared limit rather than
            # our structural estimate) and that ``actuator_torque_nm`` is not overwritten afterwards.
            # ``bom_builder`` re-runs exactly this call from ``torque_req_nm`` at the same 1.3x margin to build
            # bom.json; sizing the preserved joints at a different margin here would have made one package ship
            # two contradictory motor lists -- the defect bom_builder.py:783 documents, re-created from the
            # other end.
            if s.joint_type == "prismatic":
                # A slider needs LINEAR FORCE, while the catalog describes motor
                # shaft torque. Size a real rotary actuator through a short pinion/
                # leadscrew equivalent and expose its output force to MuJoCo. The
                # former torque-as-force mixup reduced grounded finger clamps from
                # 8 N to 1.5 N and made every real contact grasp miss.
                transmission_m = max(0.01, min(0.025, float(s.radius_m)))
                equivalent_torque = float(required) * transmission_m
                act = select_actuator(equivalent_torque, margin=margin,
                                      continuous_torque_nm=equivalent_torque * margin)
                act_mass = act.mass_kg
                # On a slider ``actuator_torque_nm`` carries a FORCE in newtons, and so does a declared
                # ``forcerange`` on a slide joint -- the two are the same quantity, so ``keep`` substitutes
                # directly for the catalog part's output force.
                output_force = keep or (act.peak_torque_nm / transmission_m)
                if not keep:
                    s.actuator_torque_nm = round(output_force, 3)
                bom.append({"role": s.name, "part": act.name, "mass_kg": act.mass_kg,
                            "peak_force_n": round(output_force, 2), "rated_force_n": round(act.rated_torque_nm / transmission_m, 2),
                            "required_force_n": round(float(required), 2), "transmission_m": transmission_m,
                            "max_speed_radps": act.max_speed_radps,
                            **_source_torque_note(s, keep, torque_where)})
                if _mass_is_ours(s):
                    s.mass_kg = round(max(0.02, struct + act_mass), 3)
                    embodied_actuator[s.name] = act_mass
                total += s.mass_kg
                continue
            # rated torque must cover the sustained requirement (with margin) -> thermal headroom; peak then has
            # a real transient reserve above it. This is what makes a grounded body BOTH walk and certify.
            act = select_actuator(required, margin=margin, continuous_torque_nm=required * margin)
            act_mass = act.mass_kg
            if not keep:
                s.actuator_torque_nm = act.peak_torque_nm      # joint limit = peak (transients); duty rides on rated
            bom.append({"role": s.name, "part": act.name, "mass_kg": act.mass_kg,
                        "stall_nm": act.peak_torque_nm, "rated_nm": act.rated_torque_nm,
                        "max_speed_radps": act.max_speed_radps, "required_nm": round(required, 2),
                        **_source_torque_note(s, keep, torque_where)})
        if _mass_is_ours(s):
            s.mass_kg = round(max(0.02, struct + act_mass), 3)  # grounded mass = structure + actuator
            embodied_actuator[s.name] = act_mass
        total += s.mass_kg
    # WHAT THIS CALL FOLDED INTO WHICH LINK, so the parts list can stop selling the same motor twice. Each
    # ``s.mass_kg`` above is structure + its actuator, and ``bom_builder`` groups those link masses into the
    # "material" lines while ALSO listing every motor as an actuator line -- so the BOM totals counted the
    # actuators a second time. Measured at HEAD: an authored horse simulated at 15.951 kg had a parts list
    # claiming 29.212 kg, of which 10.410 was its own motors billed twice; an imported Unitree G1 whose real
    # 33.341 kg we preserve exactly shipped a 94.176 kg parts list. The ledger below is what makes each part
    # countable once. Links whose mass this call did NOT derive are absent: nothing was folded into them here.
    by_name = {s.name: s for s in gene.segments}
    embodied = _embodiment_record(gene)
    embodied["actuator_kg"] = {k: round(v, 6) for k, v in embodied_actuator.items() if v > 0.0}
    # A re-ground REPLACED every mass it derived, so whatever ``embody_component_masses`` had bolted onto those
    # links is gone from the body and must go from the ledger too -- otherwise the BOM would keep subtracting a
    # battery the link no longer carries and report a negative structure.
    for key in ("component_kg", "balance_of_system_kg"):
        embodied[key] = {n: v for n, v in (embodied.get(key) or {}).items()
                         if n in by_name and not _mass_is_ours(by_name[n])}
    if not embodied.get("component_kg"):
        embodied.pop("components", None)
    total = sum(float(s.mass_kg) for s in gene.segments)
    balance_mass = 0.0
    if not preserve_mass:
        balance_mass = _apply_balance_of_system(gene, prior, own_total=total)
        total = sum(float(s.mass_kg) for s in gene.segments)
    # RECORD WHAT THESE MASSES WERE DERIVED AT. Without this, a downstream re-ground has no way to reproduce
    # the body it was handed and simply picks its own density -- which is how export shipped a robot 30% lighter
    # than the one the customer watched walk. ``preserve_mass`` runs leave any existing record alone: the masses
    # did not come from this call's material, so claiming they did would be the same lie one level down.
    if not preserve_mass:
        gene.metadata["grounding"] = {"material": material, "fill": round(float(fill), 6),
                                      "margin": round(float(margin), 6)}
    return {"material": material, "bom": bom, "total_mass_kg": round(total, 3),
            "mass_preserved": bool(preserve_mass),
            # Which links this call derived a mass for DESPITE preservation -- the disclosure that keeps
            # "we kept your robot's mass" a checkable claim rather than a slogan.
            "mass_derived_links": sorted(n for n in _derive if any(s.name == n for s in gene.segments)),
            # Visible, per-joint, in the same report the BOM travels in: a reader must be able to tell which
            # joint limits are the customer's measurement and which are our catalog estimate.
            "torque_preserved": bool(declared),
            "n_torque_preserved": len(declared),
            "torque_source": ((getattr(gene, "metadata", None) or {}).get("torque_source") if declared
                              else None),
            "torque_preserved_joints": sorted(declared),
            "actuator_count": len(bom), "physical_prior": (prior.id if prior else None),
            "mass_band_kg": (list(prior.mass_band_kg) if prior else None),
            "balance_of_system_mass_kg": round(balance_mass, 3),
            # The motor mass that is now INSIDE the link masses above. ``bom_builder`` subtracts it when it
            # groups links into material lines, so a motor is billed once.
            "actuator_mass_embodied_kg": round(sum(embodied_actuator.values()), 3),
            # Named parts the SIM BODY still does not carry. Zero once ``embody_component_masses`` has run.
            "component_mass_embodied_kg": round(sum((_embodiment_record(gene).get("component_kg") or {}).values()), 3)}


def _embodiment_record(gene) -> dict:
    """The per-link ledger of everything grounding/embodiment folded INTO a link mass, created on first use.

    Three buckets, because a reader of ``sum(seg.mass_kg)`` otherwise cannot tell structure from the parts
    bolted to it, and ``bom_builder`` cannot bill a part once:

      ``actuator_kg``           the motor :func:`ground_gene` folded into the link it drives
      ``component_kg``          the BOM's own battery/compute/sensors/wheels :func:`embody_component_masses` added
      ``balance_of_system_kg``  the residual ESTIMATE (wiring, fasteners, covers) the class band applies

    Everything left over is STRUCTURE — the material a fabricator actually buys.
    """
    meta = getattr(gene, "metadata", None)
    if not isinstance(meta, dict):
        meta = {}
        gene.metadata = meta
    rec = meta.get("embodied_mass")
    if not isinstance(rec, dict):
        rec = {}
        meta["embodied_mass"] = rec
    for key in ("actuator_kg", "component_kg", "balance_of_system_kg"):
        if not isinstance(rec.get(key), dict):
            rec[key] = {}
    return rec


def structural_link_masses(gene) -> dict[str, float]:
    """``{segment: kg}`` of STRUCTURE ONLY — the link mass minus every part folded into it.

    This is the number a materials line on a bill of materials means: the aluminium/carbon a fabricator buys.
    ``sum(seg.mass_kg)`` is not that number on a grounded body — it already contains the motor, and after
    :func:`embody_component_masses` the battery and the compute too. Billing those again under "material" is
    how a 15.951 kg horse acquired a 29.212 kg parts list.

    Floors at zero per link: :func:`ground_gene` applies a 0.02 kg minimum link mass, so a tiny link carrying a
    0.6 kg motor would otherwise report negative structure.
    """
    # Deliberately READ-ONLY (no ``_embodiment_record``): this runs from ``build_bom``, which a certificate, a
    # compliance report and a spec sheet all call on bodies they are only inspecting. A reader that quietly
    # adds a key to the gene's metadata is a reader that changes what gets serialised into robot_genome.json.
    rec = (getattr(gene, "metadata", None) or {}).get("embodied_mass")
    rec = rec if isinstance(rec, dict) else {}
    folded: dict[str, float] = {}
    for key in ("actuator_kg", "component_kg", "balance_of_system_kg"):
        for name, kg in (rec.get(key) or {}).items():
            folded[str(name)] = folded.get(str(name), 0.0) + float(kg or 0.0)
    return {s.name: max(0.0, float(s.mass_kg or 0.0) - folded.get(s.name, 0.0)) for s in gene.segments}


def _apply_balance_of_system(gene, prior: PhysicalPrior | None, *, own_total: float,
                             itemised_kg: float = 0.0) -> float:
    """Put the UN-ITEMISED balance of system on the trunk and return what it added.

    CLAMP INTO THE BAND — do not SNAP TO ITS MIDPOINT.

    What the band protects: a body grounded from geometry + actuators alone has no battery, no compute, no
    wiring, no fasteners and no covers, so it is a styrofoam twin of the robot the customer would actually
    build. Sizing actuators and signing a walk verdict against that mass is the split-brain this exists to
    close, and lifting an implausibly light body to the class floor still closes it.

    What it ALSO did, and should not have: ``target = mid(band)`` made the balance-of-system mass "whatever it
    takes to reach one number", so EVERY under-band quadruped came out at EXACTLY 13.500 kg. Measured, the
    authored dog grounds to 9.593 kg of its own structure+actuators and the cat to 10.577 -- a real 1 kg
    difference the design earned -- and both then shipped as 13.500. A dog and a cat were literally the same
    robot on the spec sheet. The band was written to stop a body being too light, not to erase what the body is.

    So: take the balance of system as the FRACTION of finished mass it really is on a commercial machine of this
    class (~25%: battery, compute, harness, shell), apply it to the body's OWN grounded mass, and clamp the
    result into the band. Lighter body in, lighter body out; nothing escapes the floor.

    ``itemised_kg`` IS THE PART OF THAT FRACTION WE NO LONGER HAVE TO GUESS. Once
    :func:`embody_component_masses` has bolted the parts list's own battery, compute and sensors onto the body,
    those are named parts with datasheet masses sitting on real links, and continuing to estimate them as well
    would bill them twice. The FINISHED-mass target is unchanged -- what changes is how much of it is a part
    number and how much is still an estimate of the wiring, fasteners and covers no parts list names.

    Known and deliberately NOT changed here: a body that grounds ABOVE the floor gets no estimate at all, so
    the mapping still steps at ``lo``. Those bodies now get their ITEMISED components (see
    :func:`embody_component_masses`), which is the half of the hole that can be closed with real numbers;
    giving them an estimated residue as well moves masses that ship today and is its own change with its own
    gate.

    IDEMPOTENT: strips whatever a previous call put on the trunk before re-applying, so ground -> embody ->
    ground leaves one estimate on the body, not three.
    """
    rec = _embodiment_record(gene)
    by_name = {s.name: s for s in gene.segments}
    for name, kg in list((rec.get("balance_of_system_kg") or {}).items()):
        seg = by_name.get(str(name))
        if seg is not None:
            seg.mass_kg = round(max(0.001, float(seg.mass_kg or 0.0) - float(kg or 0.0)), 3)
        own_total -= float(kg or 0.0)
    rec["balance_of_system_kg"] = {}
    root = gene.root()
    if prior is None or root is None or own_total >= float(prior.mass_band_kg[0]):
        return 0.0
    lo, hi = float(prior.mass_band_kg[0]), float(prior.mass_band_kg[1])
    target = min(hi, max(lo, own_total / (1.0 - _BALANCE_OF_SYSTEM_FRACTION)))
    balance_mass = max(0.0, target - own_total - float(itemised_kg))
    if balance_mass > 0.0:
        root.mass_kg = round(float(root.mass_kg or 0.0) + balance_mass, 3)
        rec["balance_of_system_kg"] = {root.name: round(balance_mass, 6)}
    gene.metadata["physical_prior"] = {
        "id": prior.id, "mass_band_kg": [lo, hi], "source": prior.source,
        "balance_of_system_mass_kg": round(balance_mass, 3),
        "balance_of_system_fraction": _BALANCE_OF_SYSTEM_FRACTION,
        # How much of the balance of system is a PART NUMBER rather than this fraction's guess.
        "itemised_component_mass_kg": round(float(itemised_kg), 3),
        "own_mass_kg": round(own_total, 3),
        "clamped_to_band": bool(target in (lo, hi)),
        "balance_of_system": ["battery", "compute", "wiring", "fasteners", "covers"],
    }
    return balance_mass


def _component_host(gene, category: str) -> str:
    """Which LINK a bolt-on part rides on. Placement is inertia, not bookkeeping: a 2.6 kg battery put on a
    foot is a different robot from one put on the trunk."""
    root = gene.root()
    root_name = root.name if root is not None else (gene.segments[0].name if gene.segments else "")
    cat = str(category or "").lower()
    if cat in _WHEEL_CATEGORIES:
        wheels = [s.name for s in gene.segments if s.shape == "cylinder" and s.joint_type == "revolute"]
        if wheels:
            return wheels[0]
    if cat in _TIP_CATEGORIES:
        tip = gene.end_effector()
        if tip is not None:
            return tip.name
    if cat in _HEAD_CATEGORIES:
        head = next((s for s in gene.segments
                     if any(w in s.name.lower() for w in ("head", "skull", "sensor", "mast", "camera"))), None)
        if head is not None:
            return head.name
    return root_name


def _align_actuator_masses(gene, bom: dict) -> float:
    """Move each link's embodied motor mass onto the part the BOM's ``actuator_map`` actually ORDERS.

    Returns the net kilograms moved. The co-design loop the mass work was parked on in 2026-07 asked for
    exactly this direction of causality -- heavier robot -> bigger motor -> heavier robot -- and it terminates
    here rather than spiralling because the joint's REQUIREMENT is pinned (``torque_req_nm``): re-grounding a
    heavier body re-selects from the same requirement, never from the last motor's peak.
    """
    from virturoid.services.component_catalog import resolve_part
    amap = bom.get("actuator_map")
    if not isinstance(amap, dict) or not amap:
        return 0.0
    rec = _embodiment_record(gene)
    act = rec.get("actuator_kg") or {}
    moved = 0.0
    by_name = {s.name: s for s in gene.segments}
    for joint, part_name in amap.items():
        seg = by_name.get(str(joint))
        if seg is None or str(joint) not in act:      # not a link whose motor mass WE folded in
            continue
        part = resolve_part(str(part_name))
        mass = float(getattr(part, "mass_kg", 0.0) or 0.0) if part is not None else 0.0
        if mass <= 0.0:
            continue
        delta = mass - float(act[str(joint)] or 0.0)
        if abs(delta) < 1e-9:
            continue
        seg.mass_kg = round(max(0.02, float(seg.mass_kg or 0.0) + delta), 3)
        act[str(joint)] = round(mass, 6)
        moved += delta
    rec["actuator_kg"] = act
    return moved


def embody_component_masses(gene, bom: dict | None = None, *, task: str = "") -> dict:
    """Put the parts list's OWN battery, compute, sensors, wheels and gripper ONTO the simulated body.

    The gap this closes, MEASURED on this checkout before the change (per-body table in the task #211 report):
    an authored hexapod that simulates at 15.043 kg has 2.851 kg of named electronics -- an Intel RealSense, a
    Jetson Orin Nano, an IMU and a 2.6 kg LiPo -- that no link carries, so the walk verdict, the actuator sizing
    and the torque margins are all signed against a robot 19% lighter than the one a customer would bolt
    together. That is a sim-to-real gap we manufacture ourselves, and unlike the modelling gaps it is free to
    close: every one of those parts is already in ``bom.lines`` with a datasheet mass.

    IT IS NOT A CONSTANT. Each part goes on the link it physically rides on (:func:`_component_host`) -- battery
    and compute on the trunk, cameras on a head if the body has one, wheels on the wheels, the gripper on the
    end effector -- because a 2.6 kg pack on the trunk and the same pack on a foot are different robots.

    NOT ON AN IMPORTED ROBOT. A customer's model carries their manufacturer's per-link masses, which already
    include their battery and their compute; adding ours on top would silently re-weigh the machine they
    handed us, which is the guarantee ``preserve_mass`` exists to keep. Those bodies get ``applied: False`` with
    the reason, and their total is untouched (delta 0.0).

    IDEMPOTENT: a previous call's components are stripped before the new ones are added, and
    :func:`ground_gene` clears the ledger for every link whose mass it re-derived, so ground -> embody ->
    ground -> embody converges instead of ratcheting.

    Returns a disclosure dict: what was added, where, and what it did to the total. Never raises -- a parts list
    that cannot be built is reported as ``applied: False``, never as a body that silently weighs less.
    """
    out: dict = {"applied": False, "reason": "", "components_kg": 0.0, "total_mass_kg": [None, None],
                 "delta_kg": 0.0, "n_parts": 0, "parts": []}
    try:
        before = round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 3)
        out["total_mass_kg"] = [before, before]
        meta = getattr(gene, "metadata", None) or {}
        if str(meta.get("mass_source") or "") == "source_model":
            out["reason"] = ("this robot's per-link masses are the CUSTOMER'S OWN (mass_source=source_model), "
                             "and a manufacturer's link mass already includes that machine's battery, compute "
                             "and sensors -- bolting ours on top would re-weigh their robot")
            out["mass_preserved"] = True
            return out
        from virturoid.services.bom_builder import build_bom
        rec = _embodiment_record(gene)
        # REMEMBER WHICH TASK THE SUITE WAS CHOSEN FOR. ``build_bom``'s sensor selection is task-adaptive, so
        # the same body embodied for "navigate a warehouse" and re-embodied with no task carries different
        # hardware -- and the export door re-grounds WITHOUT a task and then compares physical fingerprints to
        # decide whether it is shipping the body the verdict was signed on. A silent LiDAR swap there would
        # read as "this is not the same robot". So a caller that does not say keeps the last thing that did.
        task = str(task or rec.get("task") or "")
        rec["task"] = task
        by_name = {s.name: s for s in gene.segments}
        realigned, itemised = 0.0, 0.0
        added: dict[str, float] = {}
        manifest: list[dict] = []
        # RUN TO A FIXED POINT, at most twice. ``build_bom`` selects the sensor suite from the body's SIZE
        # (``scale_kg``), so bolting 2.9 kg of hardware on can in principle move a body across a threshold and
        # buy it a different LiDAR -- a parts list that no longer describes the body it was applied to. Measured
        # on today's archetypes the first pass is already stable; the second exists so that a body near a
        # threshold converges instead of alternating, and it costs one BOM build (~0.4 ms).
        for _pass in range(2):
            for name, kg in list((rec.get("component_kg") or {}).items()):    # undo the previous pass first
                seg = by_name.get(str(name))
                if seg is not None:
                    seg.mass_kg = round(max(0.02, float(seg.mass_kg or 0.0) - float(kg or 0.0)), 3)
            rec["component_kg"], rec["components"] = {}, []
            if bom is None:
                bom = build_bom(gene, task=task)
            # THE BODY CARRIES THE MOTOR THE PARTS LIST ORDERS. ``ground_gene`` sizes each joint on its own and
            # folds THAT motor's mass into the link; ``bom_builder._standardize_actuators`` then rolls small
            # groups up onto fewer part numbers, and a roll-up only ever goes UP. So the robot a customer
            # procures is slightly heavier than the one sizing embodied -- measured: an authored horse 10.363 kg
            # of fitted motors against a 10.410 kg parts list, a 6-axis arm 1.658 against 1.978 (+19%). Small,
            # and exactly the kind of unexplained residue that makes "sim == BOM" a slogan instead of a check.
            realigned += _align_actuator_masses(gene, bom)
            added, manifest = {}, []
            for ln in (bom.get("lines") or []):
                if str(ln.get("category") or "") in _NON_COMPONENT_CATEGORIES:
                    continue
                try:
                    kg = float(ln.get("mass_kg") or 0.0)
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(kg) and kg > 0.0):
                    continue
                host = _component_host(gene, str(ln.get("category") or ""))
                if host not in by_name:
                    continue
                added[host] = added.get(host, 0.0) + kg
                manifest.append({"part": str(ln.get("part") or ""), "category": str(ln.get("category") or ""),
                                 "qty": int(ln.get("qty") or 1), "mass_kg": round(kg, 4), "link": host})
            for host, kg in added.items():
                by_name[host].mass_kg = round(float(by_name[host].mass_kg or 0.0) + kg, 3)
            rec["component_kg"] = {n: round(v, 6) for n, v in added.items()}
            rec["components"] = manifest
            itemised = sum(added.values())
            # The estimate and the itemised parts are two readings of the SAME quantity. Re-strike the estimate
            # so the finished mass is reached with more part numbers in it and less guesswork -- see
            # ``_apply_balance_of_system``. ``own_total`` is passed WITH any previous estimate still in it; the
            # helper strips its own record.
            own = sum(float(s.mass_kg or 0.0) for s in gene.segments) - itemised
            _apply_balance_of_system(gene, physical_prior_for(gene), own_total=own, itemised_kg=itemised)
            recheck = build_bom(gene, task=task)
            if abs(sum(float(ln.get("mass_kg") or 0.0) for ln in (recheck.get("lines") or [])
                       if str(ln.get("category") or "") not in _NON_COMPONENT_CATEGORIES) - itemised) < 1e-6:
                break
            bom = recheck                            # the suite moved with the mass -- settle on the new one
        after = round(sum(float(s.mass_kg or 0.0) for s in gene.segments), 3)
        out.update({"applied": True, "components_kg": round(itemised, 3), "n_parts": len(manifest),
                    "parts": manifest, "total_mass_kg": [before, after], "delta_kg": round(after - before, 3),
                    "hosts": {n: round(v, 3) for n, v in sorted(added.items())},
                    "actuator_realigned_kg": round(realigned, 3),
                    "reason": (f"{len(manifest)} parts from the bill of materials ({round(itemised, 3)} kg of "
                               f"battery, compute, sensors and drive hardware) now ride on the links that "
                               f"carry them, so the simulated body weighs what the built robot weighs")})
    except Exception as exc:  # noqa: BLE001 - a body that cannot be embodied is reported, never silently light
        out["reason"] = f"{type(exc).__name__}: {exc}"
        out["error"] = True
    return out


def _source_torque_note(seg, keep: float, where: dict) -> dict:
    """The BOM row's provenance for this joint's torque limit — measured (theirs) or sized (ours)."""
    if not keep:
        return {"torque_source": "sized_from_structural_load"}
    return {"torque_source": "source_model",
            "joint_torque_limit_nm": round(float(keep), 3),
            "declared_in": str(where.get(seg.name) or "the source model"),
            "part_role": "catalog EQUIVALENT for the customer's own actuator (they already own the motor); "
                         "the joint limit is theirs, the part is the smallest in our catalog that covers it"}
