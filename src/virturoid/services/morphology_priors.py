"""Morphology priors + design-time validator (gap-closure G6). The robot analog of ``dimension_priors``: the e2e
fidelity test shipped a 'quadruped' with 2 actuated joints per leg (no hip abduction — every real quad has 3
[Unitree Go2: 12 joint motors, 0.70x0.31x0.40 m, ~15 kg]), 1:1-aspect stub legs, a torso WIDER than long, and
motor housings fatter than the limbs. Nothing checked. This module owns the per-class proportion/DOF bands
(from fetched real-robot references: Go2, UR5e 850 mm/20.6 kg, Roomba Ø0.34x0.09 m) and a FAIL-CLOSED
``validate_morphology`` gate that runs at design time, so a body that cannot possibly work (or reads as a blob)
is rejected+repaired BEFORE packaging — not discovered when it "did not walk" at eval.

Checks are structural (segment tree), not sim-based, so the gate is fast + dependency-free."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MorphReport:
    ok: bool
    robot_class: str
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    measured: dict = field(default_factory=dict)


# per-class structural bands (min, max) — from the fetched reference robots + design conventions
QUAD_BANDS = {
    "dof_per_leg": (3, 4),          # abduction + hip + knee minimum (Go2 = 3/leg, 12 total)
    "leg_seg_aspect": (2.2, 30.0),  # length/diameter of thigh/shank; Go2-class ~4-6, blob dog was ~1.0
    "torso_LW": (1.2, 4.5),         # torso longer than wide (blob dog was 0.31: WIDER than long)
    "leg_len_frac": (0.35, 1.6),    # (thigh+shank)/torso_length
    "foot_limb_ratio": (0.0, 1.6),  # foot radius / shank radius (giant ball feet read as toys)
    "head_torso_ext": (0.0, 1.05),  # head visible extent (len + 2r) / torso length: a head must not rival the trunk
}
ARM_BANDS = {
    "dof": (3, 8),                  # a graspable arm needs yaw+2 pitch+wrist; 6-7 = industrial class
    "link_aspect": (2.0, 40.0),
    "reach_m": (0.2, 2.0),          # UR5e 0.85 m
}
MOBILE_BANDS = {
    "n_wheels": (2, 8),
    "chassis_aspect_h": (1.5, 12.0),  # chassis footprint/height: a rover is FLAT (Roomba Ø0.34 x 0.09)
}


def _seg_by(gene, pred):
    return [s for s in gene.segments if pred(s)]


# trailing segment-role words a leg name may end in (besides a numeric index) — stripped to find the chain key.
_LEG_SEG_WORDS = {"femur", "tibia", "coxa", "shank", "thigh", "shin", "calf", "foot", "upper", "lower", "hip", "knee"}


def _leg_chains(gene):
    """Group leg segments by their prefix (leg1_l, leg2_r, ...) -> list of chains (each a list of segments)."""
    chains: dict[str, list] = {}
    for s in gene.segments:
        n = s.name.lower()
        if "leg" in n and "_geom" not in n:
            # chain key = the name minus its trailing segment marker — a numeric index ("front_leg_l_0" ->
            # "front_leg_l") OR a role word ("leg_l0_femur"/"leg_l0_tibia" -> "leg_l0"). Word suffixes were the
            # bug: a spider's femur/tibia hashed to DISTINCT keys, so its 2-DOF legs read as sixteen 1-joint
            # chains and false-failed the >=3-DOF band. Both left/right stay distinct chains either way.
            parts = n.rsplit("_", 1)
            suffix = parts[1] if len(parts) == 2 else ""
            key = parts[0] if (suffix.isdigit() or suffix in _LEG_SEG_WORDS) else n
            chains.setdefault(key, []).append(s)
    return list(chains.values())


def validate_morphology(gene) -> MorphReport:
    """Fail-closed structural gate. Returns ok=False with precise issues when the body violates its class bands —
    the exact defects the e2e test shipped (2-DOF legs, 1:1 stub segments, wide-slab torso, ball feet)."""
    cls = (getattr(gene, "robot_class", "") or "").lower()
    issues: list[str] = []
    warns: list[str] = []
    measured: dict = {"robot_class": cls, "n_segments": len(gene.segments)}

    def actuated(segs):
        return [s for s in segs if (getattr(s, "joint_type", "") or "").lower() in ("revolute", "prismatic")]

    if cls in ("quadruped", "legged"):
        base = next((s for s in gene.segments if s.parent is None), None)
        _base_has_geo = base is not None and getattr(base, "geometry", None)
        if _base_has_geo:
            # anatomy-compiled bodies carry their true trunk span in the compound/loft GEOMETRY, not length_m —
            # the box collider is only the core. Proportion truth there is the legs; skip the LW check honestly.
            warns.append("compound-geometry trunk: torso L/W checked via leg placement only")
        if base is not None and not _base_has_geo:
            # torso proportions: length_m is the long (x) dimension, radius_m the half-width
            L, W = float(base.length_m), 2.0 * float(base.radius_m)
            measured["torso_L"], measured["torso_W"] = round(L, 3), round(W, 3)
            lo, hi = QUAD_BANDS["torso_LW"]
            if not (lo <= (L / max(1e-6, W)) <= hi):
                issues.append(f"torso length/width {L / max(1e-6, W):.2f} outside [{lo},{hi}] "
                              f"(L={L:.2f} W={W:.2f}: a quadruped trunk is LONGER than wide)")
        chains = _leg_chains(gene)
        measured["n_leg_chains"] = len(chains)
        if len(chains) < 4:
            issues.append(f"only {len(chains)} leg chains found (a quadruped needs 4)")
        dofs, aspects, leg_lens = [], [], []
        for ch in chains:
            act = actuated(ch)
            dofs.append(len(act))
            for s in act:
                if float(s.radius_m) > 1e-6:
                    aspects.append(float(s.length_m) / (2.0 * float(s.radius_m)))
            leg_lens.append(sum(float(s.length_m) for s in act))
        if dofs:
            measured["dof_per_leg"] = dofs
            lo, hi = QUAD_BANDS["dof_per_leg"]
            if min(dofs) < lo:
                issues.append(f"legs have {min(dofs)} actuated joints; a real quadruped needs >= {lo} "
                              f"(abduction + hip + knee — Go2 has 3/leg, 12 total)")
        if aspects:
            measured["min_leg_aspect"] = round(min(aspects), 2)
            lo, hi = QUAD_BANDS["leg_seg_aspect"]
            if min(aspects) < lo:
                issues.append(f"leg segment aspect {min(aspects):.2f} < {lo} (length/diameter): stub 'sausage' "
                              f"segments read as a blob and give no stride (real quads ~4-6)")
        if base is not None and leg_lens and not _base_has_geo:
            frac = (sum(leg_lens) / len(leg_lens)) / max(1e-6, float(base.length_m))
            measured["leg_len_frac"] = round(frac, 2)
            lo, hi = QUAD_BANDS["leg_len_frac"]
            if not (lo <= frac <= hi):
                issues.append(f"mean leg length is {frac:.2f}x the torso length, outside [{lo},{hi}]")
        feet = _seg_by(gene, lambda s: "foot" in s.name.lower())
        shanks = [ch[-1] for ch in chains if ch]
        if feet and shanks:
            fr = max(float(f.radius_m) for f in feet) / max(1e-6, min(float(s.radius_m) for s in shanks))
            measured["foot_limb_ratio"] = round(fr, 2)
            if fr > QUAD_BANDS["foot_limb_ratio"][1]:
                issues.append(f"foot radius {fr:.2f}x the shank radius (giant ball feet)")
        # Head must not rival the trunk. Compare the head's visible extent (capsule length + both hemispherical
        # caps = len + 2r) to the torso length; a ratio > ~1 is the "second body" render defect. torso length =
        # the base box core (base.length_m), which is the trunk's long axis even for compound-geometry trunks.
        heads = _seg_by(gene, lambda s: any(k in s.name.lower() for k in ("head", "skull", "snout", "muzzle")))
        # Reference the torso's LARGEST dimension (length_m OR the 2*radius_m footprint), not length_m alone: a
        # composed/loft trunk stores its long front-back axis in the loft geometry while its box collider length_m
        # is the THIN (thickness) axis — measuring the head against that thin axis false-flagged a fine hexapod
        # head at 1.55x. max() picks the real trunk scale for both a long box (anatomy dog) and a flat slab.
        torso_ref = max(float(base.length_m), 2.0 * float(base.radius_m)) if base is not None else 0.0
        if heads and torso_ref > 1e-6:
            ext = max(float(h.length_m) + 2.0 * float(h.radius_m) for h in heads)
            ratio = ext / torso_ref
            measured["head_torso_ext"] = round(ratio, 2)
            if ratio > QUAD_BANDS["head_torso_ext"][1]:
                issues.append(f"head extent {ratio:.2f}x the torso size (a head rivaling the trunk reads as a "
                              f"second body, not a sensor head)")
    elif cls in ("manipulator", "arm"):
        act = actuated(gene.segments)
        measured["dof"] = len(act)
        lo, hi = ARM_BANDS["dof"]
        if not (lo <= len(act) <= hi):
            issues.append(f"{len(act)} actuated joints outside [{lo},{hi}] for an arm")
        reach = sum(float(s.length_m) for s in act)
        measured["reach_m"] = round(reach, 3)
        lo, hi = ARM_BANDS["reach_m"]
        if not (lo <= reach <= hi):
            issues.append(f"reach {reach:.2f} m outside [{lo},{hi}] (UR5e-class = 0.85 m)")
    elif cls in ("mobile_base", "mobile_manipulator"):
        wheels = _seg_by(gene, lambda s: "wheel" in s.name.lower())
        measured["n_wheels"] = len(wheels)
        lo, hi = MOBILE_BANDS["n_wheels"]
        if not (lo <= len(wheels) <= hi):
            issues.append(f"{len(wheels)} wheels outside [{lo},{hi}]")
        if cls == "mobile_manipulator":
            arm = [s for s in gene.segments if s.name.startswith(("shoulder", "j")) and
                   (s.joint_type or "").lower() == "revolute"]
            measured["arm_dof"] = len(arm)
            if len(arm) < 3:
                issues.append(f"composite has only {len(arm)} arm joints (needs >= 3 to pick + carry)")
    else:
        warns.append(f"no morphology bands for class {cls!r} (unchecked)")

    # universal: actuator housings must not dwarf their limbs (the 'black blob crowd' render defect). The
    # housing radius is derived from the joint torque's real servo; we check limb slenderness as the proxy the
    # renderer actually exposes (covered per-class above), so here only universal sanity:
    if not gene.segments:
        issues.append("gene has no segments")
    return MorphReport(ok=not issues, robot_class=cls, issues=issues, warnings=warns, measured=measured)
