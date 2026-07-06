"""Tier-3 NOVEL-body anatomy — recognizable procedural skeletons for bodies with no real-model analog.

The real-model substrate ([[real-model-substrate]]) covers humanoid/quadruped/arm/hand/drone/etc. at
production fidelity, but returns nothing for exotic morphologies (spider, frog, snake, hexapod). The old
procedural path built those as a flat slab with straight-down stubs — unrecognizable. This module is the
chosen tier-3 fallback (a 6-agent render bake-off picked improved procedural anatomy over kit-bashing real
part meshes and over generative-3D, which mispredicts joint axes): segmented SHAPED bodies (revolve/tapered
``geometry`` specs) + RADIALLY-fanned, outward-then-down limbs whose orientation is SOLVED (``_aim_euler``)
rather than guessed, + a baked rest pose. Each builder returns a valid ``RobotGene`` (compiles + stands) with
its rest pose stashed in ``gene.metadata['rest_pose']`` (joint_name -> angle), which the compiler emits as a
MuJoCo keyframe so the body shows in a real stance. Honest ceiling: clearly reads as the animal + stands +
physically valid — not photoreal (that is the real-model tier's job).
"""

from __future__ import annotations

import math

from virturoid.schemas.gene import GeneSegment, RobotGene

PI = math.pi


def _aim_euler(direction):
    """xyz-intrinsic euler that aims a limb's local +z along world ``direction``, with a horizontal local +y
    (so a knee hinge about +y bends in the vertical plane). We say WHERE a segment points and solve the
    angles, instead of guessing — the fix for limbs that pointed the wrong way."""
    import numpy as np

    z = np.array(direction, float)
    z /= (np.linalg.norm(z) or 1.0)
    up = np.array([0.0, 0.0, 1.0])
    y = np.cross(up, z)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= (np.linalg.norm(x) or 1.0)
    R = np.column_stack([x, y, z])
    b = math.asin(max(-1.0, min(1.0, R[0, 2])))
    if abs(math.cos(b)) > 1e-6:
        a = math.atan2(-R[1, 2], R[2, 2])
        c = math.atan2(-R[0, 1], R[0, 0])
    else:
        a = math.atan2(R[2, 1], R[1, 1])
        c = 0.0
    return (a, b, c)


def _revolve(profile):
    """Solid of revolution about local +z from a ``[(r, z), ...]`` profile (m) — rounded organic shells."""
    return {"family": "revolve", "profile": [[round(r, 5), round(z, 5)] for r, z in profile]}


def _tapered(length, r0, r1):
    """A frustum limb segment (thick proximal -> thin distal): real legs taper."""
    return {"family": "tapered", "length": round(length, 5), "r0": round(r0, 5), "r1": round(r1, 5)}


def _ellipsoid_profile(length, max_r, *, bulge=0.5, n=7):
    """A teardrop (r, z) profile, widest at ``bulge*length`` — spider abdomen / frog body."""
    pts = []
    for i in range(n + 1):
        z = length * i / n
        t = z / length
        shape = math.sin(math.pi * min(1.0, (t / (2 * bulge)) if t < bulge else
                                       (0.5 + 0.5 * (t - bulge) / (1 - bulge))))
        pts.append((max(0.004, max_r * (0.18 + 0.82 * shape)), z))
    return pts


def build_spider(n_legs: int = 8):
    """A recognizable spider: low cephalothorax (carries the legs + a head stub) + bulbous abdomen (rear),
    with ``n_legs`` long two-joint legs radiating outward, fanned front->back, knees bent down."""
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}
    ceph_r, ceph_h = 0.075, 0.045
    segs.append(GeneSegment(
        name="cephalothorax", parent=None, shape="cylinder", length_m=ceph_h, radius_m=ceph_r,
        mass_kg=0.6, joint_type=None,
        geometry=_revolve([(0.30 * ceph_r, 0), (0.92 * ceph_r, 0.45 * ceph_h),
                           (0.80 * ceph_r, 0.85 * ceph_h), (0.45 * ceph_r, ceph_h)])))
    ab_len, ab_r = 0.14, 0.085
    segs.append(GeneSegment(
        name="abdomen", parent="cephalothorax", shape="capsule", length_m=ab_len, radius_m=ab_r,
        mass_kg=1.0, joint_type=None, mount_offset=(-0.9 * ceph_r, 0.0, 0.4 * ceph_h),
        mount_euler=(0.0, PI / 2 + 0.35, PI),
        geometry=_revolve(_ellipsoid_profile(ab_len, ab_r, bulge=0.62))))
    head_len = 0.05
    segs.append(GeneSegment(
        name="head", parent="cephalothorax", shape="capsule", length_m=head_len, radius_m=0.030,
        mass_kg=0.08, joint_type=None, mount_offset=(0.85 * ceph_r, 0.0, 0.25 * ceph_h),
        mount_euler=(0.0, PI / 2 + 0.5, 0.0),
        geometry=_revolve([(0.030, 0), (0.022, 0.5 * head_len), (0.010, head_len)])))
    per_side = n_legs // 2
    femur, tibia = 0.13, 0.15
    f_r, t_r = 0.014, 0.010
    femur_el = 0.18
    fan = [(0.95 - 1.9 * i / max(1, per_side - 1)) for i in range(per_side)]
    coxa_len = 0.07     # long enough to clear the leg-slenderness band (aspect = 0.07/2r ~ 2.5 >= 2.2)
    for side, sy in (("l", 1.0), ("r", -1.0)):
        for i, f in enumerate(fan):
            phi = sy * (PI / 2) - f
            mx, my = ceph_r * 0.9 * math.cos(phi), ceph_r * 0.9 * math.sin(phi)
            coxa_name = f"leg_{side}{i}_coxa"
            femur_name, tibia_name = f"leg_{side}{i}_femur", f"leg_{side}{i}_tibia"
            dir_out = (math.cos(phi) * math.cos(femur_el), math.sin(phi) * math.cos(femur_el),
                       math.sin(femur_el))
            # 3-DOF insect leg (coxa YAW + femur PITCH + tibia KNEE) — a 2-DOF leg (the old build) can't lift a
            # foot to swing AND sweep it for propulsion, so the trot gait only paddled. The coxa yaws the leg
            # around vertical (abduction) and aims the chain outward; femur+tibia pitch in that plane to step.
            segs.append(GeneSegment(
                name=coxa_name, parent="cephalothorax", shape="capsule", length_m=coxa_len, radius_m=f_r,
                mass_kg=0.03, joint_type="revolute", joint_axis=(0.0, 0.0, 1.0),
                joint_lower=-0.6, joint_upper=0.6, mount_offset=(mx, my, 0.45 * ceph_h),
                mount_euler=_aim_euler(dir_out), actuator_torque_nm=4.0,
                geometry=_tapered(coxa_len, f_r * 1.4, f_r)))
            segs.append(GeneSegment(
                name=femur_name, parent=coxa_name, shape="capsule", length_m=femur, radius_m=f_r,
                mass_kg=0.05, joint_type="revolute", joint_axis=(0.0, 1.0, 0.0),
                joint_lower=-0.9, joint_upper=0.9, actuator_torque_nm=4.0,
                geometry=_tapered(femur, f_r * 1.5, f_r)))
            segs.append(GeneSegment(
                name=tibia_name, parent=femur_name, shape="capsule", length_m=tibia, radius_m=t_r,
                mass_kg=0.035, joint_type="revolute", joint_axis=(0.0, 1.0, 0.0),
                joint_lower=-0.2, joint_upper=2.6, actuator_torque_nm=3.0,
                is_end_effector=(side == "l" and i == 0),
                geometry=_tapered(tibia, t_r * 1.3, t_r * 0.7)))
            pose[f"{coxa_name}_joint"] = 0.0
            pose[f"{femur_name}_joint"] = 0.6    # hip pitched down so the knee reaches toward the ground
            pose[f"{tibia_name}_joint"] = 1.4
    gene = RobotGene(id="proc_spider", species="arachnid.proc", robot_class="legged",
                     segments=segs, base_mount="free", end_effector_type="none",
                     metadata={"rest_pose": pose})
    return gene


def _loft(sections):
    return {"family": "loft", "sections": [[float(z), float(hy), float(hx)] for z, hy, hx in sections]}


def build_dog(scale: float = 1.0):
    """A recognizable DOG / mammal quadruped — what a generic flat-disc 'quadruped.composed' template misses.

    The canine silhouette comes from anatomy a generic 4-legged template has none of: an ELONGATED, NARROW,
    low barrel body (long front-to-back, ~half as wide), a forward NECK + HEAD + SNOUT, a TAIL, upright ears,
    and fore/hind legs at the body's four corners (not bunched into a square). Body is a horizontal loft;
    head/snout/tail/ears are orientation-solved with ``_aim_euler``; legs are 2 actuated joints + a paw, with
    a baked standing rest pose. ``scale`` multiplies all linear dims (a chihuahua vs a mastiff)."""
    s = max(0.5, min(2.0, float(scale)))
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}
    body_h = 0.135 * s          # body height (the loft's local +z extent)
    half_len = 0.30 * s         # body half-length front-to-back (x)
    half_w = 0.085 * s          # body half-width (y) — NARROW: a dog is ~half as wide as long
    # TORSO (root): a long, narrow, rounded barrel. loft sections [z, half_y(width), half_x(front-back)] give
    # a body that is LONG + NARROW + low — a dog's profile, not a wide disc. radius_m (the box collider /
    # spawn footprint) tracks the true body half-width, so the collider isn't a giant flat plate.
    segs.append(GeneSegment(
        name="torso", parent=None, shape="box", length_m=body_h, radius_m=half_w, mass_kg=6.0 * s ** 3,
        joint_type=None, geometry=_loft([
            (0.0,          half_w * 0.78, half_len * 0.80),    # belly
            (0.40 * body_h, half_w * 1.00, half_len * 1.00),   # widest/longest mid barrel
            (0.78 * body_h, half_w * 0.92, half_len * 0.94),   # back/shoulders
            (body_h,        half_w * 0.66, half_len * 0.78)])))   # spine ridge
    # NECK + HEAD + SNOUT at the FRONT (+x), aimed up-and-forward then forward — the single biggest "dog" cue.
    neck_len = 0.12 * s
    segs.append(GeneSegment(
        name="neck", parent="torso", shape="capsule", length_m=neck_len, radius_m=0.045 * s, mass_kg=0.5 * s ** 3,
        joint_type=None, mount_offset=(half_len * 0.86, 0.0, 0.62 * body_h),
        mount_euler=_aim_euler((0.80, 0.0, 0.60)),
        geometry=_tapered(neck_len, 0.055 * s, 0.05 * s)))
    head_len = 0.11 * s
    segs.append(GeneSegment(
        name="head", parent="neck", shape="capsule", length_m=head_len, radius_m=0.052 * s, mass_kg=0.5 * s ** 3,
        joint_type=None, mount_euler=_aim_euler((0.92, 0.0, -0.30)),
        geometry=_revolve([(0.030 * s, 0), (0.052 * s, 0.40 * head_len), (0.050 * s, 0.72 * head_len),
                           (0.040 * s, head_len)])))
    snout_len = 0.06 * s
    segs.append(GeneSegment(
        name="snout", parent="head", shape="capsule", length_m=snout_len, radius_m=0.030 * s, mass_kg=0.1 * s ** 3,
        joint_type=None, mount_offset=(0.0, 0.0, 0.62 * head_len), mount_euler=_aim_euler((0.95, 0.0, -0.05)),
        geometry=_revolve([(0.032 * s, 0), (0.030 * s, 0.55 * snout_len), (0.024 * s, snout_len)])))
    for ey, name in ((1.0, "ear_l"), (-1.0, "ear_r")):     # upright triangular ears
        segs.append(GeneSegment(
            name=name, parent="head", shape="capsule", length_m=0.05 * s, radius_m=0.016 * s, mass_kg=0.03 * s ** 3,
            joint_type=None, mount_offset=(ey * 0.030 * s, -0.018 * s, 0.34 * head_len),
            mount_euler=_aim_euler((0.10, ey * 0.35, 0.93)),
            geometry=_tapered(0.05 * s, 0.018 * s, 0.004 * s)))
    # TAIL at the REAR (-x), aimed back-and-up (a relaxed dog tail).
    tail_len = 0.16 * s
    segs.append(GeneSegment(
        name="tail", parent="torso", shape="capsule", length_m=tail_len, radius_m=0.022 * s, mass_kg=0.2 * s ** 3,
        joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.0, joint_upper=1.0,
        mount_offset=(-half_len * 0.84, 0.0, 0.72 * body_h), mount_euler=_aim_euler((-0.80, 0.0, 0.60)),
        actuator_torque_nm=2.0, geometry=_tapered(tail_len, 0.026 * s, 0.010 * s)))
    pose["tail_joint"] = 0.0
    # FOUR LEGS at the body's corners: front pair near +x, hind near -x; thigh + shank + paw, pointing DOWN.
    thigh_len, shank_len, paw_len = 0.135 * s, 0.135 * s, 0.05 * s
    for tag, fx in (("front", 0.62 * half_len), ("hind", -0.66 * half_len)):
        for side, sy in (("l", 1.0), ("r", -1.0)):
            th, sh, pw = f"{tag}_{side}_thigh", f"{tag}_{side}_shank", f"{tag}_{side}_paw"
            segs.append(GeneSegment(
                name=th, parent="torso", shape="capsule", length_m=thigh_len, radius_m=0.035 * s,
                mass_kg=0.6 * s ** 3, joint_type="revolute", joint_axis=(0.0, 1.0, 0.0),
                joint_lower=-2.0, joint_upper=2.0, mount_offset=(fx, sy * half_w * 0.92, 0.02 * body_h),
                mount_euler=_aim_euler((0.0, sy * 0.10, -1.0)), actuator_torque_nm=18.0 * s ** 2,
                geometry=_tapered(thigh_len, 0.045 * s, 0.034 * s)))
            segs.append(GeneSegment(
                name=sh, parent=th, shape="capsule", length_m=shank_len, radius_m=0.028 * s,
                mass_kg=0.4 * s ** 3, joint_type="revolute", joint_axis=(0.0, 1.0, 0.0),
                joint_lower=-2.4, joint_upper=2.4, actuator_torque_nm=14.0 * s ** 2,
                geometry=_tapered(shank_len, 0.030 * s, 0.022 * s)))
            segs.append(GeneSegment(
                name=pw, parent=sh, shape="capsule", length_m=paw_len, radius_m=0.030 * s, mass_kg=0.15 * s ** 3,
                joint_type=None, is_end_effector=(tag == "front" and side == "l"),
                mount_euler=_aim_euler((0.6, 0.0, -0.2)),
                geometry=_tapered(paw_len, 0.032 * s, 0.028 * s)))
            # rest pose: a gentle stand — knees slightly bent (front fwd, hind back) so it stands, not splays.
            pose[f"{th}_joint"] = (-0.25 if tag == "front" else 0.25)
            pose[f"{sh}_joint"] = (0.45 if tag == "front" else -0.45)
    return RobotGene(id="proc_dog", species="mammal.dog.proc", robot_class="quadruped",
                     segments=segs, base_mount="free", end_effector_type="none",
                     metadata={"rest_pose": pose})


def build_hexapod():
    """A six-legged insect: the spider engine at n=6 (classic insect splay)."""
    g = build_spider(n_legs=6)
    g.id, g.species = "proc_hexapod", "insect.proc"
    return g


def build_snake(n_beads: int = 16):
    """A snake: a tapered chain of rounded body beads, alternating yaw joints -> serpentine S at rest."""
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}
    head_len, head_r = 0.055, 0.034
    segs.append(GeneSegment(
        name="head", parent=None, shape="capsule", length_m=head_len, radius_m=head_r, mass_kg=0.15,
        joint_type=None, mount_euler=(0.0, PI / 2, 0.0),
        geometry=_revolve([(0.022, 0), (0.034, 0.45 * head_len), (0.030, 0.8 * head_len),
                           (0.016, head_len)])))
    prev, seg_len = "head", 0.07
    for i in range(n_beads):
        t = i / max(1, n_beads - 1)
        r = (1.0 - 0.55 * t) * 0.030 + 0.006
        name = f"body_{i}"
        segs.append(GeneSegment(
            name=name, parent=prev, shape="capsule", length_m=seg_len, radius_m=r,
            mass_kg=max(0.02, 0.12 * (1 - 0.5 * t)),
            joint_type="revolute", joint_axis=(-1.0, 0.0, 0.0),   # world-up yaw (chain aimed along +x)
            joint_lower=-0.8, joint_upper=0.8, actuator_torque_nm=2.5,
            is_end_effector=(i == n_beads - 1),
            geometry=_revolve([(0.6 * r, 0), (r, 0.5 * seg_len), (0.85 * r, seg_len)])))
        pose[f"{name}_joint"] = 0.7 * math.sin(i * 0.55)
        prev = name
    gene = RobotGene(id="proc_snake", species="serpent.proc", robot_class="legged",
                     segments=segs, base_mount="free", end_effector_type="none",
                     metadata={"rest_pose": pose})
    return gene


def build_frog():
    """A frog: wide streamlined body + broad head + bulging eyes, short folded forelegs + long Z-folded
    powerful hind legs (the crouched-spring silhouette)."""
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}
    body_h, body_r = 0.085, 0.075
    segs.append(GeneSegment(
        name="body", parent=None, shape="capsule", length_m=body_h, radius_m=body_r, mass_kg=0.8,
        joint_type=None,
        geometry=_revolve([(0.55 * body_r, 0), (body_r, 0.30 * body_h), (0.95 * body_r, 0.65 * body_h),
                           (0.55 * body_r, body_h)])))
    head_len = 0.07
    segs.append(GeneSegment(
        name="head", parent="body", shape="capsule", length_m=head_len, radius_m=0.06, mass_kg=0.2,
        joint_type=None, mount_offset=(0.75 * body_r, 0.0, -0.45 * body_h),
        mount_euler=_aim_euler((0.95, 0.0, 0.10)),
        geometry=_revolve([(0.042, 0), (0.060, 0.4 * head_len), (0.052, 0.75 * head_len),
                           (0.024, head_len)])))
    for ey, name in ((1.0, "eye_l"), (-1.0, "eye_r")):
        segs.append(GeneSegment(
            name=name, parent="head", shape="capsule", length_m=0.022, radius_m=0.019, mass_kg=0.01,
            joint_type=None, mount_offset=(ey * 0.034, 0.030, head_len * 0.30),
            mount_euler=_aim_euler((0.30 * ey, 0.30, 0.90)),
            geometry=_revolve([(0.019, 0), (0.019, 0.012), (0.011, 0.022)])))
    for side, sy in (("l", 1.0), ("r", -1.0)):
        up, lo = f"fore_{side}_upper", f"fore_{side}_lower"
        segs.append(GeneSegment(
            name=up, parent="body", shape="capsule", length_m=0.05, radius_m=0.013, mass_kg=0.05,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.5, joint_upper=1.5,
            mount_offset=(0.5 * body_r, sy * 0.62 * body_r, -0.25 * body_h),
            mount_euler=_aim_euler((0.45, sy * 0.35, -0.82)), actuator_torque_nm=2.0,
            geometry=_tapered(0.05, 0.014, 0.010)))
        segs.append(GeneSegment(
            name=lo, parent=up, shape="capsule", length_m=0.045, radius_m=0.010, mass_kg=0.03,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.6, joint_upper=1.6,
            actuator_torque_nm=1.5, is_end_effector=(side == "l"),
            geometry=_tapered(0.045, 0.011, 0.008)))
        pose[f"{up}_joint"], pose[f"{lo}_joint"] = 0.0, 0.4
    for side, sy in (("l", 1.0), ("r", -1.0)):
        thigh, shank, foot = f"hind_{side}_thigh", f"hind_{side}_shank", f"hind_{side}_foot"
        segs.append(GeneSegment(
            name=thigh, parent="body", shape="capsule", length_m=0.085, radius_m=0.024, mass_kg=0.16,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-2.4, joint_upper=2.4,
            mount_offset=(-0.5 * body_r, sy * 0.7 * body_r, 0.05 * body_h),
            mount_euler=_aim_euler((-0.35, sy * 0.78, 0.52)), actuator_torque_nm=7.0,
            geometry=_tapered(0.085, 0.028, 0.017)))
        segs.append(GeneSegment(
            name=shank, parent=thigh, shape="capsule", length_m=0.085, radius_m=0.016, mass_kg=0.09,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-0.2, joint_upper=2.8,
            actuator_torque_nm=5.0, geometry=_tapered(0.085, 0.018, 0.012)))
        segs.append(GeneSegment(
            name=foot, parent=shank, shape="capsule", length_m=0.06, radius_m=0.011, mass_kg=0.03,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-2.6, joint_upper=0.4,
            actuator_torque_nm=2.0, geometry=_tapered(0.06, 0.013, 0.009)))
        pose[f"{thigh}_joint"], pose[f"{shank}_joint"], pose[f"{foot}_joint"] = 0.0, 2.5, -2.0
    gene = RobotGene(id="proc_frog", species="amphibian.proc", robot_class="legged",
                     segments=segs, base_mount="free", end_effector_type="none",
                     metadata={"rest_pose": pose})
    return gene


# prompt-word -> archetype builder. Reserved for DISTINCTIVE morphologies with no good walker-template or
# real-model equivalent. Hexapod/octopod are deliberately NOT here: "an N-legged WALKING robot" wants the
# parametric, gait-tuned 3-DOF-per-leg legged template (trainable locomotion), which morphology_composer
# already builds — routing it through a decorative archetype would lose that.
# ONLY genuinely distinctive SHAPES the general anatomy compiler can't yet express (a legless articulated
# serpent, an 8-radial-leg arachnid, a leaping amphibian). A "dog" / "cat" / "horse" is deliberately NOT here:
# routing common quadrupeds to a hand-coded builder is the per-species OVERFITTING we reject — they go to the
# general anatomy compiler (LLM-specific online, a generic quadruped offline). build_dog stays only as a
# library function (not wired into the offline router).
_ARCHETYPES = (
    (("spider", "arachnid", "tarantula"), build_spider),
    (("snake", "serpent", "serpentine", "limbless"), build_snake),
    (("frog", "amphibian", "toad"), build_frog),
)


def novel_archetype_gene(prompt: str):
    """Return a posed ``RobotGene`` for a DISTINCTIVE exotic body the real-model library and the parametric
    legged template can't capture (spider/snake/frog/dog), else ``None`` so the normal composer path handles
    it. OFFLINE/keyword fallback only — with an LLM the general anatomy compiler designs these intelligently.

    Matches on WORD BOUNDARIES (``\\bword\\b``), not substrings: a bare ``w in prompt`` made 'cat' match
    'deli**cat**e' and route a dexterous-hand request to a dog. Word boundaries fix that for every archetype."""
    import re

    p = (prompt or "").lower()
    # GEN-10 (docs/generality_plan.md): a MANY-legged body ("centipede", "millipede", or an explicit "N legs"
    # with N>=10) must not collapse to the 4-leg generic quad (measured: offline lizard==centipede==horse). Route
    # it to the PARAMETRIC radial-leg builder at that leg count. This is a leg-COUNT parameter, not a per-species
    # shape, so it stays consistent with the anti-overfitting rule (a dog/horse still goes to the general compiler).
    mm = re.search(r"(\d+)[\s-]*(?:legs?|legged)\b", p)
    n_many = int(mm.group(1)) if mm else (14 if re.search(r"\bcentipede\b", p) else
                                          (16 if re.search(r"\bmillipede\b", p) else 0))
    if n_many >= 10:
        return build_spider(n_legs=min(n_many - (n_many % 2), 16))
    for words, builder in _ARCHETYPES:
        if any(re.search(rf"\b{re.escape(w)}\b", p) for w in words):
            return builder()
    return None
