"""METHOD C — IMPROVED PROCEDURAL ANATOMY for novel bodies with no real model.

Standalone experiment (does NOT edit shared source). Prototypes a *better* procedural generator for
bodies the real-model substrate can't supply (spider, frog, snake, hexapod). The current path
(morphology_composer.compose_robot, offline) builds these from naked primitives mounted straight-down
under a flat slab — the result reads as a white box with stubby legs, not a spider.

What this prototype changes vs the baseline procedural path:
  1. SEGMENTED, SHAPED bodies — a spider gets a cephalothorax + bulbous abdomen built from `revolve`
     geometry specs (cad_geometry.realize_shape), not one box. A frog gets a streamlined revolved body
     + a wide head. A snake is a tapered chain of beads.
  2. RADIAL / SPLAYED limb mounting — legs mount around the body perimeter with a per-leg YAW (so they
     fan out front→back like a real arachnid) AND an outward ROLL via mount_euler, so the first segment
     reaches OUT and the knee bends DOWN. This is the single biggest "reads as a spider" cue and the
     thing the baseline got wrong (all legs straight down, bunched).
  3. ANATOMICAL PROPORTIONS — long thin two-segment legs (femur reaching up-and-out, tibia angling down)
     with real length ratios; a frog gets short folded forelegs + long powerful hind legs; a snake gets
     ~16 graded body beads.
  4. A baked REST POSE (per-joint angles by role) so the body renders in a recognizable stance instead
     of a default straight-out star. Pose is applied at render time only (physics/gene unchanged).

Each body is a valid `RobotGene` that compiles + stands in MuJoCo. Render with the bottom of this file.

    PYTHONPATH=src python scripts/overhaul/proc_anatomy.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from virturoid.schemas.gene import GeneSegment, RobotGene  # noqa: E402

PI = math.pi


# ----------------------------------------------------------------------------------------------------
# orientation helper: aim a limb's local +z along a chosen world direction, with a controlled "down"
# ----------------------------------------------------------------------------------------------------
def _aim_euler(direction: tuple[float, float, float]) -> tuple[float, float, float]:
    """xyz-intrinsic euler (matching MuJoCo's default eulerseq) that rotates the limb's LOCAL +z onto the
    unit world ``direction``, choosing a roll so the limb's local +y stays as horizontal as possible (so a
    knee hinge about local +y bends in the vertical plane). This removes the guesswork that made the
    baseline legs point the wrong way: we say WHERE the segment should point and solve the angles.

    Build R with columns [x_hat, y_hat, z_hat] where z_hat=direction, y_hat=horizontal & perpendicular to
    z_hat, x_hat=y_hat x z_hat. Then extract xyz-intrinsic euler from R.
    """
    import numpy as np

    z = np.array(direction, float)
    z /= (np.linalg.norm(z) or 1.0)
    up = np.array([0.0, 0.0, 1.0])
    # y_hat: horizontal, perpendicular to z (the knee axis). If z is vertical, fall back to world +y.
    y = np.cross(up, z)
    if np.linalg.norm(y) < 1e-6:
        y = np.array([0.0, 1.0, 0.0])
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    x /= (np.linalg.norm(x) or 1.0)
    R = np.column_stack([x, y, z])
    # xyz-intrinsic euler from R (R = Rx(a) @ Ry(b) @ Rz(c))
    b = math.asin(max(-1.0, min(1.0, R[0, 2])))
    if abs(math.cos(b)) > 1e-6:
        a = math.atan2(-R[1, 2], R[2, 2])
        c = math.atan2(-R[0, 1], R[0, 0])
    else:  # gimbal lock
        a = math.atan2(R[2, 1], R[1, 1])
        c = 0.0
    return (a, b, c)


# ----------------------------------------------------------------------------------------------------
# geometry-spec helpers (consumed by cad_geometry.realize_shape via the meshed compile path)
# ----------------------------------------------------------------------------------------------------
def _revolve(profile: list[tuple[float, float]]) -> dict:
    """A solid of revolution about local +z from a [(r, z), ...] profile (meters). Gives rounded,
    organic body shells (spider abdomen, frog body, snake bead) instead of a box/capsule."""
    return {"family": "revolve", "profile": [[round(r, 5), round(z, 5)] for r, z in profile]}


def _tapered(length: float, r0: float, r1: float) -> dict:
    """A frustum limb segment (thick proximal -> thin distal): real legs taper."""
    return {"family": "tapered", "length": round(length, 5), "r0": round(r0, 5), "r1": round(r1, 5)}


def _ellipsoid_profile(length: float, max_r: float, *, bulge: float = 0.5,
                       n: int = 7) -> list[tuple[float, float]]:
    """A teardrop/ellipsoid (r, z) profile from z=0..length, widest at z=bulge*length. Used for the
    spider abdomen (fat, rear-biased) and the frog body (streamlined, front-biased)."""
    pts = []
    for i in range(n + 1):
        z = length * i / n
        t = z / length
        # smooth bump peaking at `bulge`; r->small at both ends
        shape = math.sin(math.pi * min(1.0, (t / (2 * bulge)) if t < bulge else
                                       (0.5 + 0.5 * (t - bulge) / (1 - bulge))))
        pts.append((max(0.004, max_r * (0.18 + 0.82 * shape)), z))
    return pts


# ----------------------------------------------------------------------------------------------------
# SPIDER
# ----------------------------------------------------------------------------------------------------
def build_spider(n_legs: int = 8) -> tuple[RobotGene, dict]:
    """A recognizable spider: a low cephalothorax (front, carries the legs + a head/chelicerae stub) and a
    bulbous abdomen (rear), with `n_legs` long two-joint legs radiating outward, fanned front->back.

    Returns (gene, pose) where pose maps joint_name -> rest angle (rad) for the render.
    """
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}

    # --- body: cephalothorax (root, oriented so its +z points UP a touch) + abdomen behind it -----
    ceph_r, ceph_h = 0.075, 0.045        # cephalothorax: a low rounded disc
    segs.append(GeneSegment(
        name="cephalothorax", parent=None, shape="cylinder", length_m=ceph_h, radius_m=ceph_r,
        mass_kg=0.6, joint_type=None,
        geometry=_revolve([(0.30 * ceph_r, 0), (0.92 * ceph_r, 0.45 * ceph_h),
                           (0.80 * ceph_r, 0.85 * ceph_h), (0.45 * ceph_r, ceph_h)])))

    # abdomen: a fat teardrop mounted at the REAR (-x) of the cephalothorax, drooping slightly down.
    ab_len, ab_r = 0.14, 0.085
    segs.append(GeneSegment(
        name="abdomen", parent="cephalothorax", shape="capsule", length_m=ab_len, radius_m=ab_r,
        mass_kg=1.0, joint_type=None,
        mount_offset=(-0.9 * ceph_r, 0.0, 0.4 * ceph_h),
        mount_euler=(0.0, PI / 2 + 0.35, PI),     # local +z aims back (-x) and a bit down
        geometry=_revolve(_ellipsoid_profile(ab_len, ab_r, bulge=0.62))))

    # head / chelicerae stub at the FRONT (+x), small — sells the "front of the animal".
    head_len = 0.05
    segs.append(GeneSegment(
        name="head", parent="cephalothorax", shape="capsule", length_m=head_len, radius_m=0.030,
        mass_kg=0.08, joint_type=None,
        mount_offset=(0.85 * ceph_r, 0.0, 0.25 * ceph_h),
        mount_euler=(0.0, PI / 2 + 0.5, 0.0),     # aims forward + slightly down
        geometry=_revolve([(0.030, 0), (0.022, 0.5 * head_len), (0.010, head_len)])))

    # --- legs: radiate around the cephalothorax, fanned front->back, splayed OUT then bent DOWN -----
    # Each leg is aimed by AZIMUTH phi (direction around the body). The femur points OUT and slightly UP
    # to the "knee"; the tibia hinges DOWN (local +y axis = horizontal tangent) so the foot meets the floor.
    per_side = n_legs // 2
    femur, tibia = 0.13, 0.15            # long thin legs (spider legs are long; tibia >= femur)
    f_r, t_r = 0.014, 0.010
    femur_el = 0.18                      # femur reaches ~10deg ABOVE horizontal out to the knee
    # fan azimuths: front legs forward-out, rear legs backward-out (a real arachnid fan, not a row).
    fan = [(0.95 - 1.9 * i / max(1, per_side - 1)) for i in range(per_side)]  # +0.95..-0.95 rad off side
    for side, sy in (("l", 1.0), ("r", -1.0)):
        for i, f in enumerate(fan):
            phi = sy * (PI / 2) - f       # azimuth: side direction (+/-y) fanned forward/back by f
            mx = ceph_r * 0.9 * math.cos(phi)
            my = ceph_r * 0.9 * math.sin(phi)
            femur_name = f"leg_{side}{i}_femur"
            tibia_name = f"leg_{side}{i}_tibia"
            dir_out = (math.cos(phi) * math.cos(femur_el), math.sin(phi) * math.cos(femur_el),
                       math.sin(femur_el))
            # FEMUR: yaw joint about world-up (lets a gait swing the leg fore/aft); aimed out+up.
            segs.append(GeneSegment(
                name=femur_name, parent="cephalothorax", shape="capsule",
                length_m=femur, radius_m=f_r, mass_kg=0.05,
                joint_type="revolute", joint_axis=(0.0, 0.0, 1.0),
                joint_lower=-0.6, joint_upper=0.6,
                mount_offset=(mx, my, 0.45 * ceph_h),
                mount_euler=_aim_euler(dir_out),
                actuator_torque_nm=4.0, geometry=_tapered(femur, f_r * 1.5, f_r)))
            # TIBIA: knee hinge about the femur's local +y (the horizontal tangent) -> bends in the
            # vertical radial plane so the foot drops to the floor (the classic spider crook).
            segs.append(GeneSegment(
                name=tibia_name, parent=femur_name, shape="capsule",
                length_m=tibia, radius_m=t_r, mass_kg=0.035,
                joint_type="revolute", joint_axis=(0.0, 1.0, 0.0),
                joint_lower=-0.2, joint_upper=2.6,    # +ve bends the foot DOWN to the floor
                actuator_torque_nm=3.0,
                is_end_effector=(side == "l" and i == 0),
                geometry=_tapered(tibia, t_r * 1.3, t_r * 0.7)))
            pose[f"{femur_name}_joint"] = 0.0
            pose[f"{tibia_name}_joint"] = 1.6   # bend the knee DOWN -> foot on the floor (spider crook)

    gene = RobotGene(id="proc_spider", species="arachnid.proc", robot_class="legged",
                     segments=segs, base_mount="free", end_effector_type="none")
    return gene, pose


# ----------------------------------------------------------------------------------------------------
# FROG
# ----------------------------------------------------------------------------------------------------
def build_frog() -> tuple[RobotGene, dict]:
    """A frog: a wide flat streamlined body + broad head, short folded FORE legs (front, tucked) and long
    powerful HIND legs (rear, folded in the characteristic crouch with knees up and feet forward)."""
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}

    # BODY: a low, wide, squat dome with its local +z pointing UP (no root rotation, so every child's
    # mount_offset/_aim_euler is in the WORLD frame — x=forward, y=side, z=up — which is what makes the legs
    # land reliably). A sitting frog is taller-than-long; the revolve gives the rounded back.
    body_h, body_r = 0.085, 0.075
    segs.append(GeneSegment(
        name="body", parent=None, shape="capsule", length_m=body_h, radius_m=body_r,
        mass_kg=0.8, joint_type=None,
        geometry=_revolve([(0.55 * body_r, 0), (body_r, 0.30 * body_h), (0.95 * body_r, 0.65 * body_h),
                           (0.55 * body_r, body_h)])))

    # HEAD: a broad rounded head at the FRONT (+x), tilted slightly down — a frog's wide mouth-forward head.
    head_len = 0.07
    segs.append(GeneSegment(
        name="head", parent="body", shape="capsule", length_m=head_len, radius_m=0.06, mass_kg=0.2,
        joint_type=None, mount_offset=(0.75 * body_r, 0.0, -0.45 * body_h),
        mount_euler=_aim_euler((0.95, 0.0, 0.10)),     # +z aims forward, a touch up
        geometry=_revolve([(0.042, 0), (0.060, 0.4 * head_len), (0.052, 0.75 * head_len),
                           (0.024, head_len)])))
    # two bulging EYES on top of the head — the iconic froggy cue. Mount near the head's rear-top, poking up.
    for ey, name in ((1.0, "eye_l"), (-1.0, "eye_r")):
        segs.append(GeneSegment(
            name=name, parent="head", shape="capsule", length_m=0.022, radius_m=0.019, mass_kg=0.01,
            joint_type=None, mount_offset=(ey * 0.034, 0.030, head_len * 0.30),
            mount_euler=_aim_euler((0.30 * ey, 0.30, 0.90)),   # bump pokes up-&-out
            geometry=_revolve([(0.019, 0), (0.019, 0.012), (0.011, 0.022)])))

    # FORE legs: short straight little arms at the FRONT shoulders that PROP the chest up (a sitting frog).
    # Aim each down-&-forward-&-out; a slight elbow bend plants the hand.
    for side, sy in (("l", 1.0), ("r", -1.0)):
        up, lo = f"fore_{side}_upper", f"fore_{side}_lower"
        dir_fore = (0.45, sy * 0.35, -0.82)            # forward, out, mostly DOWN  (WORLD frame)
        segs.append(GeneSegment(
            name=up, parent="body", shape="capsule", length_m=0.05, radius_m=0.013, mass_kg=0.05,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.5, joint_upper=1.5,
            mount_offset=(0.5 * body_r, sy * 0.62 * body_r, -0.25 * body_h),
            mount_euler=_aim_euler(dir_fore), actuator_torque_nm=2.0,
            geometry=_tapered(0.05, 0.014, 0.010)))
        segs.append(GeneSegment(
            name=lo, parent=up, shape="capsule", length_m=0.045, radius_m=0.010, mass_kg=0.03,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-1.6, joint_upper=1.6,
            actuator_torque_nm=1.5, is_end_effector=(side == "l"),
            geometry=_tapered(0.045, 0.011, 0.008)))
        pose[f"{up}_joint"] = 0.0
        pose[f"{lo}_joint"] = 0.4        # forearm bends forward so the hand plants ahead

    # HIND legs: the frog's signature — LONG, powerful, deeply Z-FOLDED beside the body (knee up-and-out,
    # shank folded back-down, long foot flat). Aim the thigh OUT-&-UP; fold the knee + ankle so the foot
    # lands on the floor pointing backward — the classic crouched frog.
    for side, sy in (("l", 1.0), ("r", -1.0)):
        thigh, shank, foot = f"hind_{side}_thigh", f"hind_{side}_shank", f"hind_{side}_foot"
        # Thigh points UP-&-OUT so the KNEE rises ABOVE the back (the iconic cocked-spring frog silhouette);
        # the knee then folds the shank steeply down-&-forward and the long foot plants flat.
        dir_thigh = (-0.35, sy * 0.78, 0.52)
        segs.append(GeneSegment(
            name=thigh, parent="body", shape="capsule", length_m=0.085, radius_m=0.024, mass_kg=0.16,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-2.4, joint_upper=2.4,
            mount_offset=(-0.5 * body_r, sy * 0.7 * body_r, 0.05 * body_h),
            mount_euler=_aim_euler(dir_thigh), actuator_torque_nm=7.0,
            geometry=_tapered(0.085, 0.028, 0.017)))
        segs.append(GeneSegment(
            name=shank, parent=thigh, shape="capsule", length_m=0.085, radius_m=0.016, mass_kg=0.09,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-0.2, joint_upper=2.8,
            actuator_torque_nm=5.0, geometry=_tapered(0.085, 0.018, 0.012)))
        segs.append(GeneSegment(
            name=foot, parent=shank, shape="capsule", length_m=0.06, radius_m=0.011, mass_kg=0.03,
            joint_type="revolute", joint_axis=(0.0, 1.0, 0.0), joint_lower=-2.6, joint_upper=0.4,
            actuator_torque_nm=2.0,
            geometry=_tapered(0.06, 0.013, 0.009)))
        pose[f"{thigh}_joint"] = 0.0
        pose[f"{shank}_joint"] = 2.5     # knee folded sharply (the cocked-spring hind leg)
        pose[f"{foot}_joint"] = -2.0     # long foot flat on the ground, pointing back

    gene = RobotGene(id="proc_frog", species="amphibian.proc", robot_class="legged",
                     segments=segs, base_mount="free", end_effector_type="none")
    return gene, pose


# ----------------------------------------------------------------------------------------------------
# SNAKE
# ----------------------------------------------------------------------------------------------------
def build_snake(n_beads: int = 16) -> tuple[RobotGene, dict]:
    """A snake: a tapered chain of `n_beads` rounded body segments, alternating yaw joints so it rests in
    a serpentine S-curve. A slightly larger head bead at the front."""
    segs: list[GeneSegment] = []
    pose: dict[str, float] = {}

    head_len, head_r = 0.055, 0.034
    segs.append(GeneSegment(
        name="head", parent=None, shape="capsule", length_m=head_len, radius_m=head_r,
        mass_kg=0.15, joint_type=None,
        mount_euler=(0.0, PI / 2, 0.0),        # lie flat, head points +x
        geometry=_revolve([(0.022, 0), (0.034, 0.45 * head_len), (0.030, 0.8 * head_len),
                           (0.016, head_len)])))

    prev = "head"
    seg_len = 0.07
    for i in range(n_beads):
        t = i / max(1, n_beads - 1)
        r = (1.0 - 0.55 * t) * 0.030 + 0.006   # taper toward the tail
        name = f"body_{i}"
        segs.append(GeneSegment(
            name=name, parent=prev, shape="capsule", length_m=seg_len, radius_m=r,
            mass_kg=max(0.02, 0.12 * (1 - 0.5 * t)),
            # The whole chain is aimed along world +x (the head's +90deg pitch about y), so world-UP maps to
            # the LOCAL -x axis. Yawing about world-up (horizontal slither) therefore needs joint_axis local
            # (-1,0,0) — a plain local +z axis would bend it up/down and collapse to a straight rod (the bug).
            joint_type="revolute", joint_axis=(-1.0, 0.0, 0.0),
            joint_lower=-0.8, joint_upper=0.8, actuator_torque_nm=2.5,
            is_end_effector=(i == n_beads - 1),
            geometry=_revolve([(0.6 * r, 0), (r, 0.5 * seg_len), (0.85 * r, seg_len)])))
        # serpentine rest curve: a few big lateral bends so it reads as a slithering S, not a rod. Each
        # successive joint adds the same sign for a stretch then flips -> broad alternating arcs.
        pose[f"{name}_joint"] = 0.7 * math.sin(i * 0.55)
        prev = name

    gene = RobotGene(id="proc_snake", species="serpent.proc", robot_class="legged",
                     segments=segs, base_mount="free", end_effector_type="none")
    return gene, pose


# ----------------------------------------------------------------------------------------------------
# HEXAPOD (insect)
# ----------------------------------------------------------------------------------------------------
def build_hexapod() -> tuple[RobotGene, dict]:
    """A six-legged insect: a 3-segment body (head + thorax + abdomen) with 6 legs in the classic insect
    tripod arrangement, splayed wide."""
    gene, pose = build_spider(n_legs=6)
    gene.id = "proc_hexapod"
    gene.species = "insect.proc"
    return gene, pose
