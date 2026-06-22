"""STRATEGY B — Structural tube-frame humanoid with prominent harmonic-drive actuators.

A self-contained experiment (does NOT touch shared source). It generates an ORIGINAL, credible humanoid
from PURE parametric geometry (build123d / OCCT) — no real meshes, no part_catalog/kitbash, no compose
from real. The aesthetic is an honest "research robot": slender tapered structural TUBE limbs, with a
PROMINENT cylindrical ACTUATOR HUB (harmonic-drive look) straddling each shoulder / elbow / hip / knee, a
boxy-but-contoured electronics TORSO, and a compact sensor HEAD.

The mechanical honesty is the point: a visible actuator at every powered joint, sized in a believable
ratio to the limb it drives (not a can-tower, not invisible).

Pipeline (mirrors the product's): author one build123d solid per link (in the link's [0,length] +z body
frame, floored to z=0) -> export STL -> feed {segment: stl} to compile_gene_to_mjcf -> render 4 views.

    VIRTUROID_NO_LOCAL_ENV=1 PYTHONPATH=src python scripts/overhaul2/tubeframe.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import build123d as bd  # noqa: E402

OUT = ROOT / "build" / "overhaul2" / "tubeframe"
MESH = OUT / "mesh"
OUT.mkdir(parents=True, exist_ok=True)
MESH.mkdir(parents=True, exist_ok=True)

MM = 1000.0  # gene meters -> CAD millimeters


# ----------------------------------------------------------------------------------------------------
# Parametric geometry primitives (all build in the link's [0, L] +z frame, lowest point near z=0).
# Every helper returns a build123d Part in MILLIMETERS.
# ----------------------------------------------------------------------------------------------------
def _tube(L, r_top, r_bot, wall=0.0):
    """A slender tapered structural tube along +z, base at z=0. r in meters, taper r_bot(z=0)->r_top(z=L)."""
    return bd.Cone(bottom_radius=r_bot * MM, top_radius=r_top * MM, height=L * MM).moved(
        bd.Location((0, 0, L * MM / 2)))


def _hub(L, r, z_center, axis="x", thickness=None, chamfer=True):
    """A prominent harmonic-drive actuator hub: a short WIDE drum (thickness along its axis GREATER than its
    radius reads as a motor; the reverse reads as a bead) whose axis is the joint axis (default x = the limb
    pitch axis), centered at z_center*L along the limb. A thin output-flange disc on each face sells the
    harmonic-drive look. r,L,thickness in meters."""
    t = (thickness if thickness is not None else r * 1.7) * MM
    drum = bd.Cylinder(r * MM, t)
    # output flanges (slightly larger discs on each end face) -> the unmistakable drive-motor read
    drum = drum + bd.Cylinder(r * 1.12 * MM, t * 0.16).moved(bd.Location((0, 0, t / 2 - t * 0.08)))
    drum = drum + bd.Cylinder(r * 1.12 * MM, t * 0.16).moved(bd.Location((0, 0, -t / 2 + t * 0.08)))
    if axis == "x":
        drum = drum.rotate(bd.Axis.Y, 90)
    elif axis == "y":
        drum = drum.rotate(bd.Axis.X, 90)
    return drum.moved(bd.Location((0, 0, z_center * L * MM)))


def _flange(L, r, z_frac, t):
    """A thin coupling flange (a disc) perpendicular to the limb at z_frac of the length."""
    return bd.Cylinder(r * MM, t * MM).moved(bd.Location((0, 0, z_frac * L * MM)))


def _union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def limb_segment(L, r_shaft, *, proximal_hub_r, distal_hub_r=None, hub_axis="x"):
    """A structural-tube limb segment with a PROMINENT actuator hub at its PROXIMAL (z=0) joint.

    The shaft is a slender tapered tube; the proximal hub is the harmonic-drive can that drives THIS link's
    joint (straddling z=0, on the parent connection). An optional small distal coupling flange hints at the
    next joint. The actuator:shaft ratio is tuned so the hub reads as a believable motor, not a can-tower:
    hub radius ~2x the shaft radius, hub thickness ~ shaft radius * 3."""
    # A genuinely SLENDER tapered tube (real research-robot limb shrouds are thin); the hub provides the bulk.
    parts = [_tube(L, r_top=r_shaft * 0.72, r_bot=r_shaft * 0.88)]
    # Proximal actuator hub straddling z=0. WIDE drum (thickness > radius) so it reads as a motor, not a bead.
    parts.append(_hub(L, r=proximal_hub_r, z_center=0.0, axis=hub_axis,
                      thickness=proximal_hub_r * 2.2))
    # Small distal coupling collar (mechanical hint of the next joint).
    parts.append(_flange(L, r=r_shaft * 1.05, z_frac=0.95, t=r_shaft * 0.5))
    return _union(parts)


def torso_solid(L, half_w):
    """A boxy-but-contoured electronics torso: broad chest tapering to a waist, with shoulder yoke hubs and
    a pelvis block. Built in [0,L] +z; x=depth(front+), y=width. half_w = body half-width (y)."""
    w = half_w * MM
    Lm = L * MM
    depth = w * 0.62  # front-back half-depth (slimmer than wide -> reads like a chest, not a cube)
    parts = []
    # Upper chest: BROAD (full shoulder width so the shoulder drums sit flush, no gap).
    parts.append(bd.Box(2.0 * depth, 2.15 * w, 0.34 * Lm).moved(bd.Location((0, 0, 0.80 * Lm))))
    # Mid/waist: clearly PINCHED in y -> a real chest->waist taper, not a column.
    parts.append(bd.Box(1.5 * depth, 1.25 * w, 0.44 * Lm).moved(bd.Location((0, 0, 0.44 * Lm))))
    # Pelvis: wider again, where the legs bolt on.
    parts.append(bd.Box(1.7 * depth, 1.8 * w, 0.24 * Lm).moved(bd.Location((0, 0, 0.10 * Lm))))
    body = _union(parts)
    try:
        body = bd.fillet(body.edges(), 0.18 * depth)   # light fillet — keep chest/waist/pelvis distinct
    except Exception:  # noqa: BLE001
        pass
    extras = []
    # Chest electronics plate (front).
    extras.append(bd.Box(0.5 * depth, 1.2 * w, 0.34 * Lm).moved(bd.Location((depth * 0.95, 0, 0.74 * Lm))))
    # Shoulder yoke actuator drums (where the arms mount): big hubs about the y(=pitch) axis, set OUT to the
    # full shoulder width so the arm tube mounts flush off the drum face (kills the floating-arm gap).
    for sy in (-1, 1):
        drum = bd.Cylinder(0.33 * w, 0.62 * w).rotate(bd.Axis.X, 90)
        extras.append(drum.moved(bd.Location((0, sy * 1.18 * w, 0.86 * Lm))))
    # Hip actuator drums (front of pelvis, y axis) — visible leg motors.
    for sy in (-1, 1):
        drum = bd.Cylinder(0.26 * w, 0.42 * w).rotate(bd.Axis.X, 90)
        extras.append(drum.moved(bd.Location((0, sy * 0.62 * w, 0.06 * Lm))))
    # Neck mount collar (top).
    extras.append(bd.Cylinder(0.32 * w, 0.10 * Lm).moved(bd.Location((0, 0, 0.99 * Lm))))
    return _union([body] + extras)


def head_solid(L, r):
    """A compact sensor head: a rounded box cranium with a recessed front sensor visor band and a neck stub."""
    Lm = L * MM
    R = r * MM
    cranium = bd.Box(1.55 * R, 1.7 * R, 0.78 * Lm).moved(bd.Location((0, 0, 0.55 * Lm)))
    try:
        cranium = bd.fillet(cranium.edges(), 0.40 * R)
    except Exception:  # noqa: BLE001
        pass
    parts = [cranium]
    # Sensor visor band (front, dark recess look — geometry, slightly proud).
    parts.append(bd.Box(0.34 * R, 1.45 * R, 0.30 * Lm).moved(bd.Location((0.82 * R, 0, 0.62 * Lm))))
    # Neck stub (bottom, mates to torso collar).
    parts.append(bd.Cylinder(0.50 * R, 0.30 * Lm).moved(bd.Location((0, 0, 0.10 * Lm))))
    return _union(parts)


def foot_solid(L, r):
    """An ankle block + a flat sole extended toward the toe (+x), heel slightly behind."""
    Lm = L * MM
    R = r * MM
    ankle = bd.Box(0.9 * R, 0.95 * R, 0.55 * Lm).moved(bd.Location((0, 0, 0.62 * Lm)))
    sole = bd.Box(2.7 * R, 1.25 * R, 0.22 * Lm).moved(bd.Location((0.55 * R, 0, 0.11 * Lm)))
    body = _union([ankle, sole])
    try:
        body = bd.fillet(body.edges(), 0.10 * R)
    except Exception:  # noqa: BLE001
        pass
    # Proximal ankle actuator hub.
    hub = bd.Cylinder(0.62 * R, 0.9 * R).rotate(bd.Axis.Y, 90).moved(bd.Location((0, 0, 0.62 * Lm)))
    return _union([body, hub])


def hand_solid(L, r):
    """A simple gripper/end-effector clamp block (welded, static): a wrist hub + a two-prong claw."""
    Lm = L * MM
    R = r * MM
    wrist = bd.Cylinder(0.7 * R, 0.5 * Lm).moved(bd.Location((0, 0, 0.28 * Lm)))
    parts = [wrist]
    for sy in (-1, 1):
        prong = bd.Box(0.6 * R, 0.45 * R, 0.55 * Lm).moved(bd.Location((0, sy * 0.55 * R, 0.78 * Lm)))
        parts.append(prong)
    return _union(parts)


def export(part, name):
    """Floor the solid to z=0 (the compiler expects [0,length] frame), export STL, return path string."""
    try:
        minz = float(part.bounding_box().min.Z)
    except Exception:  # noqa: BLE001
        minz = 0.0
    part = part.moved(bd.Location((0, 0, -minz)))
    fp = MESH / f"{name}.stl"
    bd.export_stl(part, str(fp))
    return str(fp.resolve()).replace("\\", "/")


# ----------------------------------------------------------------------------------------------------
# The gene: a credibly-proportioned ~1.55 m humanoid. Proportions (canon-ish):
#   total height ~1.55 m. head ~ 1/7.5 of height. legs ~ half height. arms reach mid-thigh.
#   shoulder width ~ comparable to torso height; torso slimmer in depth than width.
# ----------------------------------------------------------------------------------------------------
def build_gene():
    from virturoid.schemas.gene import GeneSegment, RobotGene

    # --- proportion table (meters) ----------------------------------------------------------------
    T_LEN, T_HW = 0.50, 0.135       # torso length(z), half-width(y)  -> shoulders ~0.27 wide
    HEAD_L, HEAD_R = 0.20, 0.075
    THIGH_L, THIGH_R = 0.38, 0.052
    SHIN_L, SHIN_R = 0.38, 0.044
    FOOT_L, FOOT_R = 0.07, 0.05
    UARM_L, UARM_R = 0.30, 0.042
    FARM_L, FARM_R = 0.27, 0.036
    HAND_L, HAND_R = 0.10, 0.04
    # actuator hubs: ~1.7x the slender shaft radius -> clearly visible motors that still read as drives
    # straddling a thin tube (not a can-tower). Hips a touch bigger (they carry the most torque).
    HIP_HUB, KNEE_HUB = THIGH_R * 1.85, SHIN_R * 1.7
    SHO_HUB, ELB_HUB = UARM_R * 1.75, FARM_R * 1.7

    segs: list = []

    def add(**kw):
        segs.append(GeneSegment(**kw))

    # Root torso (free-floating base, welded root). Collision primitive ~ box matching torso envelope.
    add(name="torso", parent=None, shape="box", length_m=T_LEN, radius_m=T_HW, mass_kg=14.0,
        joint_type=None)

    # Head — welded on top, no separate neck segment (head's own neck stub bridges).
    add(name="head", parent="torso", shape="capsule", length_m=HEAD_L, radius_m=HEAD_R, mass_kg=1.2,
        joint_type="fixed", mount_offset=(0.0, 0.0, 0.0))

    # Legs: hip (thigh) + knee (shin) actuated about y(pitch); foot welded. mount_euler pi -> hang DOWN.
    # Mounted at the bottom of the torso (z=0 of torso => mount_offset z = -T_LEN since children attach at
    # parent's distal tip z=T_LEN). y=±0.6*T_HW keeps the hip on the pelvis surface.
    for side, sy in (("left", 1), ("right", -1)):
        add(name=f"{side}_thigh", parent="torso", shape="capsule", length_m=THIGH_L, radius_m=THIGH_R,
            mass_kg=3.0, joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-2.0, joint_upper=2.0,
            mount_offset=(0.0, sy * 0.60 * T_HW, -T_LEN), mount_euler=(math.pi, 0.0, 0.0))
        add(name=f"{side}_shin", parent=f"{side}_thigh", shape="capsule", length_m=SHIN_L,
            radius_m=SHIN_R, mass_kg=2.0, joint_type="revolute", joint_axis=(0, 1, 0),
            joint_lower=-2.4, joint_upper=0.05)
        add(name=f"{side}_foot", parent=f"{side}_shin", shape="box", length_m=FOOT_L, radius_m=FOOT_R,
            mass_kg=0.7, joint_type="fixed", mount_euler=(-math.pi / 2, 0.0, 0.0))

    # Arms: shoulder (upper arm) + elbow (forearm) about y; hand welded. Mount at the shoulders (top of
    # torso). y=±0.9*T_HW puts the arm root at the shoulder yoke. euler pi -> hang DOWN along the body.
    for side, sy in (("left", 1), ("right", -1)):
        add(name=f"{side}_uarm", parent="torso", shape="capsule", length_m=UARM_L, radius_m=UARM_R,
            mass_kg=1.4, joint_type="revolute", joint_axis=(0, 1, 0), joint_lower=-2.6, joint_upper=2.6,
            mount_offset=(0.0, sy * 1.02 * T_HW, -0.07), mount_euler=(math.pi, 0.0, 0.0))
        add(name=f"{side}_farm", parent=f"{side}_uarm", shape="capsule", length_m=FARM_L,
            radius_m=FARM_R, mass_kg=0.9, joint_type="revolute", joint_axis=(0, 1, 0),
            joint_lower=-2.6, joint_upper=0.05)
        is_ee = (side == "right")
        add(name=f"{side}_hand", parent=f"{side}_farm", shape="box", length_m=HAND_L, radius_m=HAND_R,
            mass_kg=0.4, joint_type="fixed", is_end_effector=is_ee)

    # --- author the parametric geometry per segment, export STL, collect mesh map ------------------
    meshes: dict[str, str] = {}
    meshes["torso"] = export(torso_solid(T_LEN, T_HW), "torso")
    meshes["head"] = export(head_solid(HEAD_L, HEAD_R), "head")
    for side in ("left", "right"):
        meshes[f"{side}_thigh"] = export(
            limb_segment(THIGH_L, THIGH_R, proximal_hub_r=HIP_HUB), f"{side}_thigh")
        meshes[f"{side}_shin"] = export(
            limb_segment(SHIN_L, SHIN_R, proximal_hub_r=KNEE_HUB), f"{side}_shin")
        meshes[f"{side}_foot"] = export(foot_solid(FOOT_L, FOOT_R), f"{side}_foot")
        meshes[f"{side}_uarm"] = export(
            limb_segment(UARM_L, UARM_R, proximal_hub_r=SHO_HUB), f"{side}_uarm")
        meshes[f"{side}_farm"] = export(
            limb_segment(FARM_L, FARM_R, proximal_hub_r=ELB_HUB), f"{side}_farm")
        meshes[f"{side}_hand"] = export(hand_solid(HAND_L, HAND_R), f"{side}_hand")

    gene = RobotGene(id="overhaul2_tubeframe", species="humanoid.tubeframe",
                     robot_class="humanoid", segments=segs, base_mount="free",
                     end_effector_type="none")
    # A natural standing rest pose: slight knee/elbow bend so it reads as a posed biped, not a star.
    gene.metadata["rest_pose"] = {
        "left_thigh_joint": -0.05, "right_thigh_joint": -0.05,  # near-neutral hips -> stands upright, no lean
        "left_shin_joint": 0.12, "right_shin_joint": 0.12,      # gentle knee bend -> reads as posed legs
        "left_uarm_joint": 0.20, "right_uarm_joint": 0.20,      # arms slightly forward
        "left_farm_joint": -0.45, "right_farm_joint": -0.45,    # clear elbow bend -> legible elbow joint
    }
    issues = gene.validate()
    if issues:
        raise SystemExit(f"INVALID GENE: {issues}")
    return gene, meshes


def _two_tone(xml, gene):
    """Mechanical-honesty two-tone: structural limb members render in a dark steel charcoal, while the torso/
    head/feet/hands (housings) stay light. Colors stay above the display-floor (0.18) so normalize_display
    doesn't wash them back to one grey. Done as a render-only MJCF rewrite (no shared-source edit)."""
    # add the two materials into the asset block
    mats = ('    <material name="mat_frame" rgba="0.24 0.26 0.32 1" specular="0.2" shininess="0.2"/>\n'
            '    <material name="mat_drive" rgba="0.62 0.64 0.68 1" specular="0.3" shininess="0.3"/>\n')
    xml = xml.replace("  </asset>", mats + "  </asset>", 1)
    frame_segs = {s.name for s in gene.segments
                  if any(k in s.name for k in ("thigh", "shin", "uarm", "farm"))}
    for name in frame_segs:
        xml = xml.replace(f'name="{name}_vis" type="mesh" mesh="{name}_vis" material="mat_body"',
                          f'name="{name}_vis" type="mesh" mesh="{name}_vis" material="mat_frame"')
    return xml


def _inject_off(xml, w=520, h=680):
    g = f'<global offwidth="{w}" offheight="{h}"/>'
    if "<visual>" in xml:
        return xml.replace("<visual>", "<visual>" + g, 1)
    return xml.replace("<worldbody>", f"<visual>{g}</visual>\n  <worldbody>", 1)


def render(gene, meshes):
    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.real_model_library import normalize_display

    spawn_z = standing_spawn_z(gene)
    mjcf = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn_z, meshes=meshes)
    mjcf = _two_tone(mjcf, gene)
    m = mujoco.MjModel.from_xml_string(_inject_off(mjcf))
    normalize_display(m)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ctr = d.geom_xpos[1:].mean(0) if m.ngeom > 1 else np.array([0.0, 0.0, 0.8])
    # frame the whole standing figure: lookat at mid-torso height
    look = [float(ctr[0]), float(ctr[1]), float(spawn_z) * 0.5 + 0.15]
    r = mujoco.Renderer(m, 680, 520)
    cam = mujoco.MjvCamera()
    views = [("front", 90, -4, 2.7), ("side", 0, -4, 2.7),
             ("3/4", 38, -14, 2.9), ("top", 40, -55, 3.1)]
    imgs = []
    for _n, az, el, dist in views:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = look
        r.update_scene(d, cam)
        imgs.append(np.asarray(r.render()))
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    out = OUT / "humanoid.png"
    Image.fromarray(grid).save(str(out))
    print(f"DOF={len(gene.actuated_joints())} parts={len(gene.segments)} geoms={m.ngeom - 1} "
          f"spawn_z={spawn_z:.3f}")
    print(f"rendered -> {out}")
    return str(out)


if __name__ == "__main__":
    g, mz = build_gene()
    render(g, mz)
