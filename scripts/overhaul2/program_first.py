"""STRATEGY D - PROGRAM-FIRST PER-PART humanoid generation.

The research's recommended pattern: treat each body part as a small parametric CAD PROGRAM. For every
anatomical part (torso/head/upperarm/forearm/hand/thigh/shin/foot) we AUTHOR an explicit profile that
captures that part's real form, executed by the build123d/OCCT kernel via cad_geometry.realize_shape:

  * torso   = a solid-of-revolution whose [r,z] profile flares at the shoulders, pinches at the waist,
              flares again at the hips (a real torso silhouette, not a box).
  * head    = revolve: domed cranium -> brow -> tapered jaw/chin -> neck stub.
  * limbs   = revolve "muscle spindles": a tapered oval that bulges (biceps/calf/thigh belly) then necks
              down at the joint, so an arm reads as an arm not a rod.
  * hand    = extrude a paddle/mitten profile (palm + tapered fingers + thumb notch).
  * foot    = extrude an L-shaped sole (heel block at the ankle, sole extended toward the toe).

These are ORIGINAL parametric programs - NO real meshes, NO part_catalog/kitbash, NO compose-from-real.
We assemble them with anthropometric proportions (head ~1/7.5 of height, shoulders ~torso width, arms
reach mid-thigh, legs ~half height) and render a 4-view contact sheet, then self-critique and repair.

    PYTHONPATH=src VIRTUROID_NO_LOCAL_ENV=1 python scripts/overhaul2/program_first.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

OUT_DIR = Path(__file__).resolve().parents[2] / "build" / "overhaul2" / "program_first"
CACHE_DIR = OUT_DIR / "_viewmesh"


# ----------------------------------------------------------------------------------------------------
# PER-PART PARAMETRIC PROGRAMS. Each returns a `geometry` dict consumed by cad_geometry.realize_shape.
# All dimensions are METERS in the LINK's local +z frame (the part is built along [0, length] +z; the
# mesh layer floors it to z=0 and drops it on the collision primitive that spans [0, length_m]).
# ----------------------------------------------------------------------------------------------------

def torso_program(length: float, half_w: float, half_d: float) -> dict:
    """A torso as a solid of revolution whose radius traces a real frontal silhouette: narrow neck base,
    broad shoulders/chest, pinched waist, flared pelvis. `half_w` sets the shoulder breadth. Built bottom
    (pelvis, z=0) -> top (shoulders, z=length). We bias the profile slightly so the mean radius reads as a
    chest, not a barrel. The revolve makes it round in plan; that is acceptable for a credible first pass
    (a human torso in silhouette is dominated by this exact chest->waist->hip curve)."""
    L = length
    # [radius, z] control points up the body. radii are fractions of half_w.
    prof = [
        [0.62 * half_w, 0.00 * L],   # pelvis floor
        [0.95 * half_w, 0.10 * L],   # hip flare (widest of lower body)
        [0.90 * half_w, 0.18 * L],
        [0.66 * half_w, 0.34 * L],   # waist pinch
        [0.64 * half_w, 0.42 * L],   # waist
        [0.82 * half_w, 0.58 * L],   # lower ribcage
        [1.00 * half_w, 0.74 * L],   # chest (broadest)
        [1.00 * half_w, 0.86 * L],   # shoulder shelf
        [0.74 * half_w, 0.95 * L],   # shoulder -> neck transition
        [0.34 * half_w, 1.00 * L],   # neck base
    ]
    return {"family": "revolve", "profile": prof}


def head_program(length: float, radius: float) -> dict:
    """A head: a short neck stub -> tapered jaw/chin -> brow -> domed cranium. Revolve profile bottom->top.
    Anthropometric: cranium is the widest band ~60-75% up; chin tapers in; a slim neck at the base."""
    L = length
    R = radius
    prof = [
        [0.34 * R, 0.00 * L],   # neck base
        [0.34 * R, 0.16 * L],   # neck
        [0.50 * R, 0.24 * L],   # jaw underside
        [0.86 * R, 0.34 * L],   # jaw / cheek
        [0.98 * R, 0.50 * L],   # cheekbone / widest face
        [1.00 * R, 0.66 * L],   # cranium widest
        [0.92 * R, 0.80 * L],   # upper skull
        [0.62 * R, 0.92 * L],   # crown taper
        [0.22 * R, 1.00 * L],   # top
    ]
    return {"family": "revolve", "profile": prof}


def limb_program(length: float, prox_r: float, dist_r: float, belly: float, belly_at: float = 0.42) -> dict:
    """A limb segment as a 'muscle spindle' solid of revolution: starts at the proximal joint radius,
    swells to a muscle belly, then necks down to the distal joint. `belly` is the peak radius; `belly_at`
    where (0..1) along the segment it peaks. Captures biceps/forearm/thigh/calf taper - reads as a limb,
    not a uniform rod or a stack of motor cans."""
    L = length
    prof = [
        [0.92 * prox_r, 0.00 * L],          # proximal joint collar
        [prox_r,        0.08 * L],
        [belly,         belly_at * L],      # muscle belly
        [0.78 * belly,  min(0.80, belly_at + 0.34) * L],
        [dist_r,        0.94 * L],          # neck down to distal joint
        [0.92 * dist_r, 1.00 * L],          # distal collar
    ]
    return {"family": "revolve", "profile": prof}


def hand_program(length: float, radius: float) -> dict:
    """A hand as an EXTRUDED paddle/mitten: a wrist neck, a palm that widens, and a rounded finger end with
    a thumb bump on one side. Extruded along the part's +z (so the plate's thickness is `height`); the 2D
    profile is the hand seen edge-on (x = across the palm, y = wrist->fingertip). Fingers are suggested as a
    rounded fan; a real static hand reads as a mitten, which this captures."""
    L = length
    R = radius
    # 2D profile in (x, y): wrist at y=0, fingertips at y=L. x is half-width across the palm.
    pw = 0.95 * R   # palm half width
    prof = [
        [-0.42 * R, 0.00 * L],     # wrist left
        [-pw,       0.30 * L],     # palm widens
        [-1.05 * R, 0.42 * L],     # thumb bump (left side)
        [-0.90 * R, 0.58 * L],
        [-0.78 * R, 0.86 * L],     # knuckles
        [-0.30 * R, 1.00 * L],     # fingertips (rounded)
        [0.30 * R,  1.00 * L],
        [0.78 * R,  0.86 * L],
        [0.92 * R,  0.58 * L],
        [0.92 * R,  0.30 * L],
        [0.42 * R,  0.00 * L],     # wrist right
    ]
    return {"family": "extrude", "profile": prof, "height": 0.42 * R, "fillet": 0.10 * R}


def foot_program(length: float, radius: float) -> dict:
    """A foot as an EXTRUDED FOOTPRINT plate. The leg chain is flipped (pi,0,0) at the hip so the foot's
    local +z points DOWN to the floor; realize_shape extrudes the 2-D profile in the local XY plane along
    +z. So the profile IS the top-down footprint (x = lateral half-width, y = heel->toe) and `height` is
    the (thin) sole thickness. NOTE the (pi,0,0) leg flip mirrors local +y about the body, so to put the
    toe FORWARD in the world we author the long axis toward LOCAL -y here; world forward = body +x is set
    by the foot footprint being longer than wide. We make it a real shoe outline: narrow heel, wide ball,
    rounded toe."""
    R = radius
    heel = -2.0 * R     # heel (local -y)
    toe = 2.6 * R       # toe (local +y), foot longer toward toe
    prof = [
        [-0.55 * R, heel],          # heel, slightly rounded
        [0.55 * R,  heel],
        [0.80 * R,  0.0],           # midfoot widens
        [0.95 * R,  1.2 * R],       # ball of foot (widest)
        [0.70 * R,  toe - 0.4 * R],
        [0.30 * R,  toe],           # toe tip
        [-0.30 * R, toe],
        [-0.70 * R, toe - 0.4 * R],
        [-0.95 * R, 1.2 * R],
        [-0.80 * R, 0.0],
    ]
    # height = sole thickness; floored to z=0 then the ankle/shin sits on top of it.
    return {"family": "extrude", "profile": prof, "height": length, "fillet": 0.10 * R}


# ----------------------------------------------------------------------------------------------------
# ASSEMBLY with anthropometric proportions.
# ----------------------------------------------------------------------------------------------------

def build_humanoid_spec() -> dict:
    """Assemble the per-part programs into a humanoid recipe with real proportions.

    Anthropometry (target total height ~1.7 m head-to-foot when standing):
      head ~ 1/7.5 of height; torso ~ 3 heads; upper arm + forearm reach to mid-thigh; legs ~ half height.
    """
    # Torso
    T_LEN = 0.52
    T_HALF_W = 0.16      # shoulder half-breadth -> shoulder width ~0.32 m (credible)
    T_HALF_D = 0.11

    # Head (length ~ height/7.5)
    H_LEN = 0.23
    H_R = 0.105

    # Leg: thigh + shin + foot. legs ~ half of ~1.7 height.
    THIGH_L, THIGH_PR, THIGH_BELLY, THIGH_DR = 0.40, 0.075, 0.085, 0.055
    SHIN_L,  SHIN_PR,  SHIN_BELLY,  SHIN_DR = 0.40, 0.055, 0.068, 0.035
    FOOT_L,  FOOT_R = 0.07, 0.050   # FOOT_L = sole thickness; FOOT_R scales footprint length/width

    # Arm: upper arm + forearm + hand. mid-thigh reach.
    UA_L, UA_PR, UA_BELLY, UA_DR = 0.31, 0.055, 0.062, 0.040
    FA_L, FA_PR, FA_BELLY, FA_DR = 0.27, 0.042, 0.050, 0.030
    HAND_L, HAND_R = 0.105, 0.052

    leg = [
        {"name": "thigh", "length": THIGH_L, "axis": (0, 1, 0), "torque": 30.0, "radius": THIGH_PR,
         "mass": 2.0, "lower": -2.2, "upper": 0.6,
         "geometry": limb_program(THIGH_L, THIGH_PR, THIGH_DR, THIGH_BELLY, 0.40)},
        {"name": "shin", "length": SHIN_L, "axis": (0, 1, 0), "torque": 24.0, "radius": SHIN_PR,
         "mass": 1.2, "lower": -0.05, "upper": 2.4,
         "geometry": limb_program(SHIN_L, SHIN_PR, SHIN_DR, SHIN_BELLY, 0.30)},
        {"name": "foot", "length": FOOT_L, "axis": (0, 1, 0), "torque": 10.0, "radius": FOOT_R,
         "mass": 0.5, "joint": "fixed", "shape": "box",
         "geometry": foot_program(FOOT_L, FOOT_R)},
    ]
    arm = [
        {"name": "uarm", "length": UA_L, "axis": (0, 1, 0), "torque": 16.0, "radius": UA_PR,
         "mass": 0.9, "lower": -2.6, "upper": 2.6,
         "geometry": limb_program(UA_L, UA_PR, UA_DR, UA_BELLY, 0.42)},
        {"name": "farm", "length": FA_L, "axis": (0, 1, 0), "torque": 10.0, "radius": FA_PR,
         "mass": 0.6, "lower": -0.05, "upper": 2.6,
         "geometry": limb_program(FA_L, FA_PR, FA_DR, FA_BELLY, 0.38)},
        {"name": "hand", "length": HAND_L, "axis": (0, 1, 0), "torque": 5.0, "radius": HAND_R,
         "mass": 0.35, "joint": "fixed", "shape": "box",
         "geometry": hand_program(HAND_L, HAND_R)},
    ]

    return {
        "robot_class": "humanoid", "base_mount": "free", "species": "humanoid.program_first",
        "base": {"name": "torso", "shape": "box", "length": T_LEN, "radius": T_HALF_W, "mass": 14.0,
                 "geometry": torso_program(T_LEN, T_HALF_W, T_HALF_D)},
        # Head welded on the torso TOP. Torso top tip is at z=T_LEN; head extends up from there.
        "links": [{"name": "head", "length": H_LEN, "radius": H_R, "joint": "fixed", "mass": 1.0,
                   "geometry": head_program(H_LEN, H_R)}],
        # Limbs mount on the torso. mount_offset is relative to the torso TOP tip (z=T_LEN per builder).
        # Legs hang from the pelvis (z=0 of torso => offset -T_LEN), spread to hip width.
        # Arms hang from just below the shoulder shelf (z ~ 0.88*T_LEN => offset ~ -0.12*T_LEN), at the
        # shoulder breadth. euler pi makes the +z limbs point DOWN.
        "limbs": [
            {"prefix": "left_leg", "parent": "torso",
             "mount_offset": (0.0, 0.52 * T_HALF_W, -T_LEN), "mount_euler": (math.pi, 0.0, 0.0),
             "links": [dict(s) for s in leg]},
            {"prefix": "right_leg", "parent": "torso",
             "mount_offset": (0.0, -0.52 * T_HALF_W, -T_LEN), "mount_euler": (math.pi, 0.0, 0.0),
             "links": [dict(s) for s in leg]},
            {"prefix": "left_arm", "parent": "torso",
             "mount_offset": (0.0, 1.12 * T_HALF_W, -0.12 * T_LEN), "mount_euler": (math.pi, 0.0, 0.0),
             "links": [dict(s) for s in arm]},
            {"prefix": "right_arm", "parent": "torso",
             "mount_offset": (0.0, -1.12 * T_HALF_W, -0.12 * T_LEN), "mount_euler": (math.pi, 0.0, 0.0),
             "links": [dict(s) for s in arm]},
        ],
        "end_effector": {"kind": "none"},
    }


# ----------------------------------------------------------------------------------------------------
# RENDER + VERIFY
# ----------------------------------------------------------------------------------------------------

def _inject(xml: str, w: int = 720, h: int = 720) -> str:
    if "offwidth" in xml:
        return xml
    g = f'<global offwidth="{w}" offheight="{h}"/>'
    if "<visual>" in xml:
        return xml.replace("<visual>", "<visual>" + g, 1)
    return xml.replace("<worldbody>", f"<visual>{g}</visual>\n  <worldbody>", 1)


def render(spec: dict, tag: str) -> str:
    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.morphology_composer import compose_from_spec
    from virturoid.services.gene_compiler import gene_to_meshed_mjcf, standing_spawn_z
    from virturoid.services.real_model_library import normalize_display

    gene = compose_from_spec(spec)
    issues = gene.validate()
    if issues:
        print("VALIDATION ISSUES:", issues)
    print(f"DOF={len(gene.actuated_joints())}  parts={len(gene.segments)}")

    spawn = standing_spawn_z(gene)
    mjcf = gene_to_meshed_mjcf(gene, cache_dir=str(CACHE_DIR), spawn_z=spawn, kitbash=False)
    m = mujoco.MjModel.from_xml_string(_inject(mjcf))
    normalize_display(m)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)

    # center on the whole body
    if m.ngeom > 1:
        ctr = d.geom_xpos[1:].mean(0)
        zmin = float((d.geom_xpos[1:, 2] - m.geom_rbound[1:]).min())
        zmax = float((d.geom_xpos[1:, 2] + m.geom_rbound[1:]).max())
        ctr[2] = 0.5 * (zmin + zmax)
    else:
        ctr = np.array([0.0, 0.0, 0.9])

    r = mujoco.Renderer(m, 600, 600)
    cam = mujoco.MjvCamera()
    imgs = []
    for _name, az, el, dist in [("front", 90, -8, 2.6), ("side", 0, -8, 2.6),
                                ("3/4", 40, -16, 2.7), ("back", 270, -8, 2.6)]:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
        r.update_scene(d, cam)
        imgs.append(np.asarray(r.render()))
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = str(OUT_DIR / f"humanoid{tag}.png")
    Image.fromarray(grid).save(out)
    print(f"rendered 4 views -> {out}  (geoms drawn: {m.ngeom - 1}, spawn_z={spawn})")
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    spec = build_humanoid_spec()
    render(spec, tag)
