"""STRATEGY A - PANELED EXO-SHELL.

Generate an ORIGINAL, credible humanoid from ONLY our own parametric geometry: every link is a
smooth, contoured SHELL authored as a build123d (OCCT) program -- lofted elliptical cross-sections,
tapered limb shrouds, a rounded visored helmet, plate hands/feet -- so the whole body reads as ONE
designed consumer-robot product, not a stack of primitives.

NO real meshes, NO part_catalog / kit-bash, NO compose-from-real. Each solid is built parametrically
here, exported to STL (our own geometry), and fed to the stock compiler as the link's VISUAL mesh over
its invisible primitive collider (physics untouched). This is exactly the "editable parametric program
executed by a real geometry kernel" the research calls for.

    VIRTUROID_NO_LOCAL_ENV=1 PYTHONPATH=src python scripts/overhaul2/paneled_exo_shell.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

OUT_DIR = ROOT / "build" / "overhaul2" / "paneled_exo_shell"
MESH_DIR = OUT_DIR / "mesh"

# ----------------------------------------------------------------------------------------------------
# Anthropometric parameter block (everything derives from total height H, 7.5-head canon).
# This is the "explicit proportions / constraints" the research demands -- one number drives the body.
# ----------------------------------------------------------------------------------------------------
H = 1.70                       # total standing height (m)
HEAD = H / 7.5                 # ~0.227 m -- the canonical head module
# Vertical budget (sum ~= H): head + neck-gap + torso + thigh + shin + foot-rise.
TORSO_LEN = 2.30 * HEAD        # chest+ab+pelvis as ONE shell
THIGH_LEN = 1.95 * HEAD
SHIN_LEN = 1.90 * HEAD
UPPER_ARM_LEN = 1.65 * HEAD    # longer so forearm+hand reach mid-thigh (was 1.45)
FOREARM_LEN = 1.45 * HEAD      # (was 1.30)
HAND_LEN = 0.78 * HEAD
FOOT_LEN = 0.30 * HEAD         # rise (height) of the foot block; sole extends in +x
HEAD_LEN = 0.94 * HEAD

# Cross-section half-widths (the SILHOUETTE). x = front/back depth, y = left/right.
SHOULDER_HY = 1.02 * HEAD      # broad shoulders (slightly wider so arms hang flush at the torso edge)
CHEST_HX, CHEST_HY = 0.62 * HEAD, 0.82 * HEAD
WAIST_HX, WAIST_HY = 0.54 * HEAD, 0.66 * HEAD   # tapered waist (less hourglass; robot, not figure)
PELVIS_HX, PELVIS_HY = 0.60 * HEAD, 0.84 * HEAD  # hips flare back out

MM = 1000.0


def _color_stl(role: str) -> str:
    """Two-tone product look: light shells, charcoal actuator rings/visor. Returned as the MuJoCo
    material name the compiler already defines (mat_body light, mat_joint charcoal, mat_accent orange)."""
    return {"ring": "mat_joint", "visor": "mat_joint", "accent": "mat_accent"}.get(role, "mat_body")


# ----------------------------------------------------------------------------------------------------
# Parametric SHELL builders (build123d, mm, floored to z=0 in the link's local +z frame).
# ----------------------------------------------------------------------------------------------------
def _loft_sections(sections):
    """Loft a smooth shell through stacked elliptical cross-sections [(z, half_x, half_y), ...] (meters)."""
    import build123d as bd
    secs = [(z * MM, max(2.0, hx * MM), max(2.0, hy * MM)) for z, hx, hy in sections]
    with bd.BuildPart() as p:
        for z, hx, hy in secs:
            with bd.BuildSketch(bd.Plane.XY.offset(z)):
                bd.Ellipse(hx, hy)
        bd.loft()
    return p.part


def torso_shell():
    """ONE coherent torso shell: pelvis (wide) -> waist (pinched) -> chest (broad) -> shoulder yoke.
    Front-back depth is real (not a flat slab); a subtle chest-plate ridge is added on the +x face."""
    import build123d as bd
    L = TORSO_LEN
    sections = [
        (0.00 * L, 0.84 * PELVIS_HX, 0.92 * PELVIS_HY),   # pelvis underside (wide enough to bridge both hips)
        (0.20 * L, PELVIS_HX,        PELVIS_HY),           # hips (widest)
        (0.42 * L, WAIST_HX,         WAIST_HY),            # waist (pinch)
        (0.66 * L, CHEST_HX,         CHEST_HY),            # chest (broad, deep)
        (0.88 * L, 0.66 * CHEST_HX,  SHOULDER_HY),         # shoulder yoke (wide in y, shallow in x)
        (1.00 * L, 0.42 * CHEST_HX,  0.40 * SHOULDER_HY),  # neck base (tapers in)
    ]
    part = _loft_sections(sections)
    # Chest-plate ridge: a soft raised panel on the front (+x) of the upper chest -> reads as a designed
    # sternum plate, the kind a consumer robot has. Built as a thin lofted bump unioned on.
    with bd.BuildPart() as ridge:
        for z, w in [(0.50 * L, 0.30), (0.68 * L, 0.40), (0.82 * L, 0.26)]:
            with bd.BuildSketch(bd.Plane.XY.offset(z * MM)):
                bd.Ellipse((CHEST_HX * 0.5) * MM, (CHEST_HY * w + 0.0) * MM)
        bd.loft()
        plate = ridge.part.moved(bd.Location(((CHEST_HX * 0.62) * MM, 0, 0)))
    try:
        part = part + plate
    except Exception:  # noqa: BLE001 - union can be finicky; bare shell is an acceptable fallback
        pass
    try:
        part = bd.fillet(part.edges().filter_by(bd.GeomType.LINE), 4.0)
    except Exception:  # noqa: BLE001
        pass
    return part


def helmet_head():
    """A rounded helmet: lofted dome (wide jaw -> narrow crown) with a recessed face VISOR band cut on
    the +x face and a charcoal visor insert dropped into it. Returns (shell, visor_insert)."""
    import build123d as bd
    L = HEAD_LEN
    R = 0.82 * HEAD
    sections = [
        (0.00 * L, 0.46 * R, 0.42 * R),    # neck stub
        (0.12 * L, 0.62 * R, 0.60 * R),    # jaw
        (0.42 * L, 0.74 * R, 0.78 * R),    # cheeks (widest)
        (0.74 * L, 0.66 * R, 0.70 * R),    # brow
        (1.00 * L, 0.40 * R, 0.44 * R),    # crown (rounded in)
    ]
    shell = _loft_sections(sections)
    # Carve a shallow visor recess on the front, then build a thin dark insert to sit in it.
    visor_h = (0.40 * L)
    visor_z = 0.52 * L
    with bd.BuildPart() as cutter:
        with bd.BuildSketch(bd.Plane.XY.offset((visor_z - visor_h / 2) * MM)):
            bd.Ellipse(0.5 * R * MM, 0.62 * R * MM)
        with bd.BuildSketch(bd.Plane.XY.offset((visor_z + visor_h / 2) * MM)):
            bd.Ellipse(0.5 * R * MM, 0.58 * R * MM)
        bd.loft()
        cut = cutter.part.moved(bd.Location((0.40 * R * MM, 0, 0)))
    try:
        shell = shell - cut
    except Exception:  # noqa: BLE001
        pass
    # Dark visor insert: a slimmer copy of the cutter, pushed slightly back so it nests flush.
    with bd.BuildPart() as ins:
        with bd.BuildSketch(bd.Plane.XY.offset((visor_z - visor_h / 2) * MM)):
            bd.Ellipse(0.42 * R * MM, 0.55 * R * MM)
        with bd.BuildSketch(bd.Plane.XY.offset((visor_z + visor_h / 2) * MM)):
            bd.Ellipse(0.42 * R * MM, 0.50 * R * MM)
        bd.loft()
        insert = ins.part.moved(bd.Location((0.40 * R * MM, 0, 0)))
    try:
        shell = bd.fillet(shell.edges().filter_by(bd.GeomType.CIRCLE), 2.0)
    except Exception:  # noqa: BLE001
        pass
    return shell, insert


def limb_shell(length_m, r_prox, r_dist, *, oval=1.25, ring_r=None):
    """A tapered limb SHROUD: lofted oval cross-section from a fatter proximal end (near the joint) to a
    slimmer distal end -- the clean shrouded look of a consumer robot's arm/leg. A charcoal actuator RING
    is built as a separate solid (returned alongside) seated at the proximal joint."""
    import build123d as bd
    L = length_m
    sections = [
        (0.00 * L, r_prox * 0.92, r_prox * 0.92 * oval),
        (0.12 * L, r_prox,        r_prox * oval),
        (0.55 * L, (r_prox + r_dist) * 0.5, (r_prox + r_dist) * 0.5 * oval),
        (0.90 * L, r_dist,        r_dist * oval),
        (1.00 * L, r_dist * 0.9,  r_dist * 0.9 * oval),
    ]
    shell = _loft_sections(sections)
    ring = None
    rr = ring_r if ring_r is not None else r_prox * 1.05   # snug, not bulging (was 1.18)
    with bd.BuildPart() as rp:                       # actuator can spanning the joint axis (along y)
        with bd.Locations(bd.Location((0, 0, 0.05 * L), (90, 0, 0))):
            bd.Cylinder(rr * MM, rr * 1.45 * MM)     # shorter so it doesn't jut past the limb (was 1.9)
    ring = rp.part
    return shell, ring


def hand_plate():
    """A rounded mitten/plate hand: a slim lofted plate, wider than deep, tapering toward the fingertips."""
    L = HAND_LEN
    w = 0.34 * HEAD
    sections = [
        (0.00 * L, 0.30 * w, 0.55 * w),
        (0.30 * L, 0.32 * w, 0.62 * w),
        (0.78 * L, 0.26 * w, 0.55 * w),
        (1.00 * L, 0.14 * w, 0.34 * w),
    ]
    return _loft_sections(sections)


def foot_plate():
    """A sleek foot: an ankle stub blended into a sole that extends forward (+x). Built as a loft along z
    for the ankle, then a forward sole bump unioned on."""
    import build123d as bd
    L = FOOT_LEN
    w = 0.40 * HEAD
    with bd.BuildPart() as p:
        for z, hx, hy in [(0.00 * L, 0.55 * w, 0.42 * w), (0.55 * L, 0.42 * w, 0.34 * w),
                          (1.00 * L, 0.30 * w, 0.28 * w)]:
            with bd.BuildSketch(bd.Plane.XY.offset(z * MM)):
                bd.Ellipse(hx * MM, hy * MM)
        bd.loft()
    ankle = p.part
    # Sole: a low, forward-extended rounded slab (toe points +x).
    sole_len, sole_w, sole_h = 1.55 * w, 0.40 * w, 0.22 * w
    with bd.BuildPart() as sp:
        with bd.Locations(bd.Location((0.30 * w * MM, 0, 0.12 * w * MM))):
            bd.Box(sole_len * MM, sole_w * 2 * MM, sole_h * 2 * MM)
        try:
            bd.fillet(sp.edges(), sole_h * 0.6 * MM)
        except Exception:  # noqa: BLE001
            pass
    try:
        foot = ankle + sp.part
    except Exception:  # noqa: BLE001
        foot = ankle
    return foot


# ----------------------------------------------------------------------------------------------------
# Bake every shell to STL (floored to z=0) and return {gene_segment_name: stl_path}.
# We also bake a few VISUAL-ONLY extra solids (visor insert, joint rings) onto their parent link's mesh
# by unioning them into that link's solid before export -- so they ride the link with no extra gene DOF.
# ----------------------------------------------------------------------------------------------------
def _floor_and_export(solid, fp):
    import build123d as bd
    try:
        minz = float(solid.bounding_box().min.Z)
    except Exception:  # noqa: BLE001
        minz = 0.0
    solid = solid.moved(bd.Location((0, 0, -minz)))
    bd.export_stl(solid, str(fp))
    return str(fp.resolve()).replace("\\", "/")


def _union(*solids):
    out = solids[0]
    for s in solids[1:]:
        if s is None:
            continue
        try:
            out = out + s
        except Exception:  # noqa: BLE001
            pass
    return out


def build_meshes():
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    meshes: dict[str, str] = {}
    mats: dict[str, str] = {}     # segment -> base material override (we keep light body, dark insert handled below)

    # Torso (single coherent shell).
    meshes["torso"] = _floor_and_export(torso_shell(), MESH_DIR / "torso.stl")

    # Head: shell + dark visor insert unioned (insert exported SEPARATELY so we can color it dark via a
    # second geom). Simpler/robust: union the insert into the head shell so it's one mesh; the recess+insert
    # still reads as a visor because of the geometry. We additionally export the insert alone for a dark geom.
    head_shell, visor = helmet_head()
    meshes["head"] = _floor_and_export(head_shell, MESH_DIR / "head.stl")
    # floor the visor by the SAME amount as the head so it nests correctly: recompute head minz.
    import build123d as bd
    head_minz = float(head_shell.bounding_box().min.Z)
    visor = visor.moved(bd.Location((0, 0, -head_minz)))
    bd.export_stl(visor, str(MESH_DIR / "head_visor.stl"))
    meshes["__head_visor"] = str((MESH_DIR / "head_visor.stl").resolve()).replace("\\", "/")

    # Limbs: shell + actuator ring unioned into each link mesh (ring is visual, rides the link).
    arm_up_s, arm_up_r = limb_shell(UPPER_ARM_LEN, 0.30 * HEAD, 0.22 * HEAD)
    arm_lo_s, arm_lo_r = limb_shell(FOREARM_LEN, 0.22 * HEAD, 0.17 * HEAD)
    thigh_s, thigh_r = limb_shell(THIGH_LEN, 0.40 * HEAD, 0.30 * HEAD)
    shin_s, shin_r = limb_shell(SHIN_LEN, 0.30 * HEAD, 0.20 * HEAD)

    for side in ("left", "right"):
        meshes[f"{side}_arm_upper"] = _floor_and_export(_union(arm_up_s, arm_up_r),
                                                        MESH_DIR / f"{side}_arm_upper.stl")
        meshes[f"{side}_arm_lower"] = _floor_and_export(_union(arm_lo_s, arm_lo_r),
                                                        MESH_DIR / f"{side}_arm_lower.stl")
        meshes[f"{side}_hand"] = _floor_and_export(hand_plate(), MESH_DIR / f"{side}_hand.stl")
        meshes[f"{side}_leg_thigh"] = _floor_and_export(_union(thigh_s, thigh_r),
                                                        MESH_DIR / f"{side}_leg_thigh.stl")
        meshes[f"{side}_leg_shin"] = _floor_and_export(_union(shin_s, shin_r),
                                                       MESH_DIR / f"{side}_leg_shin.stl")
        meshes[f"{side}_foot"] = _floor_and_export(foot_plate(), MESH_DIR / f"{side}_foot.stl")
    return meshes


# ----------------------------------------------------------------------------------------------------
# Build the GENE whose primitives match the shells (so physics is sane), then compile to MJCF with our
# meshes as the visible surface. Proportions per the parameter block above.
# ----------------------------------------------------------------------------------------------------
def build_gene():
    from virturoid.schemas.gene import GeneSegment, RobotGene

    pi = math.pi
    T_R = CHEST_HY                     # torso "radius" (box half-width y) ~ chest half-width
    segs: list[GeneSegment] = []

    # Root torso (free base). box primitive: half-x=radius, half-y=radius, half-z=length/2.
    segs.append(GeneSegment(name="torso", parent=None, shape="box",
                            length_m=TORSO_LEN, radius_m=T_R, mass_kg=9.0, joint_type=None))
    # Head (welded on top).
    segs.append(GeneSegment(name="head", parent="torso", shape="capsule",
                            length_m=HEAD_LEN, radius_m=0.6 * HEAD, mass_kg=1.0, joint_type="fixed",
                            mount_offset=(0.0, 0.0, 0.04)))

    def limb(prefix, parent, mo, links, last_is_ee=False):
        cur = parent
        for i, (nm, ln, rad, axis, lo, hi, jt, mass) in enumerate(links):
            name = f"{prefix}_{nm}"
            segs.append(GeneSegment(
                name=name, parent=cur, shape="capsule", length_m=ln, radius_m=rad,
                mass_kg=mass, joint_type=jt, joint_axis=axis, joint_lower=lo, joint_upper=hi,
                mount_offset=mo if i == 0 else (0.0, 0.0, 0.0),
                mount_euler=(pi, 0.0, 0.0) if i == 0 else (0.0, 0.0, 0.0),
                is_end_effector=(last_is_ee and i == len(links) - 1)))
            cur = name

    # CHILDREN attach at parent tip (0,0,parent.length) PLUS mount_offset. So mount_offset z is measured
    # DOWN from the torso TOP tip (z=TORSO_LEN): shoulders just below the tip (-0.18 head), hips at the
    # bottom (-TORSO_LEN). euler pi flips local +z to point DOWN so limbs hang.
    sh_y = SHOULDER_HY * 0.82   # tuck arms in toward the shoulder edge so they hang flush (was 0.92)
    arm_links = [
        ("upper", UPPER_ARM_LEN, 0.26 * HEAD, (0, 1, 0), -2.0, 2.0, "revolute", 1.0),
        ("lower", FOREARM_LEN, 0.20 * HEAD, (0, 1, 0), -2.4, 0.1, "revolute", 0.7),
        ("hand", HAND_LEN, 0.16 * HEAD, (0, 0, 1), None, None, "fixed", 0.3),
    ]
    leg_links = [
        ("thigh", THIGH_LEN, 0.34 * HEAD, (0, 1, 0), -1.8, 1.0, "revolute", 2.0),
        ("shin", SHIN_LEN, 0.26 * HEAD, (0, 1, 0), 0.0, 2.4, "revolute", 1.5),
        ("foot", FOOT_LEN, 0.18 * HEAD, (0, 1, 0), None, None, "fixed", 0.5),
    ]
    limb("left_arm", "torso", (0.0, sh_y, -0.18 * HEAD), arm_links)
    limb("right_arm", "torso", (0.0, -sh_y, -0.18 * HEAD), arm_links)
    hip_y = PELVIS_HY * 0.42   # legs closer together -> closes the crotch void (was 0.55)
    limb("left_leg", "torso", (0.0, hip_y, -TORSO_LEN), leg_links)
    limb("right_leg", "torso", (0.0, -hip_y, -TORSO_LEN), leg_links, last_is_ee=True)

    # Natural rest stance baked as a keyframe: arms abducted ~12 deg off the body so they don't clip the
    # torso, a soft knee bend so the body reads as a standing biped rather than a stiff star.
    rest = {
        "left_arm_upper_joint": 0.06, "right_arm_upper_joint": 0.06,   # arms nearly straight down, just off body
        "left_arm_lower_joint": -0.10, "right_arm_lower_joint": -0.10,
        "left_leg_thigh_joint": 0.04, "right_leg_thigh_joint": 0.04,
        "left_leg_shin_joint": 0.08, "right_leg_shin_joint": 0.08,
    }
    return RobotGene(id="exo_shell_humanoid", species="humanoid.exo_shell",
                     robot_class="humanoid", segments=segs, base_mount="free",
                     end_effector_type="none", metadata={"rest_pose": rest})


# Map gene segment names -> mesh keys produced by build_meshes (limb link names differ slightly).
_MESH_KEY = {
    "torso": "torso", "head": "head",
    "left_arm_upper": "left_arm_upper", "left_arm_lower": "left_arm_lower", "left_arm_hand": "left_hand",
    "right_arm_upper": "right_arm_upper", "right_arm_lower": "right_arm_lower", "right_arm_hand": "right_hand",
    "left_leg_thigh": "left_leg_thigh", "left_leg_shin": "left_leg_shin", "left_leg_foot": "left_foot",
    "right_leg_thigh": "right_leg_thigh", "right_leg_shin": "right_leg_shin", "right_leg_foot": "right_foot",
}


def render(gene, meshes):
    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.real_model_library import normalize_display

    seg_meshes = {seg: meshes[_MESH_KEY[seg]] for seg in _MESH_KEY if _MESH_KEY[seg] in meshes}
    spawn = standing_spawn_z(gene)
    mjcf = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=spawn, meshes=seg_meshes)

    # Inject the dark visor as an extra visual geom riding the head body, + offwidth.
    visor_path = meshes.get("__head_visor")
    if visor_path:
        mjcf = mjcf.replace(
            "  </asset>\n",
            f'    <mesh name="head_visor_vis" file="{visor_path}" scale="0.001 0.001 0.001"/>\n  </asset>\n', 1)
        # add a dark geom inside the head body (find the head body open tag)
        needle = '<body name="head"'
        idx = mjcf.find(needle)
        if idx != -1:
            close = mjcf.find(">", idx) + 1
            geom = ('\n          <geom name="head_visor_g" type="mesh" mesh="head_visor_vis" '
                    'material="mat_joint" mass="0" contype="0" conaffinity="0"/>')
            mjcf = mjcf[:close] + geom + mjcf[close:]

    if "offwidth" not in mjcf:
        mjcf = mjcf.replace("<visual>", '<visual><global offwidth="640" offheight="640"/>', 1)

    m = mujoco.MjModel.from_xml_string(mjcf)
    normalize_display(m)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    ctr = d.geom_xpos[1:].mean(0) if m.ngeom > 1 else np.array([0.0, 0.0, 0.9])
    r = mujoco.Renderer(m, 520, 520)
    cam = mujoco.MjvCamera()
    imgs = []
    for _name, az, el, dist in [("front", 90, -8, 2.6), ("side", 0, -8, 2.6),
                                ("3/4", 40, -16, 2.7), ("back34", 215, -14, 2.7)]:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
        r.update_scene(d, cam)
        imgs.append(np.asarray(r.render()))
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "humanoid.png"
    Image.fromarray(grid).save(str(out))
    return str(out), m.ngeom - 1


def main():
    print("[exo_shell] baking parametric shells...")
    meshes = build_meshes()
    print(f"[exo_shell] baked {len([k for k in meshes if not k.startswith('__')])} link shells")
    gene = build_gene()
    issues = gene.validate()
    print(f"[exo_shell] gene valid={not issues} DOF={len(gene.actuated_joints())} parts={len(gene.segments)}")
    if issues:
        print("  ISSUES:", issues)
    out, ngeom = render(gene, meshes)
    print(f"[exo_shell] rendered -> {out} (geoms drawn: {ngeom})")


if __name__ == "__main__":
    main()
