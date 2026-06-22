"""STRATEGY C — ANTHROPOMETRIC PROPORTIONS FIRST.

Thesis: a humanoid reads as human when its PROPORTIONS and STANCE follow the classical
7.5-head canon, even if every part is a simple shaped solid. So we throw away surface
detail (no kit-bashed real meshes, no role anatomy) and instead spend ALL our effort on:

  1. correct head-canon proportions (head=1H; total~7.5H; shoulders ~2H; arm ~3.3H;
     leg ~4H; torso ~3H), derived parametrically from ONE number: the head height H.
  2. credible per-link silhouettes from our OWN parametric geometry only —
     ELLIPTICAL lofts (wider-than-deep trunk/limbs, which a human is) and tapered
     solids of revolution. Authored with build123d directly (OCCT), no pasted meshes.
  3. a real standing stance: shoulders at the top of the torso, hips at the bottom,
     arms hanging to mid-thigh, legs straight down, feet flat and forward-pointing.

All geometry is ORIGINAL parametric CAD (build123d on OCCT). No part_catalog, no kitbash,
no compose-from-real. We bake our own visual STLs (so we can use an elliptical trunk that
the shared realize_shape's axisymmetric `revolve` cannot express) and feed them to the REAL
compiler `compile_gene_to_mjcf(gene, meshes=...)` — identical physics, our own surface.

Run:  VIRTUROID_NO_LOCAL_ENV=1 PYTHONPATH=src python scripts/overhaul2/anthropometric_C.py
Out:  build/overhaul2/anthropometric_C/humanoid.png
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
os.environ.setdefault("VIRTUROID_NO_LOCAL_ENV", "1")

from virturoid.schemas.gene import GeneSegment, RobotGene  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "build" / "overhaul2" / "anthropometric_C"
MESH = OUT / "mesh"
OUT.mkdir(parents=True, exist_ok=True)
MESH.mkdir(parents=True, exist_ok=True)


# ==================================================================================
# ORIGINAL PARAMETRIC GEOMETRY (build123d / OCCT). Every solid is authored in the link's
# local [0, length] +z body frame with the lowest point at z=0, so it drops straight onto
# the compiler's primitive (which spans 0..length along local +z). Cross-sections are
# ELLIPTICAL (half-width in y, half-depth in x) so a trunk can be broad-but-shallow like a
# human, which a pure axisymmetric revolve cannot do.
# A "geometry" dict here is OUR spec, consumed by realize() below (NOT shared realize_shape).
# ==================================================================================
MM = 1000.0


def _ellipse_face(bd, half_y: float, half_x: float):
    """A single elliptical cross-section face (in the local x-y plane) at z=0, mm."""
    ry = max(0.5, half_y * MM)
    rx = max(0.5, half_x * MM)
    with bd.BuildSketch() as sk:
        bd.Ellipse(x_radius=rx, y_radius=ry)
    return sk.sketch


def _loft(bd, sections: list[tuple[float, float, float]]):
    """Loft a smooth solid through elliptical sections. Each section = (z, half_y, half_x)
    in meters. Produces a body whose width (y) and depth (x) vary independently along z —
    the move that gives broad shoulders + a narrow waist + a wider front than side."""
    faces = []
    for z, hy, hx in sections:
        f = _ellipse_face(bd, hy, hx)
        faces.append(f.moved(bd.Location((0, 0, float(z) * MM))))
    with bd.BuildPart() as p:
        bd.loft(faces)
    return p.part


def realize(spec: dict):
    """Realize OUR geometry spec into a build123d solid (mm), floored to z>=0.

    families:
      'loft'    {sections:[[z,half_y,half_x],...]}  -- elliptical loft (trunk/limbs/head)
      'revolve' {profile:[[r,z],...]}               -- axisymmetric solid of revolution
      'sole'    {length,width,height,heel,toe}      -- a forward-pointing foot (rounded box)
    """
    import build123d as bd

    fam = spec["family"]
    if fam == "loft":
        part = _loft(bd, spec["sections"])
    elif fam == "revolve":
        pts = [(max(0.4, float(r) * MM), float(z) * MM) for r, z in spec["profile"]]
        with bd.BuildSketch(bd.Plane.XZ) as sk:
            with bd.BuildLine():
                bd.Polyline(*pts, (0.4, pts[-1][1]), (0.4, pts[0][1]), close=True)
            bd.make_face()
        with bd.BuildPart() as p:
            bd.add(sk.sketch)
            bd.revolve(axis=bd.Axis.Z)
        part = p.part
    elif fam == "sole":
        L = spec["length"] * MM
        W = spec["width"] * MM
        Hh = spec["height"] * MM
        heel = spec.get("heel", 0.25) * MM
        with bd.BuildPart() as p:
            # a low rounded sole: heel block + tapered toe, lying with +z = up, +x = forward.
            with bd.BuildSketch(bd.Plane.XY) as sk:
                with bd.BuildLine():
                    bd.Polyline((-heel, -W), (L, -W * 0.78), (L, W * 0.78), (-heel, W), close=True)
                bd.make_face()
            bd.extrude(amount=Hh)
            try:
                bd.fillet(p.edges().filter_by(bd.Axis.Z), min(W * 0.5, Hh * 0.8))
            except Exception:
                pass
        part = p.part
    else:
        raise ValueError(f"unknown family {fam}")

    # floor to z=0 in the link frame
    try:
        minz = float(part.bounding_box().min.Z)
        part = part.moved(bd.Location((0, 0, -minz)))
    except Exception:
        pass
    return part


def bake_meshes(gene: RobotGene) -> dict:
    """Bake one ORIGINAL visual STL per segment from its `geometry` spec and return
    {segment_name: stl_path}. Skips segments with no geometry (none here)."""
    import build123d as bd

    meshes: dict[str, str] = {}
    for s in gene.segments:
        geom = getattr(s, "geometry", None)
        if not geom:
            continue
        solid = realize(geom)
        fp = MESH / f"{s.name}.stl"
        bd.export_stl(solid, str(fp))
        meshes[s.name] = str(fp.resolve()).replace("\\", "/")
    return meshes


# ==================================================================================
# GEOMETRY AUTHORING HELPERS (head-canon parametric)
# ==================================================================================

def loft(sections: list[tuple[float, float, float]]) -> dict:
    return {"family": "loft", "sections": [[float(z), float(hy), float(hx)] for z, hy, hx in sections]}


def limb(length: float, prox_y: float, prox_x: float, mid: float, dist_y: float, dist_x: float) -> dict:
    """A muscled, slightly oval limb segment lofted through 5 elliptical sections: rounded
    proximal cap -> proximal muscle bulk -> mid waist -> distal belly -> distal cap."""
    L = length
    return loft([
        (0.0,        prox_y * 0.55, prox_x * 0.55),
        (0.07 * L,   prox_y,        prox_x),
        (0.45 * L,   mid,           mid * 0.92),
        (0.85 * L,   dist_y * 1.10, dist_x * 1.10),
        (L,          dist_y * 0.55, dist_x * 0.55),
    ])


# ==================================================================================
# THE HUMANOID — built entirely from anthropometric head-canon proportions.
# ==================================================================================

def build_humanoid(H: float = 0.215) -> RobotGene:
    """Anthropometric humanoid. H = head height (m); total stature ~7.5-8 H ~ 1.8 m.

    7.5-head canon (heads from the sole, standing):
      head                = 1.0 H
      torso (pelvis->shoulder line) ~ 2.7 H
      shoulder width      ~ 2.0 H ;  hip width ~ 1.5 H
      arm (shoulder->wrist) ~ 3.0 H  (hangs to ~mid-thigh)
      thigh ~ 2.0 H ; shin ~ 1.85 H ; foot height ~ 0.18 H
    """
    seg: list[GeneSegment] = []
    DOWN = (math.pi, 0.0, 0.0)

    # --- TORSO (root). Local +z spans pelvis (z=0) -> shoulder line (z=torso_len).
    # Canon-tightened: a shorter trunk (2.5 H) so total stature lands ~7.7 H ~ 1.8 m.
    torso_len = 2.5 * H
    sh_y = 1.06 * H     # shoulder HALF-width (y)  -> ~2.1 H span (broad)
    sh_x = 0.50 * H     # shoulder HALF-depth (x)  (front is broader than side -> human)
    chest_y, chest_x = 0.88 * H, 0.50 * H
    waist_y, waist_x = 0.66 * H, 0.42 * H   # less aggressive pinch (was 0.58) -> one smooth trunk
    hip_y, hip_x = 0.80 * H, 0.46 * H
    torso_geo = loft([
        (0.00,             hip_y * 0.94, hip_x * 0.94),   # pelvis floor
        (0.12 * torso_len, hip_y,        hip_x),          # hips
        (0.30 * torso_len, waist_y,      waist_x),        # waist (gentle pinch)
        (0.50 * torso_len, waist_y * 1.08, waist_x * 1.05),  # solar plexus (smooth ramp up)
        (0.72 * torso_len, chest_y,      chest_x),        # chest
        (0.90 * torso_len, sh_y * 0.98,  sh_x),           # upper chest / clavicle
        (0.985 * torso_len, sh_y,        sh_x * 1.02),    # SHOULDER SHELF (broadest)
        (torso_len,        sh_y * 0.48,  sh_x * 0.72),    # neck base (narrow)
    ])
    seg.append(GeneSegment(name="torso", parent=None, shape="capsule",
                           length_m=torso_len, radius_m=waist_y, mass_kg=18.0,
                           joint_type=None, geometry=torso_geo))

    # --- NECK + HEAD (head = 1.0 H). Short neck stub, head as a tapered ovoid (revolve is
    # fine here: a head/neck is ~circular in cross-section).
    neck_len = 0.30 * H
    seg.append(GeneSegment(name="neck", parent="torso", shape="cylinder",
                           length_m=neck_len, radius_m=0.30 * H, mass_kg=0.6, joint_type=None,
                           geometry={"family": "revolve",
                                     "profile": [[0.33 * H, 0.0], [0.30 * H, neck_len]]}))
    head_len = 1.0 * H
    head_r = 0.40 * H
    seg.append(GeneSegment(name="head", parent="neck", shape="capsule",
                           length_m=head_len, radius_m=head_r, mass_kg=1.2, joint_type=None,
                           geometry={"family": "revolve", "profile": [
                               [0.27 * H, 0.0],               # jaw / chin
                               [0.37 * H, 0.20 * head_len],   # cheeks
                               [head_r, 0.48 * head_len],     # widest mid-skull
                               [0.38 * H, 0.78 * head_len],   # upper skull
                               [0.14 * H, head_len]]}))       # crown

    # --- ARMS (shoulder->wrist ~3.0 H), hanging straight down. Mount at the shoulder shelf,
    # tucked so the deltoid sits AGAINST the torso (no float gap). The upper-arm's proximal
    # section is fat (deltoid) so it visually bridges torso->arm.
    up_len, fore_len, hand_len = 1.48 * H, 1.22 * H, 0.62 * H   # reach to ~mid-thigh
    up_r, fore_r, hand_r = 0.30 * H, 0.23 * H, 0.20 * H
    # shoulder mount: AT the shoulder shelf (z just under the top tip), pulled IN to sh_y*0.80
    # so the fat deltoid cap OVERLAPS the torso shoulder shelf (no float gap). A big proximal
    # deltoid section (up_r*1.45) visually fuses the arm into the shoulder.
    shoulder_z = -0.03 * torso_len
    for side, sy in (("l", 1.0), ("r", -1.0)):
        seg.append(GeneSegment(name=f"{side}_uparm", parent="torso", shape="capsule",
                               length_m=up_len, radius_m=up_r, mass_kg=1.8,
                               joint_type="revolute", joint_axis=(0, 1, 0),
                               joint_lower=-3.0, joint_upper=3.0, actuator_torque_nm=18.0,
                               mount_offset=(0.0, sy * sh_y * 0.80, shoulder_z), mount_euler=DOWN,
                               geometry=limb(up_len, up_r * 1.32, up_r * 1.16, up_r * 0.80,
                                             fore_r * 1.02, fore_r)))
        seg.append(GeneSegment(name=f"{side}_forearm", parent=f"{side}_uparm", shape="capsule",
                               length_m=fore_len, radius_m=fore_r, mass_kg=1.0,
                               joint_type="revolute", joint_axis=(0, 1, 0),
                               joint_lower=-2.6, joint_upper=0.0, actuator_torque_nm=12.0,
                               geometry=limb(fore_len, fore_r * 1.08, fore_r, fore_r * 0.74,
                                             hand_r * 1.05, hand_r)))
        seg.append(GeneSegment(name=f"{side}_hand", parent=f"{side}_forearm", shape="capsule",
                               length_m=hand_len, radius_m=hand_r, mass_kg=0.4,
                               joint_type=None, is_end_effector=(side == "r"),
                               geometry=loft([
                                   (0.0,              hand_r * 0.7, hand_r * 0.5),
                                   (0.30 * hand_len,  hand_r,       hand_r * 0.55),  # palm (flat: wide y, thin x)
                                   (0.70 * hand_len,  hand_r * 0.9, hand_r * 0.45),
                                   (hand_len,         hand_r * 0.5, hand_r * 0.30)])))  # fingers taper

    # --- LEGS (thigh ~1.9 H, shin ~1.75 H). Hips at the torso bottom, ±hip half-width.
    thigh_len, shin_len = 1.9 * H, 1.75 * H
    thigh_r, shin_r, ankle_r = 0.40 * H, 0.30 * H, 0.21 * H
    for side, sy in (("l", 1.0), ("r", -1.0)):
        seg.append(GeneSegment(name=f"{side}_thigh", parent="torso", shape="capsule",
                               length_m=thigh_len, radius_m=thigh_r, mass_kg=4.5,
                               joint_type="revolute", joint_axis=(0, 1, 0),
                               joint_lower=-2.2, joint_upper=2.2, actuator_torque_nm=42.0,
                               mount_offset=(0.0, sy * hip_y * 0.74, -torso_len), mount_euler=DOWN,
                               geometry=limb(thigh_len, thigh_r * 1.15, thigh_r * 1.05, thigh_r * 0.85,
                                             shin_r * 1.05, shin_r)))
        seg.append(GeneSegment(name=f"{side}_shin", parent=f"{side}_thigh", shape="capsule",
                               length_m=shin_len, radius_m=shin_r, mass_kg=2.6,
                               joint_type="revolute", joint_axis=(0, 1, 0),
                               joint_lower=0.0, joint_upper=2.4, actuator_torque_nm=30.0,
                               geometry=limb(shin_len, shin_r * 1.10, shin_r, shin_r * 0.66,
                                             ankle_r * 1.05, ankle_r * 0.9)))
        # foot: a forward-pointing sole. mount_euler rotates local +z to world +x (forward).
        foot_len, foot_w, foot_h = 1.0 * H, 0.34 * H, 0.16 * H
        seg.append(GeneSegment(name=f"{side}_foot", parent=f"{side}_shin", shape="box",
                               length_m=foot_h, radius_m=foot_w, mass_kg=0.6, joint_type=None,
                               mount_euler=(0.0, -math.pi / 2, 0.0),
                               geometry={"family": "sole", "length": foot_len, "width": foot_w,
                                         "height": foot_h, "heel": 0.30 * H}))

    return RobotGene(id="anthropometric_C", species="humanoid.anthropometric_C",
                     robot_class="humanoid", segments=seg, base_mount="free",
                     end_effector_type="none", metadata={})


# ==================================================================================
# RENDER — 4 views, studio lighting, standing rest stance (no splay).
# ==================================================================================

def _inject_off(xml: str, w: int, h: int) -> str:
    g = f'<global offwidth="{w}" offheight="{h}"/>'
    if "<visual>" in xml:
        return xml.replace("<visual>", "<visual>" + g, 1)
    return xml.replace("<worldbody>", f"<visual>{g}</visual>\n  <worldbody>", 1)


def render(gene: RobotGene, out_png: Path, label: str = "") -> str:
    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.real_model_library import normalize_display

    meshes = bake_meshes(gene)                       # OUR original STLs
    sz = standing_spawn_z(gene)
    mjcf = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=sz, meshes=meshes)
    mjcf = _inject_off(mjcf, 520, 780)
    m = mujoco.MjModel.from_xml_string(mjcf)
    normalize_display(m)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)                           # standing rest pose (all joints 0)

    zmin = float((d.geom_xpos[1:, 2] - m.geom_rbound[1:]).min())
    zmax = float((d.geom_xpos[1:, 2] + m.geom_rbound[1:]).max())
    ctr = d.geom_xpos[1:].mean(0)
    look_z = 0.5 * (zmin + zmax)
    height = zmax - zmin
    dist = height * 1.5

    r = mujoco.Renderer(m, 780, 520)
    cam = mujoco.MjvCamera()
    imgs = []
    for az, el in [(90, -6), (0, -6), (40, -14), (135, -12)]:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [float(ctr[0]), float(ctr[1]), look_z]
        r.update_scene(d, cam)
        imgs.append(np.asarray(r.render()))
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(out_png)
    print(f"[{label}] stature={height:.3f} m  geoms={m.ngeom - 1}  -> {out_png}")
    return str(out_png)


if __name__ == "__main__":
    g = build_humanoid()
    print("structure:")
    for s in g.segments:
        print(f"  {s.name:12} parent={str(s.parent):8} L={s.length_m:.3f} R={s.radius_m:.3f} "
              f"joint={s.joint_type} fam={(s.geometry or {}).get('family')}")
    print("validate:", g.validate() or "OK")
    render(g, OUT / "humanoid.png", label="final")
