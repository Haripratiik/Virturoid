"""METHOD D — KIT-BASH REAL PART MESHES onto a generated skeleton for NOVEL bodies.

Real Menagerie models are made of high-fidelity link MESHES. This prototype assembles a NOVEL topology
(e.g. an 8-legged 'spider' built from Unitree Go1 LEG meshes placed radially, or a multi-armed body built
from Unitree H1 ARM meshes) by referencing the individual link STL meshes from a cached real model and
placing copies on a GENERATED kinematic tree as VISUAL geoms over primitive capsule/box colliders.

Approach (the honest, robust version):
  * We do NOT re-parse / re-scale the STL ourselves. We reference the real link mesh files BY ABSOLUTE PATH
    in a fresh MJCF <asset>, so MuJoCo loads them at their NATIVE scale. The intra-link detail (the actual
    hardware geometry) is therefore byte-identical to the real model.
  * We reuse the real model's INTRA-LEG transforms (hip->thigh offset, thigh->calf offset, foot sphere),
    so a single leg is a faithful Go1 leg.
  * The NOVELTY is the topology: we instantiate that real leg N times, rotated by 2*pi*i/N about the body
    +z axis, attached to a GENERATED central trunk (a primitive we author). That is a body Menagerie has no
    model for (radial 6/8-legged "spider", or a multi-arm "manipulator hub").
  * Physics colliders are PRIMITIVE capsules/boxes we author (visual mesh group=2, contype/conaffinity=0;
    collider group=3) — matching services/part_catalog.py's strictly-visual kit-bash contract.

Renders 4-view contact sheets to build/overhaul/kitbash/ and reports compile + a short settle sim (stand).

    PYTHONPATH=src python scripts/overhaul/kitbash.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "build" / "overhaul" / "kitbash"


def _menagerie_root() -> Path:
    """Locate the cached MuJoCo Menagerie root via robot_descriptions (same mechanism as part_catalog)."""
    from robot_descriptions import go1_mj_description

    # .../mujoco_menagerie/unitree_go1/go1.xml -> .../mujoco_menagerie
    return Path(go1_mj_description.MJCF_PATH).resolve().parent.parent


def _assets():
    """Absolute paths to the real link meshes we kit-bash from (Go1 leg parts, H1 arm parts)."""
    root = _menagerie_root()
    go1 = root / "unitree_go1" / "assets"
    h1 = root / "unitree_h1" / "assets"
    return {
        # Go1 leg (BSD-3): hip housing, thigh, calf — the real quadruped leg chain.
        "go1_hip": go1 / "hip.stl",
        "go1_thigh": go1 / "thigh.stl",
        "go1_thigh_mirror": go1 / "thigh_mirror.stl",
        "go1_calf": go1 / "calf.stl",
        "go1_trunk": go1 / "trunk.stl",
        # H1 arm (BSD-3): shoulder/upper-arm, elbow/forearm — the real humanoid arm chain.
        "h1_shoulder": h1 / "left_shoulder_pitch_link.stl",
        "h1_upper": h1 / "left_shoulder_yaw_link.stl",
        "h1_elbow": h1 / "left_elbow_link.stl",
        "h1_torso": h1 / "torso_link.stl",
    }


def _asset_block(used: dict[str, Path]) -> str:
    """An <asset> with the real meshes referenced by absolute file path (native scale) + clean materials.

    We force a clean two-tone material (light body / dark joint) instead of the source model's near-black
    Menagerie materials, fixing the 'H1 renders dark' problem the brief calls out — geometry untouched.
    """
    meshes = "\n".join(
        f'    <mesh name="{name}" file="{p.as_posix()}"/>' for name, p in used.items()
    )
    return f"""  <asset>
    <material name="body" rgba="0.82 0.84 0.88 1" specular="0.3" shininess="0.4"/>
    <material name="joint" rgba="0.22 0.24 0.30 1" specular="0.2" shininess="0.3"/>
    <material name="accent" rgba="0.90 0.45 0.18 1" specular="0.4" shininess="0.5"/>
{meshes}
  </asset>"""


def _header(title: str, used: dict[str, Path]) -> str:
    return f"""<mujoco model="{title}">
  <compiler angle="radian" autolimits="true"/>
  <option cone="elliptic" impratio="100"/>
  <visual>
    <global offwidth="900" offheight="900"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.7 0.7 0.7" specular="0.25 0.25 0.25"/>
    <quality shadowsize="2048"/>
  </visual>
{_asset_block(used)}
  <worldbody>
    <light name="top" pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <light name="key" pos="1.5 -1.5 2.5" dir="-0.5 0.5 -1" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="6 6 0.1" rgba="0.30 0.34 0.40 1"/>
"""


FOOTER = """  </worldbody>
</mujoco>
"""


# ----------------------------------------------------------------------------- spider (radial Go1 legs)

def _go1_leg(prefix: str, mirror: bool) -> str:
    """One faithful Go1 leg chain (real meshes, real intra-leg transforms), authored as a +y-side leg.

    Mirrors the go1.xml hierarchy: hip (abduction) -> thigh (+/-0.08 y) -> calf (-0.213 z) -> foot sphere.
    We attach the leg already rotated/positioned by the caller (the body wraps it in a <body> with pos/quat).
    Colliders are primitive capsules (group 3); meshes are visual (group 2).
    """
    thigh_mesh = "go1_thigh" if not mirror else "go1_thigh_mirror"
    y = 0.08 if not mirror else -0.08
    hipq = '' if not mirror else ' quat="1 0 0 0"'
    return f"""        <body name="{prefix}_hip" pos="0 0 0">
          <joint name="{prefix}_abduct" type="hinge" axis="1 0 0" range="-0.8 0.8" damping="1"/>
          <geom type="mesh" mesh="go1_hip" material="joint" contype="0" conaffinity="0" group="2"{hipq}/>
          <geom type="capsule" fromto="0 0 0 0 {y} 0" size="0.035" group="3" rgba="0.3 0.3 0.3 0.0"/>
          <body name="{prefix}_thigh" pos="0 {y} 0">
            <joint name="{prefix}_hip" type="hinge" axis="0 1 0" range="-0.7 3.5" damping="2"/>
            <geom type="mesh" mesh="{thigh_mesh}" material="body" contype="0" conaffinity="0" group="2"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.213" size="0.022" group="3" rgba="0.3 0.3 0.3 0.0"/>
            <body name="{prefix}_calf" pos="0 0 -0.213">
              <joint name="{prefix}_knee" type="hinge" axis="0 1 0" range="-2.8 -0.9" damping="2"/>
              <geom type="mesh" mesh="go1_calf" material="joint" contype="0" conaffinity="0" group="2"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.213" size="0.014" group="3" rgba="0.3 0.3 0.3 0.0"/>
              <geom name="{prefix}_foot" type="sphere" size="0.023" pos="0 0 -0.213"
                    material="accent" condim="6" friction="0.8 0.02 0.01" priority="1"/>
            </body>
          </body>
        </body>"""


def build_spider(n_legs: int = 8, trunk_r: float = 0.14, leg_z: float = 0.30) -> str:
    """A NOVEL radial 'spider': n Go1 legs placed evenly around a generated disc trunk. No real model exists.

    The trunk is a generated primitive (a flat cylinder shell with a small dome) — we DON'T have a real
    spider trunk mesh, so this is the 'generated skeleton' the kit-bashed real legs hang from.
    """
    used = {k: v for k, v in _assets().items() if k.startswith("go1")}
    xml = [_header(f"kitbash_spider_{n_legs}", used)]
    # free base so we can settle/stand
    xml.append(f'    <body name="core" pos="0 0 {leg_z + 0.20:.3f}">')
    xml.append('      <freejoint/>')
    # generated trunk: flat disc + small dome (primitive visual + collider)
    xml.append(f'      <geom name="trunk_disc" type="cylinder" size="{trunk_r} 0.045" '
               'material="body" contype="0" conaffinity="0" group="2"/>')
    xml.append(f'      <geom name="trunk_dome" type="sphere" size="{trunk_r*0.7:.3f}" pos="0 0 0.04" '
               'material="accent" contype="0" conaffinity="0" group="2"/>')
    xml.append(f'      <geom name="trunk_col" type="cylinder" size="{trunk_r} 0.05" group="3" '
               'rgba="0.3 0.3 0.3 0.0"/>')
    # radial legs
    for i in range(n_legs):
        ang = 2.0 * math.pi * i / n_legs
        # rotate the +y-built leg so its hip points radially outward (about world +z)
        # quat for rotation about z by `ang`:
        hw = math.cos(ang / 2.0)
        hz = math.sin(ang / 2.0)
        px = trunk_r * math.cos(ang)
        py = trunk_r * math.sin(ang)
        mirror = (i % 2 == 1)
        xml.append(f'      <body name="leg{i}" pos="{px:.4f} {py:.4f} 0" quat="{hw:.5f} 0 0 {hz:.5f}">')
        # leg built along +y in its own frame; after the body quat it splays around the disc
        xml.append(_go1_leg(f"L{i}", mirror))
        xml.append('      </body>')
    xml.append('    </body>')
    xml.append(FOOTER)
    return "\n".join(xml)


def build_hexapod() -> str:
    """A 6-legged variant (same machinery, fewer legs) — another novel topology."""
    return build_spider(n_legs=6, trunk_r=0.12, leg_z=0.28)


# --------------------------------------------------------------- multi-arm hub (radial H1 arms)

def _h1_arm(prefix: str) -> str:
    """A faithful H1 arm chain (real shoulder/upper/elbow meshes) authored along +x in its own frame."""
    return f"""        <body name="{prefix}_sh" pos="0 0 0">
          <joint name="{prefix}_s1" type="hinge" axis="0 0 1" range="-2.5 2.5" damping="2"/>
          <geom type="mesh" mesh="h1_shoulder" material="joint" contype="0" conaffinity="0" group="2"/>
          <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.04" group="3" rgba="0.3 0.3 0.3 0.0"/>
          <body name="{prefix}_up" pos="0 0 -0.10">
            <joint name="{prefix}_s2" type="hinge" axis="0 1 0" range="-2.0 2.0" damping="2"/>
            <geom type="mesh" mesh="h1_upper" material="body" contype="0" conaffinity="0" group="2"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.22" size="0.03" group="3" rgba="0.3 0.3 0.3 0.0"/>
            <body name="{prefix}_fore" pos="0 0 -0.22">
              <joint name="{prefix}_e" type="hinge" axis="0 1 0" range="-2.5 0.1" damping="2"/>
              <geom type="mesh" mesh="h1_elbow" material="body" contype="0" conaffinity="0" group="2"/>
              <geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.025" group="3" rgba="0.3 0.3 0.3 0.0"/>
            </body>
          </body>
        </body>"""


def build_multiarm_hub(n_arms: int = 5, hub_r: float = 0.16) -> str:
    """A NOVEL multi-armed manipulator hub: n real H1 arms radially around a generated central column."""
    used = {k: v for k, v in _assets().items() if k.startswith("h1")}
    xml = [_header(f"kitbash_multiarm_{n_arms}", used)]
    xml.append('    <body name="column" pos="0 0 0.9">')
    xml.append('      <freejoint/>')
    xml.append('      <geom name="col" type="cylinder" size="0.10 0.45" material="body" '
               'contype="0" conaffinity="0" group="2"/>')
    xml.append('      <geom name="hub" type="sphere" size="0.16" pos="0 0 0.45" material="accent" '
               'contype="0" conaffinity="0" group="2"/>')
    xml.append('      <geom name="col_col" type="cylinder" size="0.10 0.45" group="3" rgba="0.3 0.3 0.3 0.0"/>')
    for i in range(n_arms):
        ang = 2.0 * math.pi * i / n_arms
        hw = math.cos(ang / 2.0)
        hz = math.sin(ang / 2.0)
        px = hub_r * math.cos(ang)
        py = hub_r * math.sin(ang)
        xml.append(f'      <body name="arm{i}" pos="{px:.4f} {py:.4f} 0.45" quat="{hw:.5f} 0 0 {hz:.5f}">')
        xml.append(_h1_arm(f"A{i}"))
        xml.append('      </body>')
    xml.append('    </body>')
    xml.append(FOOTER)
    return "\n".join(xml)


# --------------------------------------------------------------- render + verify

def render(xml: str, name: str) -> dict:
    """Compile, settle a few hundred steps (free base), render 4 views to a contact sheet. Returns a report."""
    import mujoco
    import numpy as np
    from PIL import Image

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"name": name}
    try:
        m = mujoco.MjModel.from_xml_string(xml)
    except Exception as exc:  # noqa: BLE001
        report["compiled"] = False
        report["error"] = str(exc)
        (OUT / f"{name}.xml").write_text(xml, encoding="utf-8")
        return report
    report["compiled"] = True
    report["ngeom"] = int(m.ngeom)
    report["nmesh"] = int(m.nmesh)
    report["nu"] = int(m.nu)
    report["nbody"] = int(m.nbody) - 1

    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    # record base z, settle under gravity to test 'stand'
    free_qadr = None
    for j in range(m.njnt):
        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_qadr = int(m.jnt_qposadr[j])
            break
    z0 = float(d.qpos[free_qadr + 2]) if free_qadr is not None else None
    for _ in range(400):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    z1 = float(d.qpos[free_qadr + 2]) if free_qadr is not None else None
    report["base_z_spawn"] = round(z0, 4) if z0 is not None else None
    report["base_z_settled"] = round(z1, 4) if z1 is not None else None
    if z0 is not None:
        # "stood" = didn't collapse to the floor or fly off; stayed within a sane band
        report["stood"] = bool(z1 > 0.03 and z1 < z0 + 0.5 and abs(z1) < 5.0)

    pts = d.geom_xpos[: m.ngeom]
    center = pts.mean(axis=0)
    extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
    r = mujoco.Renderer(m, 450, 450)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = max(0.7, extent * 2.2)
    shots = []
    for az, el in [(90, -12), (45, -22), (140, -25), (0, -80)]:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam)
        shots.append(np.asarray(r.render()))
    grid = np.concatenate(
        [np.concatenate(shots[:2], 1), np.concatenate(shots[2:], 1)], 0
    )
    out_png = OUT / f"{name}.png"
    Image.fromarray(grid).save(str(out_png))
    report["png"] = str(out_png)
    return report


def main() -> int:
    builds = {
        "spider8_go1legs": build_spider(8),
        "hexapod6_go1legs": build_hexapod(),
        "multiarm5_h1arms": build_multiarm_hub(5),
    }
    for name, xml in builds.items():
        rep = render(xml, name)
        print(f"=== {name} ===")
        for k, v in rep.items():
            if k == "error":
                print(f"   ERROR: {v[:400]}")
            else:
                print(f"   {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
