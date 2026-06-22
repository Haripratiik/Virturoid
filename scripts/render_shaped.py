"""Render a procedural robot with its STAGE-3 shaped CAD links (beam + motor hub) posed at the body's
kinematics — vs the bare primitive capsules. Visual proof that procedural bodies now have real geometry.

    PYTHONPATH=src python scripts/render_shaped.py --prompt "a quadruped walking robot"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a quadruped walking robot")
    ap.add_argument("--out", default="build/inspect/shaped.png")
    ap.add_argument("--cad-dir", default="build/cad_render")
    ap.add_argument("--size", type=int, default=560)
    args = ap.parse_args(argv)

    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.cad_geometry import export_gene_cad
    from virturoid.services.gene_compiler import compile_gene_to_mjcf
    from virturoid.services.morphology_composer import compose_robot

    g = compose_robot(args.prompt)
    man = export_gene_cad(g, args.cad_dir)                        # real STEP + STL per link
    m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(g, include_floor=True))
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    stldir = str((Path(args.cad_dir) / "stl").resolve()).replace("\\", "/")

    assets, geoms = [], []
    for gi in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi)
        if not name or name == "floor" or not name.endswith("_geom"):
            continue   # skip the visual-only detail geoms (_motor/_collar); only structural links have STLs
        seg = name[:-5]
        pos = d.geom_xpos[gi]
        q = np.zeros(4); mujoco.mju_mat2Quat(q, d.geom_xmat[gi])
        assets.append(f'<mesh name="{seg}" file="{stldir}/{seg}.stl" scale="0.001 0.001 0.001"/>')
        geoms.append(f'<geom type="mesh" mesh="{seg}" pos="{pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}" '
                     f'quat="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f} {q[3]:.5f}" rgba="0.80 0.83 0.86 1"/>')

    xml = f"""<mujoco>
  <visual><global offwidth="{args.size}" offheight="{args.size}"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/></visual>
  <asset>
    {''.join(assets)}
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.25 0.30" rgb2="0.25 0.30 0.35" width="240" height="240"/>
    <material name="grid" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <geom name="floor" type="plane" size="4 4 0.1" material="grid"/>
    {''.join(geoms)}
  </worldbody>
</mujoco>"""
    rm = mujoco.MjModel.from_xml_string(xml)
    rd = mujoco.MjData(rm); mujoco.mj_forward(rm, rd)
    renderer = mujoco.Renderer(rm, args.size, args.size)
    pts = rd.geom_xpos[: rm.ngeom]
    center = pts.mean(axis=0)
    extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
    cam = mujoco.MjvCamera(); cam.lookat[:] = center; cam.distance = max(0.8, extent * 2.2)
    shots = []
    for az, el in [(90, -12), (45, -20), (140, -22), (0, -80)]:
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(rd, cam); shots.append(renderer.render())
    grid = np.concatenate([np.concatenate(shots[:2], axis=1), np.concatenate(shots[2:], axis=1)], axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.out)
    print(f"shaped CAD render: {man['part_count']} parts, {man['total_mass_kg']} kg -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
