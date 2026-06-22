"""Render the Method-C improved-procedural bodies (spider/frog/snake/hexapod) to contact-sheet PNGs and
verify each compiles + stands in MuJoCo.

Renders the MESHED compile path (build123d role/revolve/tapered solids over invisible collision
primitives), applies each body's baked rest pose, normalizes lighting (Menagerie-style brighten), and
also reports whether the body SETTLES standing under gravity (a stand check).

    PYTHONPATH=src python scripts/overhaul/render_proc_anatomy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "src"))
sys.path.insert(0, str(_HERE))            # so proc_anatomy.py imports as a sibling module

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from virturoid.services.gene_compiler import (  # noqa: E402
    compile_gene_to_mjcf, gene_to_meshed_mjcf, standing_spawn_z)
from proc_anatomy import (  # noqa: E402
    build_frog, build_hexapod, build_snake, build_spider)

OUT = Path(__file__).resolve().parent.parent.parent / "build" / "overhaul" / "proc_anatomy"


def _inject_off(xml: str, w: int = 560, h: int = 560) -> str:
    g = f'<global offwidth="{w}" offheight="{h}"/>'
    if "offwidth" in xml:
        return xml
    if "<visual>" in xml:
        return xml.replace("<visual>", "<visual>" + g, 1)
    return xml.replace("<worldbody>", f"<visual>{g}</visual>\n  <worldbody>", 1)


def _apply_pose(m, d, pose: dict[str, float]) -> None:
    for jname, ang in pose.items():
        try:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        except Exception:  # noqa: BLE001
            jid = -1
        if jid < 0:
            continue
        adr = int(m.jnt_qposadr[jid])
        lo, hi = float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])
        if m.jnt_limited[jid]:
            ang = max(lo + 0.02, min(hi - 0.02, ang))
        d.qpos[adr] = ang


def _spawn_z_for_pose(gene, pose: dict[str, float]) -> float:
    """Standing spawn height that accounts for the rest pose (the default standing_spawn_z measures the
    zero pose; a posed spider reaches differently)."""
    try:
        m = mujoco.MjModel.from_xml_string(compile_gene_to_mjcf(gene, include_floor=False))
        d = mujoco.MjData(m)
        # set the free-joint base high, apply pose, measure lowest geom, then place so it clears the floor.
        if m.njnt and int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
            d.qpos[2] = 1.0
        _apply_pose(m, d, pose)
        mujoco.mj_forward(m, d)
        bottoms = d.geom_xpos[1:, 2] - m.geom_rbound[1:]
        return float(1.0 - bottoms.min() + 0.01)
    except Exception:  # noqa: BLE001
        return standing_spawn_z(gene)


def render(gene, pose: dict[str, float], label: str) -> tuple[str, dict]:
    OUT.mkdir(parents=True, exist_ok=True)
    sz = _spawn_z_for_pose(gene, pose)
    xml = gene_to_meshed_mjcf(gene, cache_dir=str(OUT / "_mesh"), include_floor=True, spawn_z=sz,
                              kitbash=False, synth=False)
    m = mujoco.MjModel.from_xml_string(_inject_off(xml))
    # lighting normalization (the known dark-render issue): brighten headlight like render_real.py
    m.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    m.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
    m.vis.headlight.specular[:] = [0.2, 0.2, 0.2]
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    _apply_pose(m, d, pose)
    mujoco.mj_forward(m, d)

    geoms = d.geom_xpos[1:]
    ctr = geoms.mean(0)
    extent = float((geoms.max(0) - geoms.min(0)).max())
    r = mujoco.Renderer(m, 560, 560)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
    cam.distance = max(0.5, extent * 1.9)
    imgs = []
    for az, el in [(90, -18), (35, -22), (135, -28), (0, -78)]:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam)
        imgs.append(np.asarray(r.render()))
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    out = str(OUT / f"{label}.png")
    Image.fromarray(grid).save(out)

    # stand check: drop it under gravity and see if the base stays roughly upright + above the floor
    stand = _stand_check(gene, pose, sz)
    print(f"{label}: parts={len(gene.segments)} DOF={len(gene.actuated_joints())} "
          f"meshed_geoms={m.ngeom - 1} spawn_z={sz:.3f} stand={stand} -> {out}")
    return out, stand


def _stand_check(gene, pose: dict[str, float], sz: float, steps: int = 400) -> dict:
    """Compile the PRIMITIVE (physics) model, drop it, and report base height drift + final upright-ness."""
    try:
        xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=sz)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        _apply_pose(m, d, pose)
        mujoco.mj_forward(m, d)
        z0 = float(d.qpos[2]) if int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE) else 0.0
        for _ in range(steps):
            mujoco.mj_step(m, d)
        z1 = float(d.qpos[2]) if int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE) else 0.0
        # upright: body z-axis dot world-up (quat w,x,y,z at qpos[3:7])
        upright = None
        if int(m.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE):
            w, x, y, zq = (float(v) for v in d.qpos[3:7])
            upright = round(1 - 2 * (x * x + y * y), 3)   # cos(tilt) of body +z vs world +z
        finite = bool(np.all(np.isfinite(d.qpos)))
        # A snake/limbless body has no "upright" axis — it correctly lies FLAT on the floor; judge it by
        # remaining above the floor + finite (stable), not by body-axis tilt.
        limbless = not any("femur" in s.name or "thigh" in s.name for s in gene.segments)
        ok_up = limbless or (upright is not None and upright > 0.4)
        return {"z0": round(z0, 3), "z1": round(z1, 3), "drop": round(z0 - z1, 3),
                "upright": upright, "finite": finite,
                "settled": bool(finite and z1 > 0.003 and ok_up)}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:80]}


def main() -> int:
    results = []
    for fn, label in [(build_spider, "spider"), (build_frog, "frog"),
                      (build_snake, "snake"), (build_hexapod, "hexapod")]:
        gene, pose = fn()
        out, stand = render(gene, pose, label)
        results.append((label, out, stand))
    print("\nSUMMARY")
    for label, out, stand in results:
        print(f"  {label:9} {out}  {stand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
