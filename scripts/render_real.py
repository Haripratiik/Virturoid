"""Render a REAL robot model (MuJoCo Menagerie via robot_descriptions) from 4 angles — to compare real,
production-grade robots against the procedural primitive bodies.

    PYTHONPATH=src python scripts/render_real.py --prompt "a quadruped" --out build/inspect/real_quad.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a quadruped")
    ap.add_argument("--out", default="build/inspect/real.png")
    ap.add_argument("--size", type=int, default=640)
    args = ap.parse_args(argv)

    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.real_model_library import load_real_model, normalize_display

    r = load_real_model(args.prompt)
    if not r["ok"]:
        print("no real model:", r["note"]); return 1
    m = mujoco.MjModel.from_xml_path(r["path"])
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), args.size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), args.size)
    normalize_display(m)        # lift near-black Menagerie materials + neutral lighting (display-only)
    d = mujoco.MjData(m)
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)          # the model's natural "home" pose
    else:
        mujoco.mj_resetData(m, d)
    mujoco.mj_forward(m, d)

    pts = d.geom_xpos[: m.ngeom]
    center = pts.mean(axis=0)
    extent = float((pts.max(axis=0) - pts.min(axis=0)).max())
    renderer = mujoco.Renderer(m, args.size, args.size)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = max(0.6, extent * 2.3)
    shots = []
    for az, el in [(90, -12), (45, -20), (140, -22), (0, -85)]:
        cam.azimuth, cam.elevation = az, el
        renderer.update_scene(d, cam)
        shots.append(renderer.render())
    top = np.concatenate(shots[:2], axis=1)
    bot = np.concatenate(shots[2:], axis=1)
    grid = np.concatenate([top, bot], axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(args.out)
    print(f"{r['label']}: bodies={r['bodies']} actuators={r['actuated']} meshes={r['meshes']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
