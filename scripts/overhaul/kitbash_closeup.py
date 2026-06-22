"""Close-up render of the kit-bash spider to inspect the trunk<->leg seam fidelity honestly."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "overhaul"))

import numpy as np
import mujoco
from PIL import Image
from kitbash import build_spider, OUT


def main() -> int:
    xml = build_spider(8)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    for _ in range(400):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    pts = d.geom_xpos[: m.ngeom]
    center = pts.mean(axis=0)
    r = mujoco.Renderer(m, 700, 700)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = center
    cam.distance = 0.55  # tight closeup on trunk/leg junction
    shots = []
    for az, el in [(60, -8), (60, -35)]:
        cam.azimuth, cam.elevation = az, el
        r.update_scene(d, cam)
        shots.append(np.asarray(r.render()))
    grid = np.concatenate(shots, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "spider8_closeup.png"
    Image.fromarray(grid).save(str(out))
    print("closeup ->", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
