"""Reusable robot-inspection harness — so morphology/fidelity can be SEEN and verified, not claimed.

Composes a robot from a prompt (or the offline template), prints its full part/joint structure, and
renders it from four angles into one contact-sheet PNG. Use after any morphology change to confirm what
the software actually produces.

    PYTHONPATH=src python scripts/inspect_robot.py --prompt "a humanoid robot" [--offline] [--out build/inspect]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _inject(xml: str, w: int = 640, h: int = 640) -> str:
    if "offwidth" in xml:
        return xml
    g = f'<global offwidth="{w}" offheight="{h}"/>'
    if "<visual>" in xml:
        return xml.replace("<visual>", "<visual>" + g, 1)
    return xml.replace("<worldbody>", f"<visual>{g}</visual>\n  <worldbody>", 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a humanoid robot")
    ap.add_argument("--out", default="build/inspect")
    ap.add_argument("--offline", action="store_true", help="force the no-LLM template path")
    ap.add_argument("--meshed", action="store_true", help="render the high-fidelity visual-mesh layer")
    args = ap.parse_args(argv)
    if args.offline:
        os.environ["VIRTUROID_NO_LOCAL_ENV"] = "1"

    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, gene_to_meshed_mjcf
    from virturoid.services.morphology_composer import compose_robot

    g = compose_robot(args.prompt)
    print(f"prompt: {args.prompt!r}")
    print(f"robot_class={g.robot_class}  DOF={len(g.actuated_joints())}  "
          f"design_source={getattr(g, 'design_source', '?')}  parts={len(g.segments)}")
    print("structure (segment | parent | joint | shape | length):")
    for s in g.segments:
        print(f"  {s.name:18} {str(getattr(s, 'parent', None)):14} {str(s.joint_type):9} "
              f"{str(s.shape):9} {getattr(s, 'length_m', '?')}")

    mjcf = gene_to_meshed_mjcf(g) if args.meshed else compile_gene_to_mjcf(g, include_floor=True)
    m = mujoco.MjModel.from_xml_string(_inject(mjcf))
    d = mujoco.MjData(m); mujoco.mj_resetData(m, d)
    if m.nkey > 0:                                # a baked rest stance (novel archetypes) -> use it
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        for j in range(m.njnt):                   # else a gentle articulated pose so chains don't render as a pole
            if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                ang = 0.45 * (1 if j % 2 else -1)
                if m.jnt_limited[j]:
                    ang = max(float(m.jnt_range[j][0]) + 0.05, min(float(m.jnt_range[j][1]) - 0.05, ang))
                d.qpos[int(m.jnt_qposadr[j])] = ang
    mujoco.mj_forward(m, d)
    ctr = d.geom_xpos[1:].mean(0) if m.ngeom > 1 else np.array([0.0, 0.0, 0.3])
    r = mujoco.Renderer(m, 460, 460); cam = mujoco.MjvCamera()
    imgs = []
    for _name, az, el, dist in [("front", 90, -6, 1.7), ("side", 0, -6, 1.7),
                                ("3/4", 40, -18, 1.8), ("top", 40, -65, 2.0)]:
        cam.azimuth, cam.elevation, cam.distance = az, el, dist
        cam.lookat[:] = [float(ctr[0]), float(ctr[1]), float(ctr[2])]
        r.update_scene(d, cam); imgs.append(np.asarray(r.render()))
    grid = np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:], 1)], 0)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    out = str(Path(args.out) / ("robot_" + (g.species or g.robot_class or "x").replace("/", "_") + ".png"))
    Image.fromarray(grid).save(out)
    print(f"rendered 4 views -> {out}  (geoms drawn: {m.ngeom - 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
