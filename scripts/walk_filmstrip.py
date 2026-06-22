"""Render a legged robot WALKING (scripted trot) as a horizontal filmstrip — visual proof of locomotion,
not just a static pose. Reuses the real gait via run_locomotion_episode's on_step hook (no duplication).

    PYTHONPATH=src python scripts/walk_filmstrip.py --prompt "a quadruped walking robot" [--frames 6]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="a quadruped walking robot")
    ap.add_argument("--out", default="build/inspect/walk.png")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--size", type=int, default=460)
    ap.add_argument("--legacy-spawn", action="store_true",
                    help="spawn at the legacy 0.1 m height the scripted gait is tuned for (vs standing spawn)")
    args = ap.parse_args(argv)

    import mujoco
    import numpy as np
    from PIL import Image

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    from virturoid.services.locomotion_controller import run_locomotion_episode
    from virturoid.services.morphology_composer import compose_robot

    g = compose_robot(args.prompt)
    spawn = None if args.legacy_spawn else standing_spawn_z(g)
    xml = compile_gene_to_mjcf(g, include_floor=True, spawn_z=spawn)
    if "offwidth" not in xml:                       # offscreen framebuffer big enough for the renderer
        glob = f'<visual><global offwidth="{args.size}" offheight="{args.size}"/></visual>\n  <worldbody>'
        xml = xml.replace("<worldbody>", glob, 1)
    model = mujoco.MjModel.from_xml_string(xml)
    base_body = next((int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == 0), 0)

    renderer = mujoco.Renderer(model, args.size, args.size)
    cam = mujoco.MjvCamera()
    # FIXED side camera (does NOT track the robot) so forward translation is visible across the strip
    cam.distance, cam.azimuth, cam.elevation = 3.0, 90, -12
    cam.lookat[:] = [0.45, 0.0, 0.2]
    horizon = 2400
    every = max(1, horizon // args.frames)
    shots: list = []

    def grab(m, d, s):
        if s % every == 0 and len(shots) < args.frames:
            renderer.update_scene(d, cam)
            shots.append(renderer.render().copy())

    res = run_locomotion_episode(model, horizon=horizon, on_step=grab)
    if not shots:
        print("no frames captured"); return 1
    strip = np.concatenate(shots[: args.frames], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(args.out)
    print(f"{args.prompt!r}: status={res['status']} forward={res['forward_m']}m dist={res['distance_m']}m "
          f"upright={res['upright']} -> {len(shots)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
