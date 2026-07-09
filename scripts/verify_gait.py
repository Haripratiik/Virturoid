"""Honest gait verifier — the discipline that stops over-claiming a "walk" from a handful of frames.

Deploys a trained MorphPolicy via the DEPLOY-ALIGNED recipe rollout (the same drive the bank/render use),
captures qpos, replays it to a dense GIF + a fixed-camera side grid, and prints the anti-Goodhart metrics
with an explicit verdict. A credible WALK requires ALL of: survived (didn't fall) + upright at the body's
true stance height (not a crouch) + real foot-lift cadence + genuine stepping support + forward travel.
A body that stands still scores cadence 0 -> SLIDE; a low unstable body scores low upright_frac -> CROUCH.

Usage:
  python scripts/verify_gait.py <gene.json> <policy.npz> <out_dir> [tag] [--steps N] [--no-gif]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def verify_gait(gene, policy, *, steps: int = 1500, frame_every: int = 5, real_actuator: bool = True) -> dict:
    """Run the deploy-aligned rollout and return the honest metrics (+ qpos_frames when replayable)."""
    from virturoid.services.morph_policy import recipe_rollout_morph
    return recipe_rollout_morph(gene, policy, steps=steps, record_qpos=True, frame_every=frame_every,
                                real_actuator=real_actuator)


# classify + orientation_summary now live in the PACKAGE so the product (ai_native_tools) never imports scripts/.
# Re-exported here so this CLI and existing `from scripts.verify_gait import classify` callers keep working.
from virturoid.services.gait_quality import classify, orientation_summary  # noqa: E402,F401


def render_gif(gene, qpos_frames, out_dir: str, tag: str) -> None:
    import mujoco
    import numpy as np
    import PIL.Image

    from virturoid.services.gene_compiler import compile_gene_to_mjcf, standing_spawn_z
    xml = compile_gene_to_mjcf(gene, include_floor=True, spawn_z=standing_spawn_z(gene))
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    rr = mujoco.Renderer(m, height=480, width=640)
    gif, grid_imgs = [], []
    grid_idx = set(np.linspace(len(qpos_frames) * 0.25, len(qpos_frames) * 0.75, 12).astype(int))
    for i, qp in enumerate(qpos_frames):
        d.qpos[:] = qp
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [float(qp[0]), float(qp[1]), 0.18]
        cam.distance, cam.azimuth, cam.elevation = 1.8, 130, -12
        rr.update_scene(d, camera=cam)
        gif.append(PIL.Image.fromarray(rr.render().copy()))
        if i in grid_idx and len(grid_imgs) < 12:
            c2 = mujoco.MjvCamera()
            c2.lookat[:] = [float(qp[0]), 0.0, 0.18]; c2.distance = 1.7; c2.azimuth = 90; c2.elevation = -6
            rr.update_scene(d, camera=c2)
            grid_imgs.append(PIL.Image.fromarray(rr.render().copy()))
    rr.close()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if gif:
        gif[0].save(os.path.join(out_dir, f"{tag}_walk.gif"), save_all=True,
                    append_images=gif[1:], duration=70, loop=0)
    if grid_imgs:
        w, h = grid_imgs[0].size
        grid = PIL.Image.new("RGB", (w * 4, h * 3), (28, 32, 38))
        for i, im in enumerate(grid_imgs):
            grid.paste(im, ((i % 4) * w, (i // 4) * h))
        grid.save(os.path.join(out_dir, f"{tag}_grid.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gene_json"); ap.add_argument("policy_npz"); ap.add_argument("out_dir")
    ap.add_argument("tag", nargs="?", default="gait")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "glfw")

    from virturoid.schemas.gene import RobotGene
    from virturoid.services.morph_policy import MorphPolicy
    gene = RobotGene.from_dict(json.loads(Path(a.gene_json).read_text()))
    policy = MorphPolicy.from_npz(a.policy_npz)
    r = verify_gait(gene, policy, steps=a.steps)
    print(f"[{a.tag}] {len(gene.segments)} segs / {len(gene.actuated_joints())} act")
    for k in ("survived", "forward", "speed", "height_ratio", "upright_frac", "upright_tau",
              "cadence", "support_frac", "alive"):
        print(f"  {k:14s}: {r.get(k)}")
    o = orientation_summary(r.get("qpos_frames") or [])
    if o:
        print(f"  orientation   : roll {o['roll_max']:.0f}deg max / pitch {o['pitch_max']:.0f}deg max / "
              f"yaw {o['yaw_max']:.0f}deg max  (final r{o['roll_final']:.0f} p{o['pitch_final']:.0f} y{o['yaw_final']:.0f})")
    print(f"  VERDICT: {classify(r)}")
    if not a.no_gif and r.get("qpos_frames"):
        render_gif(gene, r["qpos_frames"], a.out_dir, a.tag)
        print(f"  wrote {a.tag}_walk.gif + {a.tag}_grid.png in {a.out_dir}")


if __name__ == "__main__":
    main()
