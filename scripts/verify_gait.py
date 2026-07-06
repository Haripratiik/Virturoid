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


def orientation_summary(qpos_frames) -> dict:
    """Max + final roll/pitch/yaw (deg) from the base quaternion trace. The anti-Goodhart signal for a
    LEGGED body's true failure mode: a trot that loses lateral balance ROLLS over, an over-driven gait
    PITCH-dives over its front legs, an asymmetric gait YAW-drifts off a straight line. Height alone can't
    tell these apart (all three end with a low base) — the classifier needs the orientation to name the fall."""
    import numpy as np
    if not qpos_frames:
        return {}
    R = P = Y = 0.0; rr = pp = yy = 0.0
    for f in qpos_frames:
        w, x, y, z = (float(f[3]), float(f[4]), float(f[5]), float(f[6]))
        rr = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        pp = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        yy = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        R = max(R, abs(rr)); P = max(P, abs(pp)); Y = max(Y, abs(yy))
    return {"roll_max": round(R, 1), "pitch_max": round(P, 1), "yaw_max": round(Y, 1),
            "roll_final": round(rr, 1), "pitch_final": round(pp, 1), "yaw_final": round(yy, 1)}


def classify(r: dict) -> str:
    """Explicit, honest verdict from the anti-Goodhart metrics + orientation fall-mode."""
    survived = bool(r.get("survived"))
    up = float(r.get("upright_frac", 0.0)); hr = float(r.get("height_ratio", 0.0))
    cad = float(r.get("cadence", 0.0)); sup = float(r.get("support_frac", 0.0))
    fwd = float(r.get("forward", 0.0))
    if survived and up >= 0.6 and cad >= 1.0 and sup >= 0.25 and fwd >= 0.3:
        return "CREDIBLE WALK"
    # not a credible walk: name the DOMINANT fall mode from the orientation trace (if replayable) so the
    # verdict is diagnostic, not just "bad" — a roll-over reads as CROUCH by height alone, which misleads.
    o = orientation_summary(r.get("qpos_frames") or [])
    if o:
        modes = [("ROLL-OVER", o["roll_max"]), ("PITCH-DIVE", o["pitch_max"]), ("YAW-DRIFT", o["yaw_max"])]
        name, val = max(modes, key=lambda t: t[1])
        if not survived and val >= 45.0:
            return f"FELL by {name} (roll {o['roll_max']:.0f} / pitch {o['pitch_max']:.0f} / yaw {o['yaw_max']:.0f} deg max)"
    if up < 0.6 or hr < 0.6:
        return "CROUCH (low/unstable stance)"
    if cad < 1.0 or sup < 0.25:
        return "SLIDE (feet barely lift / no real stepping)"
    return "FORWARD BUT SHORT / not a credible walk"


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
