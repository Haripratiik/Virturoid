"""Vision navigation: a robot reaches a goal it can SEE, using only its onboard camera.

The visual analogue of the range-sensing navigator ([[product-perception]]): instead of a rangefinder ring,
the robot has a forward camera, detects the colored goal in its image, and drives toward it -- searching by
rotating when the goal is out of frame. Sensed-only: the controller never reads the goal's true coordinates,
only what the camera sees (the ground-truth goal xy is used solely to score "reached"). CPU-first with MuJoCo's
camera renderer; the same loop is what a GPU-trained TinyVisionEncoder policy (see [[vision-cv-decision]])
would replace the hand-written color detector with.
"""
from __future__ import annotations

import math


def _yaw_quat(yaw: float):
    import numpy as np

    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=np.float64)


def detect_goal_bearing(img, target_rgb, *, tol: float = 0.28, min_frac: float = 0.0025):
    """Find the goal-colored region in an (H, W, 3) image -> (bearing, visible_frac).

    bearing is the goal's horizontal offset in the frame, -1 (hard left) .. +1 (hard right), or None when the
    goal is not meaningfully visible. Pure color match (no learning) -- the stand-in for a trained encoder."""
    import numpy as np

    frame = np.asarray(img, dtype=np.float32) / 255.0
    target = np.asarray(target_rgb, dtype=np.float32)
    mask = np.abs(frame - target).max(axis=2) < tol            # pixels close to the goal color
    frac = float(mask.mean())
    if frac < min_frac:
        return None, frac
    cols = np.nonzero(mask)[1]
    bearing = (float(cols.mean()) / frame.shape[1] - 0.5) * 2.0
    return float(np.clip(bearing, -1.0, 1.0)), frac


def build_vision_scene(goal_xy, goal_rgb=(0.1, 0.85, 0.15), obstacles=(), *, render_px: int = 64) -> str:
    """A lit floor + a mocap robot body carrying a forward camera + a brightly colored goal box (+ obstacles)."""
    gx, gy = goal_xy
    gr, gg, gb = goal_rgb
    obs = "".join(
        f'<geom type="cylinder" pos="{ox:.3f} {oy:.3f} 0.3" size="0.25 0.3" rgba="0.55 0.42 0.35 1"/>'
        for ox, oy in obstacles)
    return f"""<mujoco>
  <visual>
    <global offwidth="{render_px}" offheight="{render_px}"/>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <light pos="{gx * 0.5:.3f} {gy * 0.5:.3f} 6" dir="0 0 -1" diffuse="1 1 1"/>
    <geom type="plane" size="25 25 0.1" rgba="0.45 0.45 0.5 1"/>
    <body name="robot" mocap="true" pos="0 0 0.2">
      <geom type="box" size="0.18 0.14 0.12" rgba="0.2 0.3 0.85 1"/>
      <camera name="robot_cam" pos="0.19 0 0.07" xyaxes="0 -1 0 0 0 1" fovy="72"/>
    </body>
    <geom name="goal" type="box" pos="{gx:.3f} {gy:.3f} 0.35" size="0.25 0.25 0.35" rgba="{gr:.3f} {gg:.3f} {gb:.3f} 1"/>
    {obs}
  </worldbody>
</mujoco>"""


def run_vision_nav_episode(goal_xy=(3.0, 0.0), goal_rgb=(0.1, 0.85, 0.15), *, start=(0.0, 0.0),
                           start_yaw=0.0, obstacles=(), horizon: int = 500, v_max: float = 0.035,
                           turn_gain: float = 1.4, search_rate: float = 0.07, reach_m: float = 0.7,
                           render_px: int = 64, bearing_fn=None, record_frames: bool = False,
                           frame_every: int = 12):
    """Drive a differential-drive robot to a goal it can SEE. Returns reached / final_distance_m / steps.

    bearing_fn(frame) -> (bearing|None, frac) overrides the hand-written color detector -- pass a LEARNED
    perception (a trained TinyVisionEncoder predictor) to drive the robot end-to-end on what it learned to see.
    record_frames keeps every frame_every-th camera frame (the robot's-eye view) in the returned ``frames``."""
    _detect = bearing_fn if bearing_fn is not None else (lambda f: detect_goal_bearing(f, goal_rgb))
    frames: list = []
    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(build_vision_scene(goal_xy, goal_rgb, obstacles, render_px=render_px))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=render_px, width=render_px)   # one renderer, reused every step

    x, y, yaw = float(start[0]), float(start[1]), float(start_yaw)
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    dist = math.hypot(gx - x, gy - y)
    reached, seen_ever, step = False, False, 0
    last_bearing, lost = 0.0, 999
    try:
        for step in range(horizon):
            data.mocap_pos[0] = [x, y, 0.2]
            data.mocap_quat[0] = _yaw_quat(yaw)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="robot_cam")
            frame = renderer.render()
            if record_frames and step % frame_every == 0:
                frames.append(frame.copy())
            bearing, _frac = _detect(frame)
            if bearing is not None:
                seen_ever, last_bearing, lost = True, bearing, 0
            else:
                lost += 1
            if lost == 0:                                       # goal in view: center it (capped) + drive
                turn = max(-0.15, min(0.15, -turn_gain * bearing))
                v = v_max * (1.0 - 0.55 * min(1.0, abs(bearing)))
            elif lost < 25:                                     # just lost it: keep re-centering, creep forward
                turn = max(-0.12, min(0.12, -turn_gain * last_bearing))
                v = 0.35 * v_max
            else:                                               # blind: sweep toward the side last seen
                turn = -search_rate if last_bearing >= 0 else search_rate
                v = 0.0
            yaw += turn
            x += v * math.cos(yaw)
            y += v * math.sin(yaw)
            dist = math.hypot(gx - x, gy - y)
            if dist < reach_m:
                reached = True
                break
    finally:
        renderer.close()
    return {"reached": reached, "final_distance_m": round(dist, 3), "steps": step + 1,
            "goal_was_seen": seen_ever, "sensed_only": True, "frames": frames}


def save_vision_montage(frames, out_path, *, upscale: int = 4, max_cols: int = 8) -> str:
    """Save a horizontal strip of the robot's-eye camera frames (a vision-in-action montage) as a PNG."""
    from PIL import Image  # noqa: PLC0415 - optional, demo-only

    if not frames:
        raise ValueError("no frames to montage (run with record_frames=True)")
    picks = frames[:: max(1, len(frames) // max_cols)][:max_cols]
    tiles = [Image.fromarray(f).resize((f.shape[1] * upscale, f.shape[0] * upscale), Image.NEAREST) for f in picks]
    w, h, pad = tiles[0].width, tiles[0].height, 4
    strip = Image.new("RGB", (len(tiles) * w + (len(tiles) - 1) * pad, h), (16, 16, 20))
    for i, t in enumerate(tiles):
        strip.paste(t, (i * (w + pad), 0))
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    strip.save(out_path)
    return str(out_path)


def evaluate_vision_nav(*, goal_x: float = 3.0, render_px: int = 64) -> dict:
    """A few starts (goal ahead, and goal off to each side so the robot must search by rotating)."""
    starts = [((0.0, 0.0), 0.0), ((0.0, 0.0), math.pi / 2), ((0.0, 0.0), -math.pi / 2), ((0.0, 0.0), math.pi)]
    runs = [run_vision_nav_episode(goal_xy=(goal_x, 0.0), start=s, start_yaw=yw, render_px=render_px)
            for s, yw in starts]
    reached = sum(1 for r in runs if r["reached"])
    return {"success_rate": round(reached / len(runs), 3), "reached": reached, "n_starts": len(runs),
            "runs": runs, "sensed_only": True}
