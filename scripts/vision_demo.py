"""Tiny-vision demo, end-to-end: train the encoder to SEE the goal, then drive a robot by that learned perception.

    python scripts/vision_demo.py

Prints the learn-to-see metric and the navigation result, and writes build/vision_demo/learned_vision_pov.png --
the robot's-eye montage (search -> acquire -> home in). The 'small CV model' is a 2-conv CNN @ 16x16 (~24K
params), trained CPU-fast (the Squint recipe); at-scale GPU vision-RL is the next leg (needs Madrona-MJX in an
isolated jax<0.6 env). See services/vision_encoder.py, vision_train.py, vision_nav.py.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from virturoid.services.vision_nav import run_vision_nav_episode, save_vision_montage  # noqa: E402
from virturoid.services.vision_train import make_learned_bearing_fn, train_goal_seer  # noqa: E402


def main() -> int:
    print("[1/3] Training the tiny encoder to SEE the goal (supervised, CPU)...")
    params, report = train_goal_seer(n=400, epochs=80)
    print(f"      bearing MAE {report['test_mae']:.4f} ({report['test_mae_deg']}deg) vs "
          f"{report['baseline_mae']:.4f} untrained -> {report['improvement_x']}x  (learned={report['learned']})")

    print("[2/3] Driving the robot to the goal on LEARNED perception, 4 starts...")
    bearing_fn = make_learned_bearing_fn(params)
    reached, montage_frames = 0, None
    for yaw, name in [(0.0, "ahead"), (math.pi / 2, "left"), (-math.pi / 2, "right"), (math.pi, "behind")]:
        r = run_vision_nav_episode(goal_xy=(3.0, 0.0), start_yaw=yaw, bearing_fn=bearing_fn,
                                   horizon=500, record_frames=(name == "left"))
        reached += r["reached"]
        if name == "left":
            montage_frames = r["frames"]
        print(f"      start {name:6s}: reached={r['reached']} dist={r['final_distance_m']}m steps={r['steps']}")
    print(f"      LEARNED-perception nav: {reached}/4 reached")

    print("[3/3] Saving the robot's-eye montage...")
    out = save_vision_montage(montage_frames, Path("build/vision_demo/learned_vision_pov.png"))
    print(f"      {out}")
    return 0 if reached == 4 and report["learned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
