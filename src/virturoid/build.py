from __future__ import annotations

import argparse
import json
from pathlib import Path

from virturoid.services.autonomous_builder import build_robot_package_from_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Virturoid robot package from a task prompt.")
    parser.add_argument(
        "--prompt",
        default="Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
        help="Robot/task prompt.",
    )
    parser.add_argument(
        "--output",
        default=str(Path("build") / "autonomous_robot"),
        help="Output directory for the generated package.",
    )
    parser.add_argument("--payload-kg", type=float, default=None, help="Optional payload override in kilograms.")
    parser.add_argument("--reach-m", type=float, default=None, help="Optional reach override in meters.")
    parser.add_argument("--sensor", default=None, help="Optional sensor requirement, e.g. rgbd_camera or lidar.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="For supported robot classes, train and export an inference-ready controller bundle.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run a real MuJoCo pick-and-place sorting evaluation and cluster task failures (requires mujoco).",
    )
    parser.add_argument(
        "--co-optimize",
        dest="co_optimize",
        action="store_true",
        help="Co-optimize robot hardware against real task success for supported robot classes (requires mujoco).",
    )
    parser.add_argument(
        "--perceive",
        action="store_true",
        help="Run the task evaluation with perception in the loop (camera-estimated object poses, not privileged sim state).",
    )
    args = parser.parse_args()

    result = build_robot_package_from_prompt(
        prompt=args.prompt,
        output_dir=Path(args.output),
        payload_kg=args.payload_kg,
        reach_m=args.reach_m,
        sensor=args.sensor,
        train=args.train,
        evaluate=args.evaluate,
        co_optimize=args.co_optimize,
        perceive=args.perceive,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
