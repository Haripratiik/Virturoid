"""Autonomous robot build entrypoint.

`python -m virturoid.autobuild --prompt "..."` runs the full closed loop and
prints a verdict on whether the software built a robot that performs the task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from virturoid.services.autonomous_build import autonomous_build
from virturoid.services.memory_store import DEFAULT_MEMORY_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomously build a robot from a task prompt.")
    parser.add_argument(
        "--prompt",
        default="Build a tabletop robot arm that sorts red and blue blocks into matching bins.",
        help="Robot/task prompt.",
    )
    parser.add_argument("--output", default=str(Path("build") / "autonomous_build"))
    parser.add_argument("--target-success", type=float, default=0.8, help="Task success rate required to declare success.")
    parser.add_argument("--train", action="store_true", help="Also train and export a controller for the converged robot.")
    parser.add_argument("--rounds", type=int, default=3, help="Max improvement rounds (hardware + controller co-optimization).")
    parser.add_argument("--memory-dir", default=str(DEFAULT_MEMORY_DIR), help="Directory for build memory.")
    args = parser.parse_args()

    report = autonomous_build(
        prompt=args.prompt,
        output_dir=Path(args.output),
        target_success_rate=args.target_success,
        train=args.train,
        memory_dir=Path(args.memory_dir),
        max_rounds=args.rounds,
    )

    print(
        json.dumps(
            {
                "autonomous_build_succeeded": report.succeeded,
                "robot_class": report.robot_class,
                "species": report.species,
                "task_type": report.task_type,
                "feasible": report.feasible,
                "initial_success_rate": report.initial_success_rate,
                "final_success_rate": report.final_success_rate,
                "target_success_rate": report.target_success_rate,
                "converged_design": report.converged_design,
                "decisions": [
                    {"stage": d.stage, "action": d.action, "success_after": d.success_after} for d in report.decisions
                ],
                "memory_reused": report.memory_reused,
                "compute_provenance": report.compute,
                "output_dir": args.output,
                "autonomy_report_json": str(Path(args.output) / "reports" / "autonomy_report.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
