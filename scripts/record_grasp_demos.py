"""Record an expert-demonstration dataset from the banked grasp skill (§41) and fit an IL baseline.

The learned grasp policy is the expert: roll it out over randomized in-region scenes, keep the
successful episodes, and write a §41.3 demonstration dataset + dataset card. Then fit a baseline
behavior-cloning regressor to prove the dataset is learnable (§12.1). Closes the loop
learned-skill → demonstrations → imitation policy.

    PYTHONPATH=src python scripts/record_grasp_demos.py --n 16 --out build/skills/demos
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="grasp")
    ap.add_argument("--robot-class", default="manipulator")
    ap.add_argument("--params", default="build/skills/grasp_tabletop_arm.npz")
    ap.add_argument("--out", default="build/skills/demos")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args(argv)

    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.demonstration_dataset import (
        record_grasp_demonstrations, write_demonstration_dataset, fit_bc_baseline)

    with MemoryDB() as db:
        skill = db.recall_skill(args.robot_class, args.task)
        if skill is None:
            print(f"no banked {args.task} skill; bank one first."); return 1
        eps = record_grasp_demonstrations(skill, params_path=args.params, n=args.n)
        if not eps:
            print("no successful demonstrations recorded."); return 1
        card = write_demonstration_dataset(
            eps, args.out, robot=skill.get("species", "manipulator"), task=args.task,
            obs_modalities=["proprioception", "object_pose"], action_space="joint_torque[5]")
    bc = fit_bc_baseline(args.out)
    print(json.dumps({"episodes": card.episodes, "success_rate": card.success_rate,
                      "transitions": card.total_transitions, "formats": card.formats,
                      "bc_baseline": bc}, indent=2))
    print(f"dataset card -> {Path(args.out) / 'dataset_card.json'} (valid: {card.validate().ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
