"""Evaluate a banked learned skill the §32.5 way (held-out, reproducible) and emit the §12.3 outputs.

Pulls the trained grasp skill from Virturoid Memory (the flywheel), runs the unified evaluation suite
on CPU MuJoCo, writes the Policy Card + evaluation episodes + summary, and stamps the bank with the
honest held-out success. This is the bridge between "trained a skill on the GPU" and "evaluated it the
same way as every other policy" (the §32.5 antidote to train-metric self-reporting).

    PYTHONPATH=src python scripts/evaluate_grasp_skill.py \
        --params build/skills/grasp_tabletop_arm.npz --out build/skills/eval
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
    ap.add_argument("--out", default="build/skills/eval")
    ap.add_argument("--baseline-n", type=int, default=3)
    ap.add_argument("--randomized-n", type=int, default=10)
    args = ap.parse_args(argv)

    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.skill_evaluator import evaluate_and_record

    with MemoryDB() as db:
        skill = db.recall_skill(args.robot_class, args.task)
        if skill is None:
            print(f"no banked {args.task} skill for {args.robot_class}; bank one first "
                  "(scripts/bank_grasp_skill.py)."); return 1
        card, summary = evaluate_and_record(skill, args.out, memory=db, params_path=args.params,
                                            baseline_n=args.baseline_n, randomized_n=args.randomized_n)
    print(json.dumps(summary, indent=2))
    print(f"policy card -> {Path(args.out) / 'policy_card.json'}  (validate ok: {card.validate().ok})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
