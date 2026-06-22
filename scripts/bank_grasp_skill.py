"""Bank the trained grasp skill (the real SKILL_CARD from the GPU save-run) into Virturoid Memory and
verify the flywheel read path (recall + warm-start selection). End-to-end proof of M3-G4.

    PYTHONPATH=src python scripts/bank_grasp_skill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# The exact SKILL_CARD emitted by `scripts/mjx_residual_grasp.py --save` on the GPU box (grasped 92%).
CARD = {
    "task_type": "grasp", "robot_class": "manipulator",
    "species": "fixed_arm.three_dof.tabletop.gripper",
    "gene_id": "gene_fixed_arm_three_dof_tabletop__gripper", "success_rate": 0.9219,
    "region": {"x": [0.43, 0.49], "y": [-0.08, 0.08]},
    "base_config": {"kpj": 120.0, "kdj": 12.0, "kpf": 300.0, "kdf": 10.0, "alpha": 0.3, "dq": 0.1,
                    "phases": [0.15, 0.5, 0.65], "ep_len": 200, "fclose": 0.045},
    "reward_spec": {"controlled_lift": 10.0, "both_contact": 1.0, "speed_penalty": 0.3,
                    "d_tcp_shaping": 0.15, "grasped_bonus": 6.0,
                    "success_gate": "both & lifted>0.05 & d_tcp<0.06"},
    "obs_dim": 34, "act_dim": 5,
    "params_path": "build/skills/grasp_tabletop_arm.npz",   # lives on the GPU box where warm-start runs
}


def main() -> int:
    from virturoid.services.memory_db import MemoryDB
    from virturoid.services.skill_flywheel import bank_skill_from_card, warm_start_for

    with MemoryDB() as db:   # canonical build/memory/virturoid_memory.db
        sid = bank_skill_from_card(db, CARD, notes="M3-G2 residual grasp, 192 envs, full band")
        print(f"banked skill_id = {sid}")
        print(f"skills in bank  = {db.stats()['skills']}")

        sk = db.recall_skill("manipulator", "grasp", obs_dim=34, act_dim=5)
        print(f"recall_skill    -> {sk['skill_id']} success={sk['success_rate']:.2f} "
              f"region={sk['region']} params={sk['params_path']}")

        # what would a NEW build warm-start from? (exact, IO-compatible -> direct params)
        ws = warm_start_for(db, "manipulator", "grasp", species="fixed_arm.three_dof.tabletop.gripper",
                            obs_dim=34, act_dim=5)
        print(f"warm_start_for  -> mode={ws['warm_start']} from={ws['params_path']}")

        # a RELATED body with no exact skill -> cross-body transfer candidate (the flywheel's reach)
        ws2 = warm_start_for(db, "humanoid", "grasp", species="humanoid.upper.arm.gripper",
                             obs_dim=40, act_dim=6)
        print(f"cross-body      -> {('none' if ws2 is None else ws2['warm_start'] + ' from ' + ws2['skill_id'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
