"""CPU diagnosis: why does a composed 7-DOF arm fail task execution while a 6-DOF works? Drive the
scripted PD to an IK target on each and measure how close the end-effector dynamically gets to the box
(the IK target itself is reachable ~0.003). If the dynamic ee-box distance stays large, the long chain
isn't being tracked by the scripted PD — the case for a learned controller (research before GPU RL).

    PYTHONPATH=src python scripts/highdof_base_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    import numpy as np
    import mujoco

    from virturoid.services.morphology_composer import compose_robot
    from virturoid.services.gene_compiler import compile_gene_with_scene
    from virturoid.services.pick_place_controller import (
        plan_joint_targets, _actuated_joint_map, _actuator_force_clamps, _ee_site_id, _ee_position)
    from virturoid.schemas.scenes import SceneObject

    for label, prompt in [("3dof", "sort blocks on a table"),
                          ("6dof", "a precise 6-dof industrial arm"),
                          ("7dof", "a Franka-style 7-dof arm for precise insertion")]:
        g = compose_robot(prompt)
        cube = SceneObject("box", "cube", (0.4, 0.0, 0.05, 0, 0, 0), mass_kg=0.03, scale=1.0)
        mj = mujoco.MjModel.from_xml_string(compile_gene_with_scene(g, [cube]))
        ee = _ee_site_id(mj)
        act = _actuated_joint_map(mj)
        clamps = _actuator_force_clamps(mj)
        bq = int(mj.jnt_qposadr[mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, "free_box")])
        q, ik_res = plan_joint_targets(mj, (0.4, 0.0, 0.06))
        d = mujoco.MjData(mj); mujoco.mj_resetData(mj, d)
        kp, kd = 30.0, 4.0
        snaps = []
        for t in range(600):
            for slot, (qadr, vadr, jnt) in enumerate(act):
                tgt = q[slot] if slot < len(q) else 0.0
                tau = d.qfrc_bias[vadr] + kp * (tgt - d.qpos[qadr]) - kd * d.qvel[vadr]
                d.ctrl[slot] = float(np.clip(tau, -clamps[slot], clamps[slot]))
            mujoco.mj_step(mj, d)
            if t in (150, 300, 599):
                box = d.qpos[bq:bq + 3]
                snaps.append(round(float(np.linalg.norm(_ee_position(d, ee) - box)), 3))
        n_act = len(act)
        print(f"{label}: actuated={n_act} IK_residual={ik_res:.3f}  dynamic ee-box dist @[150,300,599]={snaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
